"""
Exporta MIDIs generados a WAV/MP3 usando el SoundFont sgm_plus de Magenta.

El reproductor html-midi-player usa por defecto:
https://storage.googleapis.com/magentadata/js/soundfonts/sgm_plus

Este script no necesita FluidSynth ni un .sf2. Descarga en cache las muestras
MP3 necesarias del SoundFont de Magenta, las decodifica con ffmpeg y las mezcla
en audio PCM para generar archivos incrustables en LaTeX/PPTX.

Uso desde la raíz del proyecto:

    python -m src.musical_analisis.export_midi_audio
    python -m src.musical_analisis.export_midi_audio --format mp3
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pretty_midi
from scipy.io import wavfile


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[1]

DEFAULT_SOUNDFONT_URL = "https://storage.googleapis.com/magentadata/js/soundfonts/sgm_plus"
DEFAULT_SAMPLE_RATE = 44_100
DEFAULT_FADE_SECONDS = 0.1


@dataclass(frozen=True)
class RenderTarget:
    name: str
    midi_path: Path


DEFAULT_TARGETS = [
    RenderTarget(
        "pretraining_v1_g2",
        ROOT_DIR / "output" / "generation_pretraining_tfg_first" / "generated_from_json2.mid",
    ),
    RenderTarget(
        "pretraining_v2_g7",
        ROOT_DIR / "output" / "generation_pretraining_tfg_second" / "generated_from_json7.mid",
    ),
    RenderTarget(
        "finetuning_v2_g0",
        ROOT_DIR / "output" / "generation_finetuning_tfg_second" / "generated_from_json0.mid",
    ),
    RenderTarget(
        "finetuning_v2_g9",
        ROOT_DIR / "output" / "generation_finetuning_tfg_second" / "generated_from_json9.mid",
    ),
]


@dataclass(frozen=True)
class InstrumentSpec:
    name: str
    min_pitch: int
    max_pitch: int
    duration_seconds: float
    release_seconds: float
    percussive: bool
    velocities: tuple[int, ...] | None


@dataclass(frozen=True)
class MidiNote:
    pitch: int
    velocity: int
    start: float
    end: float
    program: int
    is_drum: bool

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


class MagentaSoundFontRenderer:
    def __init__(
        self,
        soundfont_url: str,
        cache_dir: Path,
        ffmpeg: str,
        sample_rate: int,
        fade_seconds: float = DEFAULT_FADE_SECONDS,
    ) -> None:
        self.soundfont_url = soundfont_url.rstrip("/")
        self.cache_dir = cache_dir
        self.ffmpeg = ffmpeg
        self.sample_rate = sample_rate
        self.fade_seconds = fade_seconds
        self.soundfont_spec = self._load_soundfont_spec()
        self.instrument_specs: dict[str, InstrumentSpec] = {}
        self.decoded_samples: dict[tuple[str, int, int | None], np.ndarray] = {}

    def render_midi(self, midi_path: Path) -> np.ndarray:
        notes = self._collect_notes(midi_path)
        if not notes:
            raise RuntimeError(f"No se encontraron notas en {midi_path}")

        tail_seconds = 5.0
        total_seconds = max(note.end for note in notes) + tail_seconds
        audio = np.zeros((math.ceil(total_seconds * self.sample_rate), 2), dtype=np.float32)

        for note in notes:
            instrument_name = self._instrument_name(note.program, note.is_drum)
            if instrument_name is None:
                continue

            spec = self._load_instrument_spec(instrument_name)
            if note.pitch < spec.min_pitch or note.pitch > spec.max_pitch:
                print(
                    f"Aviso: pitch {note.pitch} fuera de rango para {instrument_name} "
                    f"({spec.min_pitch}-{spec.max_pitch}); se omite.",
                    file=sys.stderr,
                )
                continue

            sample_velocity = self._nearest_velocity(note.velocity, spec.velocities)
            sample = self._load_sample(instrument_name, note.pitch, sample_velocity)
            self._mix_note(audio, sample, note, spec)

        return audio

    def _collect_notes(self, midi_path: Path) -> list[MidiNote]:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        notes: list[MidiNote] = []
        for instrument in pm.instruments:
            for note in instrument.notes:
                if note.velocity <= 0 or note.end <= note.start:
                    continue
                notes.append(
                    MidiNote(
                        pitch=int(note.pitch),
                        velocity=int(note.velocity),
                        start=float(note.start),
                        end=float(note.end),
                        program=int(instrument.program),
                        is_drum=bool(instrument.is_drum),
                    )
                )
        notes.sort(key=lambda item: (item.start, item.pitch, item.end))
        return notes

    def _load_soundfont_spec(self) -> dict:
        return self._load_json("soundfont.json")

    def _load_instrument_spec(self, instrument_name: str) -> InstrumentSpec:
        if instrument_name in self.instrument_specs:
            return self.instrument_specs[instrument_name]

        data = self._load_json(f"{instrument_name}/instrument.json")
        spec = InstrumentSpec(
            name=str(data["name"]),
            min_pitch=int(data["minPitch"]),
            max_pitch=int(data["maxPitch"]),
            duration_seconds=float(data["durationSeconds"]),
            release_seconds=float(data["releaseSeconds"]),
            percussive=bool(data["percussive"]),
            velocities=tuple(int(v) for v in data["velocities"]) if "velocities" in data else None,
        )
        self.instrument_specs[instrument_name] = spec
        return spec

    def _load_json(self, relative_path: str) -> dict:
        cache_path = self.cache_dir / relative_path
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))

        url = f"{self.soundfont_url}/{relative_path.replace(chr(92), '/')}"
        payload = self._download(url)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(payload)
        return json.loads(payload.decode("utf-8"))

    def _load_sample(self, instrument_name: str, pitch: int, velocity: int | None) -> np.ndarray:
        key = (instrument_name, pitch, velocity)
        if key in self.decoded_samples:
            return self.decoded_samples[key]

        sample_name = f"p{pitch}_v{velocity}.mp3" if velocity is not None else f"p{pitch}.mp3"
        relative_path = f"{instrument_name}/{sample_name}"
        cache_path = self.cache_dir / relative_path
        if not cache_path.exists():
            url = f"{self.soundfont_url}/{relative_path}"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(self._download(url))

        sample = self._decode_mp3(cache_path)
        self.decoded_samples[key] = sample
        return sample

    def _download(self, url: str) -> bytes:
        print(f"Descargando {url}")
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return response.read()
        except urllib.error.URLError as exc:
            raise RuntimeError(
                "No se pudo descargar el SoundFont de Magenta. "
                "Comprueba la conexion o reutiliza una cache ya descargada."
            ) from exc

    def _decode_mp3(self, path: Path) -> np.ndarray:
        cmd = [
            self.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ac",
            "2",
            "-ar",
            str(self.sample_rate),
            "pipe:1",
        ]
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        data = np.frombuffer(result.stdout, dtype=np.float32)
        if len(data) % 2:
            data = data[:-1]
        return data.reshape((-1, 2)).copy()

    def _instrument_name(self, program: int, is_drum: bool) -> str | None:
        instruments = self.soundfont_spec["instruments"]
        key = "drums" if is_drum else str(program)
        instrument = instruments.get(key)
        if instrument is None and not is_drum:
            print(
                f"Aviso: sgm_plus no tiene programa {program}; se omite ese instrumento.",
                file=sys.stderr,
            )
        return instrument

    @staticmethod
    def _nearest_velocity(velocity: int, velocities: tuple[int, ...] | None) -> int | None:
        if velocities is None:
            return None
        if velocity <= 0:
            velocity = 80
        return min(velocities, key=lambda item: abs(item - velocity))

    def _mix_note(
        self,
        audio: np.ndarray,
        sample: np.ndarray,
        note: MidiNote,
        spec: InstrumentSpec,
    ) -> None:
        start_sample = int(round(note.start * self.sample_rate))
        if spec.percussive or note.duration >= spec.duration_seconds:
            self._add(audio, sample, start_sample)
            return

        sustain_end = int(round((note.duration + self.fade_seconds) * self.sample_rate))
        sustain_end = max(1, min(sustain_end, len(sample)))
        sustain = sample[:sustain_end].copy()
        self._fade_out(sustain, self.fade_seconds)
        self._add(audio, sustain, start_sample)

        release_start = int(round(spec.duration_seconds * self.sample_rate))
        if release_start < len(sample):
            release = sample[release_start:].copy()
            self._fade_in(release, self.fade_seconds)
            self._fade_out(release, self.fade_seconds)
            self._add(audio, release, int(round(note.end * self.sample_rate)))

    def _add(self, audio: np.ndarray, segment: np.ndarray, start: int) -> None:
        if start >= len(audio) or len(segment) == 0:
            return
        if start < 0:
            segment = segment[-start:]
            start = 0
        end = min(start + len(segment), len(audio))
        audio[start:end] += segment[: end - start]

    def _fade_in(self, segment: np.ndarray, seconds: float) -> None:
        fade_len = min(int(round(seconds * self.sample_rate)), len(segment))
        if fade_len > 1:
            segment[:fade_len] *= np.linspace(0.0, 1.0, fade_len, dtype=np.float32)[:, None]

    def _fade_out(self, segment: np.ndarray, seconds: float) -> None:
        fade_len = min(int(round(seconds * self.sample_rate)), len(segment))
        if fade_len > 1:
            segment[-fade_len:] *= np.linspace(1.0, 0.0, fade_len, dtype=np.float32)[:, None]


def normalize_audio(audio: np.ndarray, peak_db: float | None) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= 0.0:
        return audio
    if peak_db is None:
        return np.clip(audio, -1.0, 1.0)

    target_peak = 10.0 ** (peak_db / 20.0)
    return np.clip(audio * (target_peak / peak), -1.0, 1.0)


def write_wav(path: Path, sample_rate: int, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    wavfile.write(path, sample_rate, pcm)


def write_mp3(ffmpeg: str, wav_path: Path, mp3_path: Path, bitrate: str) -> None:
    mp3_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(wav_path),
        "-codec:a",
        "libmp3lame",
        "-b:a",
        bitrate,
        str(mp3_path),
    ]
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--format",
        choices=("wav", "mp3", "both"),
        default="both",
        help="Formato de salida. Por defecto genera WAV y MP3.",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["all"],
        help="Nombres a exportar o 'all'. Opciones: " + ", ".join(t.name for t in DEFAULT_TARGETS),
    )
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--soundfont-url", default=DEFAULT_SOUNDFONT_URL)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT_DIR / "output" / "soundfont_cache" / "sgm_plus",
    )
    parser.add_argument("--wav-dir", type=Path, default=ROOT_DIR / "output" / "wav")
    parser.add_argument("--mp3-dir", type=Path, default=ROOT_DIR / "output" / "mp3")
    parser.add_argument("--mp3-bitrate", default="192k")
    parser.add_argument(
        "--normalize-peak-db",
        type=float,
        default=-1.0,
        help="Pico objetivo de normalizacion. Usa --no-normalize para desactivarlo.",
    )
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg"))
    return parser.parse_args()


def select_targets(names: Iterable[str]) -> list[RenderTarget]:
    names = list(names)
    if names == ["all"]:
        return DEFAULT_TARGETS

    by_name = {target.name: target for target in DEFAULT_TARGETS}
    unknown = [name for name in names if name not in by_name]
    if unknown:
        raise ValueError(f"Targets no reconocidos: {', '.join(unknown)}")
    return [by_name[name] for name in names]


def main() -> None:
    args = parse_args()
    if not args.ffmpeg:
        raise RuntimeError("No se encontro ffmpeg en PATH. Instalalo o pasa --ffmpeg <ruta>.")

    targets = select_targets(args.targets)
    missing = [target.midi_path for target in targets if not target.midi_path.exists()]
    if missing:
        raise FileNotFoundError("No se encontraron estos MIDI:\n" + "\n".join(str(p) for p in missing))

    renderer = MagentaSoundFontRenderer(
        soundfont_url=args.soundfont_url,
        cache_dir=args.cache_dir,
        ffmpeg=args.ffmpeg,
        sample_rate=args.sample_rate,
    )

    peak_db = None if args.no_normalize else args.normalize_peak_db
    for target in targets:
        print(f"\nRenderizando {target.name}: {target.midi_path}")
        audio = renderer.render_midi(target.midi_path)
        audio = normalize_audio(audio, peak_db)

        wav_path = args.wav_dir / f"{target.name}.wav"
        mp3_path = args.mp3_dir / f"{target.name}.mp3"
        if args.format in ("wav", "both"):
            write_wav(wav_path, args.sample_rate, audio)
            print(f"Guardado WAV: {wav_path}")

        if args.format in ("mp3", "both"):
            source_wav = wav_path
            if args.format == "mp3":
                source_wav = args.wav_dir / f".{target.name}.tmp.wav"
                write_wav(source_wav, args.sample_rate, audio)
            write_mp3(args.ffmpeg, source_wav, mp3_path, args.mp3_bitrate)
            print(f"Guardado MP3: {mp3_path}")
            if args.format == "mp3" and source_wav.exists():
                source_wav.unlink()


if __name__ == "__main__":
    main()
