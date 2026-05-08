from pathlib import Path
from miditok import REMI

tokenizer = REMI(params=Path(r"../../tokenizer/tokenizer_REMI_BPE_v5.json"))

tokens = tokenizer(Path(r"sonata08-1.mid"))

print(type(tokens))
if isinstance(tokens, list):
    print("N streams:", len(tokens))
    print([len(seq.ids) for seq in tokens])
else:
    print("Stream único:", len(tokens.ids))


"""from pathlib import Path
import pretty_midi

in_path = Path(r"sonata08-1.mid")
out_path = Path(r"sonata08-1_merged_piano.mid")

pm = pretty_midi.PrettyMIDI(str(in_path))

merged = pretty_midi.PrettyMIDI(initial_tempo=pm.estimate_tempo())
piano = pretty_midi.Instrument(program=0, is_drum=False, name="Merged Piano")

for inst in pm.instruments:
    if not inst.is_drum and inst.program == 0:
        piano.notes.extend(inst.notes)
        piano.control_changes.extend(inst.control_changes)
        piano.pitch_bends.extend(inst.pitch_bends)

piano.notes.sort(key=lambda n: (n.start, n.pitch, n.end))
piano.control_changes.sort(key=lambda c: c.time)
piano.pitch_bends.sort(key=lambda b: b.time)

merged.instruments.append(piano)
merged.write(str(out_path))"""