# Estructura de `data/pretraining_raw`

Corpus MIDI originales para preentrenamiento. No se versionan por tamaño ni por
ser datos externos.

```text
data/pretraining_raw/
|-- maestro-v3.0.0/
|   |-- 2004/
|   |-- 2006/
|   `-- ...
`-- ariamidi/
    |-- aa/
    |-- ab/
    `-- ...
```

MAESTRO mantiene la organización por año del dataset. ARIAMidi usa subcarpetas
alfabéticas de dos letras.
