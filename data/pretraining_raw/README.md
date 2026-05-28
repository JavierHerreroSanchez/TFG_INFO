# Estructura de `data/pretraining_raw`

Corpus MIDI originales para preentrenamiento. No se versionan por tamano ni por
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

MAESTRO mantiene la organizacion por ano del dataset. ARIAMidi usa subcarpetas
alfabeticas de dos letras.
