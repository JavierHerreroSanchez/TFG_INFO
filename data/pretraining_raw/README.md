# Estructura de `data/pretraining_raw`

Esta carpeta almacena los corpus MIDI originales usados en la fase de preentrenamiento. El contenido no se versiona por tamano y por tratarse de datos externos.

Estructura prevista:

```text
data/pretraining_raw/
├── maestro-v3.0.0/
│   ├── 2004/
│   ├── 2006/
│   └── ...
└── ariamidi/
    ├── aa/
    ├── ab/
    └── ...
```

En MAESTRO, la organizacion sigue las carpetas por ano del propio dataset. En ARIAMidi, la organizacion local usa carpetas alfabeticas de dos letras; se muestran solo algunos ejemplos porque el conjunto completo contiene muchas subcarpetas equivalentes.
