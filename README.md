# TFG_INFO

Repositorio asociado al Trabajo de Fin de Grado sobre generación musical simbólica
mediante modelos Transformer. El objetivo del proyecto es construir un flujo de
procesado, entrenamiento, finetuning y evaluación capaz de generar secuencias
musicales en formato MIDI, con especial interés en repertorio pianístico y formas
musicales clásicas.

## Estructura del proyecto

- `src/model`: definición del modelo Music Transformer autorregresivo.
- `src/tokenization`: construcción de índices y tokenización de corpus MIDI.
- `src/training`: scripts de preentrenamiento y utilidades de caché binaria.
- `src/finetuning`: limpieza, aumento de datos y ajuste fino sobre el corpus objetivo.
- `src/generation`: generación de muestras y conversión de salidas tokenizadas a MIDI.
- `src/evaluation`: evaluación simbólica y espectral de las piezas generadas.
- `data`: índices, datos intermedios y cachés locales de entrenamiento.
- `output`: checkpoints, muestras generadas e informes de evaluación.
- `tokenizer`: tokenizadores BPE entrenados para la representación simbólica.

## Flujo principal

1. Construcción de índices del corpus MIDI.
2. Tokenización de las piezas y conversión a secuencias de ids.
3. Preentrenamiento del modelo Transformer sobre el corpus tokenizado.
4. Fine-tuning sobre el subconjunto musical objetivo.
5. Generación de nuevas secuencias y conversión a MIDI.
6. Evaluación simbólica y espectral de las muestras generadas.

## Requisitos

Las dependencias Python se recogen en `requirements.txt`. El proyecto está
preparado para ejecutarse en entorno local y usa CUDA cuando PyTorch detecta una
GPU disponible.

```bash
pip install -r requirements.txt
```

## Nota

Para la lectura del código principal conviene partir
de `src/model`, `src/training/pretraining_v2.py`, `src/finetuning/finetuning_v2.py`,
`src/generation` y `src/evaluation`.
