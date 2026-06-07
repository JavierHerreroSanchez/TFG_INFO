# TFG_INFO

Trabajo de Fin de Grado sobre generación musical con modelos Transformer. El
repositorio contiene el pipeline de preparación de datos, entrenamiento,
generación y evaluación.

## Requisitos

```bash
pip install -r requirements.txt
```

Ejecutar desde la raíz del proyecto:

```bash
python -m src.<paquete>.<script>
```

## Estructura

```text
src/
|-- model/              # Modelo y bloques Transformer
|-- tokenization/       # Tokenización REMI+BPE, attribute controls e índices
|-- pretraining/        # Preentrenamiento
|-- finetuning/         # Limpieza, aumento de datos y ajuste fino
|-- generation/         # Generación y JSON -> MIDI
|-- evaluation/         # Métricas y gráficos
|-- inspection/         # Utilidades de revisión
`-- musical_analisis/   # Figuras de análisis musical
```

## Datos

Los datasets y los bins de entrenamiento no se proporcionan por su tamaño.

```text
data/
|-- pretraining_raw/
|   |-- maestro-v3.0.0/
|   `-- ariamidi/
|-- finetuning/
|   |-- finetuning_sonatas_raw/
|   |-- finetuning_sonatas_clean/
|   `-- finetuning_sonatas_aug/
|-- finetuning_v2/
|   |-- mozart_sonatas_raw/
|   |-- mozart_sonatas_merged/
|   `-- mozart_sonatas_aug/
|-- interim/
`-- bin/
```

Las carpetas excluidas de Git incluyen un `README.md` con su estructura local.

## Flujo Principal

### 1. Tokenizador

```bash
python -m src.tokenization.tokenizer_train
```

Tokenizador usado: `tokenizer/tokenizer_REMI_BPE_v2.json`.

### 2. Preentrenamiento

```bash
python -m src.tokenization.tokenize_pretraining_v2
python -m src.pretraining.pretraining_v2
```

Salidas principales:

```text
data/interim/tokenized_json_bpe_v2/
data/interim/indexes/index_pretraining_v2.csv
data/bin/bin_for_pretraining_v2/
output/checkpoints/pretraining_v2/
```

### 3. Fine-tuning

```bash
python -m src.finetuning.clean_finetuning_data_tfg
python -m src.finetuning.augment_finetuning_data
python -m src.tokenization.tokenize_finetuning_v2
python -m src.finetuning.finetuning_v2
```

Salidas principales:

```text
data/interim/tokenized_finetuning_v2/
data/interim/indexes/index_finetuning_v2.csv
data/bin/bin_for_finetuning_v2/
output/checkpoints/finetuning_v2/
```

### 4. Generación

```bash
python -m src.generation.generation_from_pretraining_v2
python -m src.generation.generation_from_finetuning_v2
python -m src.generation.generated_json_to_midi
```

Salidas principales:

```text
output/generation_pretraining_tfg_second/
output/generation_finetuning_tfg_second/
```

### 5. Evaluación

```bash
python -m src.evaluation.evaluate_generated_pretraining
python -m src.evaluation.evaluate_generated_finetuning
python -m src.evaluation.generate_spectogram_pretraining
python -m src.evaluation.generate_spectrograms_finetuning
python -m src.evaluation.generar_graficos_evaluacion
```

Los resultados se guardan en las carpetas `output/generation_*`.

## Artefactos

```text
tokenizer/                         tokenizadores entrenados
data/interim/indexes/*.csv         índices de tokenización
data/bin/                          caches binarias
output/checkpoints/*/best.pt       checkpoints/modelos principales
output/generation_*/               muestras generadas e informes de evaluación
```

## Reproducibilidad

Las rutas se resuelven desde la raíz del proyecto. Si se cambian datasets,
tokenizadores o checkpoints, deben regenerarse índices, caches y evaluaciones
siguiendo los pasos proporcionados.

## Licencia

Copyright (c) 2026 Javier Herrero Sánchez.

El código fuente original de este repositorio se distribuye bajo la
[GNU General Public License v3.0](LICENSE) (`GPL-3.0-only`).

Esta licencia no se aplica automáticamente a datasets, archivos MIDI,
composiciones, dependencias ni otros materiales de terceros. Dichos contenidos
conservan sus licencias y condiciones de uso originales.
