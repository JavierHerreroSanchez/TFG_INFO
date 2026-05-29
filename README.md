# TFG_INFO

Trabajo de Fin de Grado sobre generacion musical simbolica con modelos
Transformer. El repositorio contiene el pipeline de preparacion de datos,
entrenamiento, generacion y evaluacion.

## Requisitos

```bash
pip install -r requirements.txt
```

Ejecutar desde la raiz del proyecto:

```bash
python -m src.<paquete>.<script>
```

## Estructura

```text
src/
|-- model/              # Modelo Transformer
|-- tokenization/       # Tokenizacion REMI+BPE e indices
|-- training/           # Preentrenamiento
|-- finetuning/         # Limpieza, aumentacion y ajuste fino
|-- generation/         # Generacion y JSON -> MIDI
|-- evaluation/         # Metricas y graficos
|-- inspection/         # Utilidades de revision
`-- musical_analisis/   # Figuras de analisis musical
```

## Datos

Los datasets y artefactos pesados no se versionan.

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
data/interim/tokenized_finetuning_v3/
data/interim/indexes/index_finetuning_v3.csv
data/bin/bin_for_finetuning_v3/
output/checkpoints/finetuning_v2/
```

### 4. Generacion

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

### 5. Evaluacion

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
data/interim/indexes/*.csv         indices de tokenizacion
data/bin/                          caches binarias
output/checkpoints/*/best.pt       checkpoints principales
output/generation_*/               muestras e informes
```

## Reproducibilidad

Las rutas se resuelven desde la raiz del proyecto. Si se cambian datasets,
tokenizadores o checkpoints, deben regenerarse indices, caches y evaluaciones.
