# TFG_INFO

Repositorio asociado al Trabajo de Fin de Grado sobre generacion musical
simbolica mediante modelos Transformer. El proyecto implementa un pipeline
completo para preparar corpus MIDI, entrenar un modelo autorregresivo, adaptar el
modelo a un dominio musical concreto, generar nuevas piezas y evaluarlas con
metricas simbolicas, espectrales y graficas.

El foco experimental esta en repertorio pianistico, con una fase de
preentrenamiento sobre corpus amplios y una fase posterior de fine-tuning sobre
un subconjunto objetivo de sonatas.

## Requisitos

Se recomienda crear un entorno virtual e instalar las dependencias desde la raiz
del repositorio:

```bash
pip install -r requirements.txt
```

## Datasets utilizados

Los datos pesados no se incluyen en el repositorio. Deben colocarse localmente
respetando la estructura indicada:

```text
data/
|-- pretraining_raw/
|   |-- maestro-v3.0.0/
|   `-- ariamidi/
|-- finetuning/
|   |-- finetuning_sonatas_raw/
|   |-- finetuning_sonatas_clean/
|   `-- finetuning_sonatas_aug/
`-- finetuning_v2/
    |-- mozart_sonatas_raw/
    |-- mozart_sonatas_merged/
    `-- mozart_sonatas_aug/
```

Para preentrenamiento se usan MAESTRO v3.0.0 y ARIAMidi. MAESTRO se organiza por
anio de concurso, mientras que ARIAMidi aparece dividido en subcarpetas
alfabeticas de dos letras. En los README internos de las carpetas excluidas se
documenta la estructura esperada sin enumerar archivos individuales.

Para fine-tuning se usan corpus locales de sonatas pianisticas. El primer flujo
trabaja con `data/finetuning`; el segundo flujo, usado como configuracion
principal en los scripts `*_v2`, trabaja con `data/finetuning_v2`.

## Organizacion del repositorio

```text
TFG_INFO/
|-- src/
|   |-- model/              # Arquitectura Music Transformer
|   |-- tokenization/       # Tokenizadores, indices y conversion MIDI -> JSON
|   |-- training/           # Preentrenamiento y caches binarias
|   |-- finetuning/         # Limpieza, aumentacion y ajuste fino
|   |-- generation/         # Generacion autoregresiva y conversion JSON -> MIDI
|   |-- evaluation/         # Evaluacion simbolica, espectral y graficos
|   |-- inspection/         # Utilidades de auditoria de tokens, MIDIs y checkpoints
|   `-- musical_analisis/   # Figuras de analisis musical de muestras concretas
|-- data/                   # Datos locales, tokenizaciones y caches binarios
|-- output/                 # Checkpoints, muestras generadas e informes
|-- tokenizer/              # Tokenizadores REMI+BPE entrenados
`-- requirements.txt
```

Las carpetas `data/pretraining_raw`, `data/finetuning`, `data/finetuning_v2`,
`data/interim/tokenized_*`, `data/bin` y algunas carpetas de muestras de
`output/checkpoints` estan excluidas de Git por tamano. Cada una contiene un
`README.md` breve con la estructura esperada.

## Flujo de reproduccion recomendado

La forma mas reproducible desde terminal es ejecutar los scripts como modulos
desde la raiz del repositorio (`TFG_INFO`):

```bash
python -m src.<modulo>.<script>
```

Esto es equivalente a ejecutarlos directamente desde PyCharm siempre que la
configuracion de ejecucion use `TFG_INFO` como working directory, o que PyCharm
anada la raiz del proyecto al `PYTHONPATH`. Esta suele ser la configuracion por
defecto al abrir el proyecto completo.

Por ejemplo, estas dos formas lanzan el mismo script de aumentacion:

```bash
python -m src.finetuning.augment_finetuning_data
```

o, desde PyCharm, ejecutar directamente:

```text
src/finetuning/augment_finetuning_data.py
```

Algunos scripts no tienen argumentos de consola: al ejecutarlos comienzan el
proceso completo con los parametros definidos en la cabecera del fichero. Antes
de relanzar una fase costosa conviene revisar rutas de entrada/salida, semilla y
opciones como `DRY_RUN`, `CONTINUE_ON_FAILURE` o numero de muestras.

### 1. Preparar tokenizador

El tokenizador REMI+BPE se entrena con:

```bash
python -m src.tokenization.tokenizer_train
```

Los tokenizadores resultantes se guardan en `tokenizer/`. Los experimentos
principales usan `tokenizer_REMI_BPE_v5.json`.

### 2. Tokenizar el corpus de preentrenamiento

Para el flujo principal:

```bash
python -m src.tokenization.tokenize_pretraining_v2
```

Salida esperada:

```text
data/interim/tokenized_json_bpe_v2/
data/interim/indexes/index_pretraining_v2.csv
```

La version historica del primer experimento se conserva en
`src/tokenization/tokenize_pretraining.py`.

### 3. Construir caches y preentrenar

El script de entrenamiento construye automaticamente los `.bin` si no existen:

```bash
python -m src.training.pretraining_v2
```

Salidas principales:

```text
data/bin/bin_for_pretraining_v2/
output/checkpoints/pretraining_v2/
```

El script `src/training/pretraining.py` conserva la configuracion del primer
experimento.

### 4. Preparar el corpus de fine-tuning

El flujo de fine-tuning parte de MIDIs limpios o consolidados y genera variantes
por transposicion y estiramiento temporal:

```bash
python -m src.finetuning.clean_finetuning_data_tfg
python -m src.finetuning.augment_finetuning_data
```

En el segundo flujo experimental, los datos preparados se ubican en
`data/finetuning_v2/`.

### 5. Tokenizar fine-tuning

Para el flujo principal:

```bash
python -m src.tokenization.tokenize_finetuning_v2
```

Salida esperada:

```text
data/interim/tokenized_finetuning_v3/
data/interim/indexes/index_finetuning_v3.csv
```

### 6. Entrenar fine-tuning

```bash
python -m src.finetuning.finetuning_v2
```

Este script carga el checkpoint de preentrenamiento desde
`output/checkpoints/pretraining_v2/best.pt` y guarda el ajuste fino en:

```text
output/checkpoints/finetuning_v2/
data/bin/bin_for_finetuning_v3/
```

### 7. Generar muestras

Preentrenamiento:

```bash
python -m src.generation.generate_pretraining_json_and_midi
```

Fine-tuning:

```bash
python -m src.generation.generate_finetuning_json_and_midi
```

Los scripts generan JSON con tokens y, cuando procede, MIDIs exportados para
escucha y evaluacion. Las salidas se guardan en carpetas de `output/` como:

```text
output/generation_pretraining_tfg_second/
output/generation_finetuning_tfg_second/
```

### 8. Evaluar los MIDIs generados

Evaluacion simbolica:

```bash
python -m src.evaluation.evaluate_generated_pretraining
python -m src.evaluation.evaluate_generated_finetuning
```

Evaluacion espectral:

```bash
python -m src.evaluation.generate_spectogram_pretraining
python -m src.evaluation.generate_spectrograms_finetuning
```

Generacion de graficos finales:

```bash
python -m src.evaluation.generar_graficos_evaluacion
```

Los resultados se guardan dentro de las carpetas de generacion correspondientes,
normalmente en subdirectorios `midi_eval`, `midi_eval_windows`,
`midi_spectral_eval` o `midi_spectral_eval_windows`.

## Artefactos principales

- `tokenizer/tokenizer_REMI_BPE_v5.json`: tokenizador principal.
- `data/interim/indexes/*.csv`: indices usados para localizar JSON tokenizados.
- `data/bin/*`: caches binarias de entrenamiento.
- `output/checkpoints/*/best.pt`: mejor checkpoint de cada fase.
- `output/generation_*/*.json`: muestras generadas en formato tokenizado.
- `output/generation_*/*.mid`: conversiones MIDI de las muestras.
- `output/generation_*/midi_eval*`: informes simbolicos.
- `output/generation_*/midi_spectral_eval*`: informes espectrales.

## Reproducibilidad

Los scripts fijan semillas en sus secciones de configuracion para que los splits,
muestreos y procesos de generacion sean reproducibles dentro del mismo entorno.
Las rutas principales se resuelven desde la raiz del proyecto. Si se modifican
datasets, tokenizadores o checkpoints, deben regenerarse tambien los indices,
caches binarias y evaluaciones asociadas.

Los checkpoints y datasets completos no se versionan. Para reproducir el
experimento en otra maquina es necesario disponer de los datasets originales o
de una copia local con la estructura indicada en este documento.
