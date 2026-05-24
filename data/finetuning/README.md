# Estructura de `data/finetuning`

Esta carpeta contiene el corpus objetivo y sus variantes para el primer flujo de fine-tuning. Se excluye del repositorio porque contiene MIDIs derivados y copias intermedias.

Estructura prevista:

```text
data/finetuning/
├── finetuning_sonatas_raw/
├── finetuning_sonatas_clean/
└── finetuning_sonatas_aug/
```

`finetuning_sonatas_raw` contiene la entrada original del flujo. `finetuning_sonatas_clean` recoge las piezas tras limpieza y filtrado. `finetuning_sonatas_aug` contiene las variantes aumentadas que se tokenizan para entrenamiento.
