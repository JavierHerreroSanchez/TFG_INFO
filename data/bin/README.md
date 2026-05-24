# Estructura de `data/bin`

Esta carpeta contiene caches binarias generadas a partir de los JSON tokenizados. Se excluyen del control de versiones porque pueden reconstruirse desde los indices y ocupan mucho espacio.

Estructura prevista:

```text
data/bin/
├── bin_for_pretraining/
├── bin_for_pretraining_v2/
├── bin_for_finetuning/
├── bin_for_finetuning_v2/
└── bin_for_finetuning_v3/
```

Cada subcarpeta corresponde a una configuracion experimental concreta y contiene los splits binarios de entrenamiento, validacion y test junto con sus metadatos.
