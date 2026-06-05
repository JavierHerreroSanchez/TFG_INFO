# Estructura de `data/bin`

Caches binarias generadas desde los JSON tokenizados. Se pueden reconstruir a
partir de los índices, por eso no se versionan.

```text
data/bin/
|-- bin_for_pretraining/
|-- bin_for_pretraining_v2/
|-- bin_for_finetuning/
|-- bin_for_finetuning_v2/
`-- bin_for_finetuning_v3/
```

Cada subcarpeta contiene los splits binarios y sus metadatos.
