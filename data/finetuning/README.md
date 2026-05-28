# Estructura de `data/finetuning`

Corpus objetivo del primer flujo de fine-tuning. No se versiona porque contiene
MIDIs derivados y copias intermedias.

```text
data/finetuning/
|-- finetuning_sonatas_raw/
|-- finetuning_sonatas_clean/
`-- finetuning_sonatas_aug/
```

`raw` contiene la entrada original, `clean` las piezas filtradas y `aug` las
variantes aumentadas para entrenamiento.
