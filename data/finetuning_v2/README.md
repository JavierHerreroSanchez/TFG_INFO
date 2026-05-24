# Estructura de `data/finetuning_v2`

Esta carpeta contiene el corpus objetivo empleado en el segundo flujo de fine-tuning. El contenido se excluye del control de versiones por tamano y reproducibilidad.

Estructura prevista:

```text
data/finetuning_v2/
├── mozart_sonatas_raw/
├── mozart_sonatas_merged/
└── mozart_sonatas_aug/
```

`mozart_sonatas_raw` conserva los MIDIs de partida. `mozart_sonatas_merged` contiene las versiones consolidadas por pieza. `mozart_sonatas_aug` almacena las variantes generadas mediante aumentacion.
