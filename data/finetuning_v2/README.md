# Estructura de `data/finetuning_v2`

Corpus de Mozart usado en el flujo principal de finetuning. Se utilizaron
todas las sonatas para piano de Mozart disponibles en
[Kunst der Fuge](https://www.kunstderfuge.com/mozart.htm) en el momento de
preparar el corpus.

```text
data/finetuning_v2/
|-- mozart_sonatas_raw/
|-- mozart_sonatas_merged/
`-- mozart_sonatas_aug/
```

`raw` contiene los MIDIs de partida, `merged` las piezas consolidadas y `aug`
las variantes aumentadas.

## Obras utilizadas

### Sonatas para piano

1. Sonata para piano n.º 1 en do mayor, K. 279.
2. Sonata para piano n.º 2 en fa mayor, K. 280.
3. Sonata para piano n.º 3 en si bemol mayor, K. 281.
4. Sonata para piano n.º 4 en mi bemol mayor, K. 282.
5. Sonata para piano n.º 5 en sol mayor, K. 283.
6. Sonata para piano n.º 6 en re mayor, K. 284.
7. Sonata para piano n.º 7 en do mayor, K. 309.
8. Sonata para piano n.º 8 en re mayor, K. 311.
9. Sonata para piano n.º 9 en la menor, K. 310.
10. Sonata para piano n.º 10 en do mayor, K. 330.
11. Sonata para piano n.º 11 en la mayor, K. 331.
12. Sonata para piano n.º 12 en fa mayor, K. 332.
13. Sonata para piano n.º 13 en si bemol mayor, K. 333.
14. Sonata para piano n.º 14 en do menor, K. 457.
15. Sonata para piano n.º 15 en fa mayor, K. 533/494.
16. Sonata para piano n.º 16 en do mayor, K. 545.
17. Sonata para piano n.º 17 en fa mayor, K. 547a.
18. Sonata para piano n.º 18 en si bemol mayor, K. 570.
19. Sonata para piano n.º 19 en re mayor, K. 576.
20. Movimiento de sonata en si bemol mayor, K. 400 (372a).

### Sonata para dos pianos

21. Sonata para dos pianos en re mayor, K. 448 (375a):
    `Allegro con spirito`, `Andante` y `Allegro molto`.

Los tres movimientos se obtuvieron como archivos MIDI separados. En total, el corpus de partida
contiene 21 obras distribuidas en 23 archivos MIDI.

## Disponibilidad

Los archivos MIDI originales y sus versiones procesadas o aumentadas no se
redistribuyen en este repositorio. Permanecen excluidos mediante `.gitignore`
debido a las condiciones de uso de Kunst der Fuge y a los derechos de las
personas autoras de las secuencias.
