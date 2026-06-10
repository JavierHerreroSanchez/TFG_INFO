# Estructura de `data/finetuning`

Corpus objetivo del primer flujo de finetuning, formado por una selección de
sonatas para teclado y piano procedentes de GiantMIDI-Piano. No se versiona
porque contiene MIDIs derivados y copias protegidas por derechos de autor.

```text
data/finetuning/
|-- finetuning_sonatas_raw/
|-- finetuning_sonatas_clean/
`-- finetuning_sonatas_aug/
```

`raw` contiene la entrada original, `clean` las piezas filtradas y `aug` las
variantes aumentadas para entrenamiento.

## Selección original

La carpeta `finetuning_sonatas_raw/` contiene 130 archivos MIDI de siete
compositores. Tras el proceso de limpieza y filtrado se obtuvieron 124 archivos
en `finetuning_sonatas_clean/`; algunas entradas se descartaron y cuatro obras
se dividieron en dos partes.

### Carl Philipp Emanuel Bach

42 sonatas para teclado:

- H. 2, 15, 21, 32.5, 38, 39, 40, 41, 51, 52, 53, 55, 58, 59, 60, 61, 66,
  83, 105, 116, 117, 118, 121, 128, 130, 131, 132, 174, 176, 177, 186, 187,
  208, 209, 210, 211, 213, 240, 245, 270, 281 y 287.

### Ludwig van Beethoven

15 sonatas para piano:

- Sonata n.º 2, op. 2 n.º 2.
- Sonatas n.º 5, 6 y 7, op. 10 n.º 1, 2 y 3.
- Sonata n.º 8, op. 13.
- Sonatas n.º 9 y 10, op. 14 n.º 1 y 2.
- Sonata n.º 12, op. 26.
- Sonata n.º 15, op. 28.
- Sonatas n.º 16, 17 y 18, op. 31 n.º 1, 2 y 3.
- Sonatas n.º 19 y 20, op. 49 n.º 1 y 2.
- Sonata n.º 25, op. 79.

### Muzio Clementi

5 archivos fuente:

- Tres sonatas para piano, op. 33.
- Tres sonatas para piano, op. 37.
- Sonata para piano n.º 1, op. 2 n.º 2.
- Sonata para piano, op. 26.
- Sonata para piano, op. 46.

### Joseph Haydn

33 sonatas para teclado:

- Hob. XVI:15, 18, 20, 21, 22, 23, 24, 25, 26, 27, 29, 30, 31, 32, 33, 34,
  35, 36, 37, 38, 39, 40, 41, 43, 44, 47, 48, 49, 50, 51 y 52.
- Hob. XVI:G1.
- Una sonata en re mayor catalogada como Hob. XVI:deest.

### Leopold Koželuch

3 sonatas para piano:

- Op. 38 n.º 1, 2 y 3.

### Leopold Mozart

2 sonatas para piano:

- Sonata en do mayor.
- Sonata en fa mayor.

### Wolfgang Amadeus Mozart

30 archivos fuente:

- Sonatas para piano K. 279, 280, 281, 282, 283, 284, 309, 310, 311, 331,
  333, 457, 533/494, 545, 570 y 576.
- Sonata para piano en si bemol mayor, K. 498a.
- Sonata para piano en fa mayor, K. Anh. 135/547a.
- Fragmentos de sonata K. Anh. 31/569a, K. Anh. 29/590a, K. Anh. 30/590b y
  K. Anh. 37/590c.
- Una entrada denominada `Piano Sonata No.20 in C minor` sin número de catálogo
  en el nombre del archivo fuente.
- Sonata para dos pianos en re mayor, K. 448/375a.
- Sonatas para piano a cuatro manos K. 19d, 358/186c, 381/123a y 497.
- Movimiento de sonata para dos pianos, K. Anh. 42/375b.
- Movimiento de sonata en si bemol mayor, K. Anh. 43/375c.

## Disponibilidad

Los archivos MIDI de GiantMIDI-Piano y sus versiones procesadas o aumentadas
no se redistribuyen en este repositorio. El inventario se conserva para
documentar de forma reproducible la selección del primer finetuning.
