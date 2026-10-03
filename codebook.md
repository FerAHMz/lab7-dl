# Codebook — Laboratorio 7

## 1. WikiText-103 (corpus para entrenar embeddings)

**Fuente:** Merity et al. (2016), *Pointer Sentinel Mixture Models*. Descargado con `load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")` (versión *raw*: texto original, sin `<unk>`). Se cachea en `data/raw/hf/` (no versionado, ~880 MB).

| Variable | Tipo | Descripción | Valores |
|---|---|---|---|
| `text` | `str` | Una línea del volcado: título de artículo (` = Título = `), subtítulo (` = = Sección = = `), párrafo o línea vacía | Texto en inglés pretokenizado por espacios; separadores internos ` @-@ `, ` @,@ `, ` @.@ ` |

| Split | Artículos | Filas | Párrafos | Tokens (espacios) | Tokens (nuestro tokenizador) |
|---|---|---|---|---|---|
| train | 28,472 | 1,801,350 | 860,934 | 101,425,671 | 84,614,118 |
| validation | 60 | 3,760 | 1,841 | 213,886 | 179,562 |
| test | 60 | 4,358 | 2,187 | 241,211 | 201,732 |

**Subconjunto de entrenamiento:** los primeros 8,478 artículos completos del split `train` (25,002,277 tokens, 255,175 párrafos, 352,393 tipos). Se usa el mismo para el SGNS propio y para Word2Vec de gensim. Se guarda en `data/processed/wikitext_subset.npz`:

| Arreglo | Tipo | Descripción |
|---|---|---|
| `ids` | `int32[25M]` | Tokens como IDs; el ID es el rango de frecuencia (0 = `the`) |
| `starts` | `int64[párrafos+1]` | Posición donde empieza cada párrafo (las ventanas no los cruzan) |
| `words` | `str[352,393]` | ID → palabra |
| `counts` | `int64[352,393]` | Frecuencia de cada ID en el subconjunto |

## 2. AG News (clasificación)

**Fuente:** Zhang, Zhao & LeCun (2015). `load_dataset("fancyzhx/ag_news")`.

| Variable | Tipo | Descripción | Valores |
|---|---|---|---|
| `text` | `str` | Título + descripción de la noticia | Inglés, con artefactos de scraping (`\`, `#39;`, `&lt;b&gt;`) |
| `label` | `int64` | Categoría | 0–3 (ver tabla) |

| Índice | Clase | Train oficial | Test |
|---|---|---|---|
| 0 | World | 30,000 | 1,900 |
| 1 | Sports | 30,000 | 1,900 |
| 2 | Business | 30,000 | 1,900 |
| 3 | Sci/Tech | 30,000 | 1,900 |

| Partición | Tamaño | Origen |
|---|---|---|
| Entrenamiento | 108,000 (27,000 por clase) | 90 % del train oficial, estratificado, `random_state=42` |
| Validación | 12,000 (3,000 por clase) | 10 % restante del train oficial |
| Test | 7,600 (1,900 por clase) | test oficial, usado solo para la evaluación final |
| Fracciones 1 / 10 / 50 % | 1,080 / 10,800 / 54,000 | submuestras estratificadas del set de entrenamiento |

## 3. Modelos preentrenados y benchmarks

| Recurso | Origen | Contenido |
|---|---|---|
| GloVe 6B-100d | `gensim.downloader.load("glove-wiki-gigaword-100")` | 400,000 palabras en minúsculas, d = 100, Wikipedia 2014 + Gigaword 5 (6B tokens) |
| `questions-words.txt` | `gensim.test.utils.datapath` | 19,544 analogías `a b c d` en 14 categorías (5 semánticas, 9 sintácticas) |
| `wordsim353.tsv` | idem | 353 pares de palabras con similitud/relación humana 0–10 |
| `simlex999.txt` | idem | 999 pares con similitud estricta 0–10 (solo evaluación final) |

## 4. Transformaciones aplicadas (`src/text.py`)

1. `html.unescape`; `#39;` → `'`, `#36;` → `$`; se eliminan etiquetas HTML, `\` y entidades `#NN;` restantes.
2. WikiText: ` @-@ ` → `-`, ` @,@ ` → `,`, ` @.@ ` → `.`.
3. Minúsculas; `’` → `'`.
4. Clíticos al estilo Penn Treebank: `n't`, `'s`, `'re`, `'ve`, `'ll`, `'d`, `'m` se separan.
5. Tokenización por expresión regular: acrónimos con punto (`u.s.`), clíticos, números con formato (`3.5`, `1,000`), palabras con guion o apóstrofo internos; la puntuación se descarta.
6. Vocabulario con `min_count` (5 por defecto): los tokens menos frecuentes se eliminan del flujo para SGNS / gensim (igual que word2vec).
7. Submuestreo por epoch con $P_{conservar}(w) = (\sqrt{f/t} + 1)\,t/f$, $t = 10^{-4}$.
8. AG News: mismos pasos 1–5; vocabulario del clasificador según la inicialización (ver notebook, sección 6).
