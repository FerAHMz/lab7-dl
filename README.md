# Laboratorio 7 - NLP end-to-end y Embeddings

CC3092 Deep Learning y Sistemas Inteligentes.

Fernando Andree Hernández Martínez - 23645
Fernando José Rueda Rodas - 23748

Pipeline completo de NLP, **texto crudo → tokens → IDs → vectores → modelo → salida**. Se entrenan embeddings de palabras con una implementación propia de **skip-gram con negative sampling (SGNS) en PyTorch** sobre un subconjunto de 25M tokens de WikiText-103, se validan contra **Word2Vec de gensim** entrenado sobre el mismo corpus y se comparan con **GloVe 6B-100d** preentrenado: analogías (3CosAdd / 3CosMul), WordSim-353, SimLex-999, paralelismo de vectores diferencia y t-SNE. Al final los tres conjuntos de embeddings se usan para clasificar noticias de **AG News** contra un baseline TF-IDF y embeddings aleatorios.

## Estructura

```
lab7-dl/
├── notebooks/
│   └── Laboratorio7_NLP_Embeddings.ipynb   # notebook completo, ejecutado y comentado
├── src/
│   ├── config.py      # rutas, semilla, tamaño del subconjunto, iteraciones de SGNS
│   ├── text.py        # normalización, tokenizador, WikiText-103 -> IDs, Zipf, vocabulario, submuestreo, pares
│   ├── sgns.py        # modelo SGNS, generación de pares en GPU, entrenamiento, Word2Vec de gensim
│   ├── evaluation.py  # analogías 3CosAdd/3CosMul, WordSim/SimLex, vecinos, analogia(), paralelismo
│   ├── classify.py    # AG News: split, vocabularios, EmbeddingBag + MLP, TF-IDF + LogReg
│   └── plots.py       # estilo y gráficas comunes
├── results/
│   ├── runs/          # configuración, historial por epoch, tiempos y memoria de cada iteración (JSON)
│   ├── vectors/       # sgns_best_d100.kv (mejor modelo) — el resto de vectores no se versiona
│   └── *.csv          # tablas de iteraciones, analogías, paralelismo, clasificación y comparación
├── figures/           # gráficas generadas por el notebook
├── data/raw/          # WikiText-103, AG News y GloVe (se descargan al ejecutar; no versionado)
├── codebook.md        # descripción de los datos, particiones y transformaciones
├── requirements.txt
└── README.md
```

## Contenido del notebook

1. Datos: WikiText-103, AG News, GloVe y benchmarks.
2. Exploración y preprocesamiento: tamaño del corpus, normalización alineada con GloVe, tokenizador propio contra spaCy y NLTK en 12 oraciones, ley de Zipf y cobertura, vocabulario con distintos `min_count`, submuestreo de palabras frecuentes y conteo de pares skip-gram antes y después del submuestreo.
3. Investigación: `nn.Embedding`, `from_pretrained`, `nn.EmbeddingBag` con offsets, CBOW vs skip-gram, matrices W y W', Word2Vec vs GloVe.
4. Entrenamiento: 13 iteraciones de SGNS (learning rate, dimensión, tamaño del corpus, ventana, negativos, submuestreo, `min_count`) con pérdida, analogías, WordSim-353 y vecinos por epoch, tiempo y memoria pico; configuración final, Word2Vec de gensim y GloVe.
5. Aritmética vectorial: función `analogia` propia verificada contra gensim, 20 analogías de 7 tipos con y sin exclusión, benchmark por categoría con vocabulario compartido (3CosAdd y 3CosMul), t-SNE de 519 palabras y paralelismo de vectores diferencia.
6. Clasificación de AG News: TF-IDF + regresión logística, EmbeddingBag aleatorio, SGNS / gensim / GloVe congelados y con fine-tuning, evaluación en test con matrices de confusión, SimLex-999 y experimento con 1 %, 10 %, 50 % y 100 % de los datos.
7. Tabla comparativa y gráficas (analogías contra tokens del corpus, F1 contra fracción de datos).
8. Discusión y análisis.

El informe escrito (máximo 5 páginas) se entrega aparte en PDF.

## Vectores del mejor modelo

`results/vectors/sgns_best_d100.kv` (40 MB, 99,347 palabras, d = 100), formato `KeyedVectors` de gensim:

```python
from gensim.models import KeyedVectors
kv = KeyedVectors.load("results/vectors/sgns_best_d100.kv")
kv.most_similar(positive=["king", "woman"], negative=["man"])
```

## Cómo ejecutar

macOS / Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
jupyter notebook notebooks/Laboratorio7_NLP_Embeddings.ipynb
```

Windows:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
jupyter notebook notebooks\Laboratorio7_NLP_Embeddings.ipynb
```

El código usa CUDA si está disponible, si no MPS (Apple Silicon) y si no CPU. La semilla es 42. La primera ejecución descarga ~1 GB de datos en `data/raw/` y cachea el corpus tokenizado en `data/processed/`. Cada iteración de SGNS guarda su resultado en `results/runs/` y sus vectores en `results/vectors/`; al re-ejecutar, las iteraciones ya entrenadas se cargan del disco en lugar de reentrenarse (borrar esos archivos para entrenar de nuevo). Una ejecución completa desde cero tarda unos 80 minutos en un Apple M4 Pro (la mayor parte son las 13 iteraciones de SGNS y la configuración final); desactive la suspensión del equipo mientras corre.
