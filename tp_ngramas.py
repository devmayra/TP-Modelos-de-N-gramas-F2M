# IMPORTS
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import spacy
import pandas as pd
import matplotlib.pyplot as plt

# DESCARGAS NECESARIAS
nltk.download("stopwords")
nltk.download("punkt_tab")

# RECURSOS GLOBALES
STOP_WORDS = set(stopwords.words("spanish"))
nlp = spacy.load("es_core_news_sm")


# FUNCIÓN 1: leer el corpus
def leer_corpus(ruta):
    with open(ruta, encoding="latin-1") as f:
        lineas = [linea.strip() for linea in f.readlines() if linea.strip()]
    return lineas


# FUNCIÓN 2: preprocesar una oración
def preprocesar_linea(texto):

    tokens = word_tokenize(texto.lower(), language="spanish")

    tokens_filtrados = [
        w
        for w in tokens
        if w.isalpha() and w not in STOP_WORDS and len(w) > 2
    ]

    doc = nlp(" ".join(tokens_filtrados))

    lemas = [
        token.lemma_
        for token in doc
        if token.lemma_.strip()
    ]

    return " ".join(lemas)


# FUNCIÓN 3: calcular n-gramas
def calcular_ngramas(corpus_procesado, n):
    vectorizer = CountVectorizer(
        ngram_range=(n, n),
        min_df=2
    )

    X = vectorizer.fit_transform(corpus_procesado)

    return vectorizer, X


# FUNCIÓN 4: obtener frecuencias
def obtener_frecuencias(vectorizer, X):

    df = pd.DataFrame(
        X.sum(axis=0).T,
        index=vectorizer.get_feature_names_out(),
        columns=["freq"]
    ).sort_values(
        by="freq",
        ascending=False
    )

    return df


# FUNCIÓN 5: graficar
def graficar(df_bi, df_tri):

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(16, 6)
    )

    df_bi.head(15).sort_values("freq").plot(
        kind="barh",
        ax=ax1,
        title="Bigramas"
    )

    df_tri.head(15).sort_values("freq").plot(
        kind="barh",
        ax=ax2,
        title="Trigramas"
    )

    plt.tight_layout()
    plt.show()


# CÓDIGO PRINCIPAL

corpus = leer_corpus("CorpusEducacion.txt")

corpus_procesado = [
    preprocesar_linea(oracion)
    for oracion in corpus
]

vectorizer_bi, X_bi = calcular_ngramas(
    corpus_procesado,
    n=2
)

df_bi = obtener_frecuencias(
    vectorizer_bi,
    X_bi
)

print("\nBIGRAMAS:")
print(df_bi.head(10))

vectorizer_tri, X_tri = calcular_ngramas(
    corpus_procesado,
    n=3
)

df_tri = obtener_frecuencias(
    vectorizer_tri,
    X_tri
)

print("\nTRIGRAMAS:")
print(df_tri.head(10))

graficar(df_bi, df_tri)