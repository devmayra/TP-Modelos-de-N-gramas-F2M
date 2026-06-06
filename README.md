# TP — Modelos de N-gramas en PLN

- **Materia:** Técnicas de Procesamiento del Habla
- **Bloque:** Modelos y Clasificación
- **Unidad:** 2

---

## Descripción

Trabajo práctico sobre modelos de N-gramas aplicados a un corpus real en español. El objetivo es encontrar las secuencias de palabras más frecuentes (bigramas y trigramas) en opiniones de estudiantes colombianos sobre educación superior, y visualizar los resultados en un gráfico comparativo.

---

## ¿Qué hace el código?

1. **Lee el corpus** desde `CorpusEducacion.txt` (36 opiniones de alumnos)
2. **Preprocesa** cada oración: tokenización, eliminación de stop words en español y lematización
3. **Calcula bigramas y trigramas** usando `CountVectorizer` de scikit-learn con `min_df=2`
4. **Grafica** la comparación de frecuencias entre ambos tipos de n-gramas

---

## Contenido del repositorio

```
├── tp_ngramas.py          # código principal organizado en funciones
├── CorpusEducacion.txt    # corpus: opiniones de estudiantes colombianos 
└── README.md
```

---

## Requisitos

```bash
pip install nltk scikit-learn matplotlib pandas spacy
```

Además, es necesario instalar el modelo de español de spaCy:

```bash
python -m spacy download es_core_news_sm
```

Los recursos de NLTK se descargan automáticamente al ejecutar el script:

```python
nltk.download("stopwords")
nltk.download("punkt_tab")
```

---

## Preprocesamiento

Antes de calcular los n-gramas, cada oración pasa por tres etapas:

- **Tokenización:** separa el texto en palabras individuales
- **Stop words:** elimina palabras sin valor semántico (el, la, que, de...)
- **Lematización:** transforma las palabras a su forma base o lema (estudiantes → estudiante, estudiando → estudiar).

---

## Ejemplo de resultado

Los bigramas más frecuentes del corpus reflejan los temas centrales de las opiniones:

| N-grama | Frecuencia |
|---|---|
| sueño educación | 15 |
| educación calidad | 7 |
| educación sueño | 6 |
| institución educativo | 4 |