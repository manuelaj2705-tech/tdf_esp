import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from nltk.stem import SnowballStemmer

st.title("🚀 Buscador de Información sobre el Espacio")

# Documentos de ejemplo
default_docs = """La Tierra gira alrededor del Sol en una órbita que dura aproximadamente un año.
Los astronautas viajan al espacio en cohetes y estaciones espaciales.
La Luna es el satélite natural de la Tierra y refleja la luz del Sol.
Los telescopios permiten observar estrellas, planetas y galaxias lejanas.
Los agujeros negros tienen una gravedad tan fuerte que nada puede escapar de ellos.
Las estrellas producen luz y energía mediante reacciones nucleares en su interior."""

# Stemmer en español
stemmer = SnowballStemmer("spanish")

def tokenize_and_stem(text):
    text = text.lower()
    text = re.sub(r'[^a-záéíóúüñ\s]', ' ', text)
    tokens = [t for t in text.split() if len(t) > 1]
    stems = [stemmer.stem(t) for t in tokens]
    return stems

# Layout en dos columnas
col1, col2 = st.columns([2, 1])

with col1:
    text_input = st.text_area("📚 Textos sobre el espacio (uno por línea):", default_docs, height=150)
    question = st.text_input("❓ Haz una pregunta sobre el universo:", "¿Qué cuerpo celeste refleja la luz del Sol?")

with col2:
    st.markdown("### 💡 Preguntas sugeridas:")
    
    if st.button("¿Qué cuerpo celeste refleja la luz del Sol?", use_container_width=True):
        st.session_state.question = "¿Qué cuerpo celeste refleja la luz del Sol?"
        st.rerun()
    
    if st.button("¿Qué usan los científicos para observar el espacio?", use_container_width=True):
        st.session_state.question = "¿Qué usan los científicos para observar el espacio?"
        st.rerun()
        
    if st.button("¿Qué tienen los agujeros negros?", use_container_width=True):
        st.session_state.question = "¿Qué tienen los agujeros negros?"
        st.rerun()
        
    if st.button("¿Qué produce energía en las estrellas?", use_container_width=True):
        st.session_state.question = "¿Qué produce energía en las estrellas?"
        st.rerun()
        
    if st.button("¿Cómo viajan los astronautas al espacio?", use_container_width=True):
        st.session_state.question = "¿Cómo viajan los astronautas al espacio?"
        st.rerun()

# Actualizar pregunta si se seleccionó una sugerida
if 'question' in st.session_state:
    question = st.session_state.question

if st.button("🔍 Analizar", type="primary"):
    documents = [d.strip() for d in text_input.split("\n") if d.strip()]
    
    if len(documents) < 1:
        st.error("⚠️ Ingresa al menos un texto.")
    elif not question.strip():
        st.error("⚠️ Escribe una pregunta.")
    else:
        vectorizer = TfidfVectorizer(
            tokenizer=tokenize_and_stem,
            min_df=1
        )
        
        X = vectorizer.fit_transform(documents)
        
        st.markdown("### 📊 Matriz TF-IDF")
        df_tfidf = pd.DataFrame(
            X.toarray(),
            columns=vectorizer.get_feature_names_out(),
            index=[f"Texto {i+1}" for i in range(len(documents))]
        )
        st.dataframe(df_tfidf.round(3), use_container_width=True)
        
        question_vec = vectorizer.transform([question])
        similarities = cosine_similarity(question_vec, X).flatten()
        
        best_idx = similarities.argmax()
        best_doc = documents[best_idx]
        best_score = similarities[best_idx]
        
        st.markdown("### 🎯 Resultado")
        st.markdown(f"**Tu pregunta:** {question}")
        
        if best_score > 0.01:
            st.success(f"**Texto más relacionado:** {best_doc}")
            st.info(f"📈 Nivel de similitud: {best_score:.3f}")
        else:
            st.warning(f"**Resultado con baja coincidencia:** {best_doc}")
            st.info(f"📉 Nivel de similitud: {best_score:.3f}")
