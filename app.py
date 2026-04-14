import streamlit as st
import PyPDF2
import re
import nltk
import numpy as np

from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')

# -----------------------------
# Extract text from PDF
# -----------------------------
def extract_pdf_text(pdf_file):

    reader = PyPDF2.PdfReader(pdf_file)
    text = ""

    for page in reader.pages:
        text += page.extract_text()

    return text


# -----------------------------
# Clean text
# -----------------------------
def clean_text(text):

    text = text.lower()
    text = re.sub(r'[^a-zA-Z. ]', '', text)

    return text


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📄 NLP PDF Chatbot")

st.write("Upload a PDF and ask questions from it")

pdf_file = st.file_uploader("Upload PDF", type="pdf")


if pdf_file is not None:

    text = extract_pdf_text(pdf_file)

    text = clean_text(text)

    sentences = sent_tokenize(text)

    vectorizer = TfidfVectorizer()

    X = vectorizer.fit_transform(sentences)

    question = st.text_input("Ask a question")

    if question:

        q_clean = clean_text(question)

        q_vector = vectorizer.transform([q_clean])

        similarity = cosine_similarity(q_vector, X)

        index = np.argmax(similarity)

        answer = sentences[index]

        st.write("### Answer")
        st.success(answer)
