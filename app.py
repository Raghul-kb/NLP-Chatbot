import streamlit as st
import numpy as np
import PyPDF2
import nltk
import re

from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

nltk.download('punkt')

st.title("📄 PDF Chatbot")

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
# Upload PDF
# -----------------------------

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:

    text = extract_pdf_text(uploaded_file)

    text = clean_text(text)

    sentences = nltk.sent_tokenize(text)

    st.write("PDF Loaded Successfully")

    # -----------------------------
    # Tokenization
    # -----------------------------

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentences)

    sequences = tokenizer.texts_to_sequences(sentences)

    max_len = 25

    X = pad_sequences(sequences, maxlen=max_len)

    vocab_size = len(tokenizer.word_index) + 1

    # -----------------------------
    # LSTM Model
    # -----------------------------

    model = Sequential()

    model.add(Embedding(vocab_size, 64, input_length=max_len))
    model.add(LSTM(64))
    model.add(Dense(max_len))

    model.compile(
        optimizer='adam',
        loss='mse'
    )

    st.write("Training model...")

    model.fit(X, X, epochs=20, verbose=0)

    st.write("Model Ready")

    # Precompute sentence vectors
    sentence_vectors = model.predict(X)

    # -----------------------------
    # Question Input
    # -----------------------------

    question = st.text_input("Ask a question from the document")

    if question:

        q_seq = tokenizer.texts_to_sequences([question])
        q_pad = pad_sequences(q_seq, maxlen=max_len)

        q_vector = model.predict(q_pad)

        similarity = cosine_similarity(q_vector, sentence_vectors)

        index = np.argmax(similarity)

        answer = sentences[index]

        st.success(answer)
