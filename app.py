import streamlit as st
import fitz  # PyMuPDF to process PDFs
import faiss  # FAISS for vector storage
import numpy as np
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings

# PDF paths
pdf_paths = ['pdfs/doc1_SarvM.pdf', 'pdfs/doc2_SarvM.pdf', 'pdfs/doc3_SarvM.pdf']

# Function to extract text from PDFs
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

# Initialize Ollama embeddings model
ollama_embeddings = OllamaEmbeddings(model="gemma:2b")

# Embedding creation with Ollama gemma:2b
def get_ollama_embedding(text):
    try:
        embedding = ollama_embeddings.embed_query(text)
        return embedding
    except Exception as e:
        st.error(f"Error generating embedding: {e}")
        return None

# Vector storage (using FAISS)
index = faiss.IndexFlatL2(2048)  # Dimension depends on Ollama output

# Process PDFs and create embeddings
documents = []
for path in pdf_paths:
    text = extract_text_from_pdf(path)
    embedding = get_ollama_embedding(text)
    if embedding is not None:
        documents.append((text, embedding))
        index.add(np.array([embedding]))

# Chatbot interaction using ChatGroq
chat = ChatGroq(api_key="your_groq_api_key", model="llama3-8b-8192")

# Function to handle the Q&A process
def answer_question(question):
    try:
        # Get embedding for question
        question_embedding = get_ollama_embedding(question)
        if question_embedding is None:
            return "Error generating embedding for the question."

        # Perform a similarity search in FAISS to find the closest document embeddings
        D, I = index.search(np.array([question_embedding]), k=3)  # Return top 3 matches
        relevant_docs = [documents[i][0] for i in I[0]]

        # Combine relevant document text and question into a prompt
        context = "\n\n".join(relevant_docs)
        prompt = f"Context:\n{context}\n\nQuestion:\n{question}"

        # Call the `generate` method on the chat model with the combined prompt
        response = chat.generate(prompt)

        # Return the response text
        return response["text"]
    except Exception as e:
        st.error(f"Error during question answering: {e}")
        return None

# Streamlit UI for asking questions
st.title("PDF Q&A Chatbot")
question = st.text_input("Ask a question about the documents:")

if question:
    # Get the answer using the chatbot
    answer = answer_question(question)

    # Display the answer
    if answer:
        st.write(answer)
