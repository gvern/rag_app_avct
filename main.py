import os
import streamlit as st
from pkg.advanced_chatbot.services.rag_service import RagService
from pathlib import Path
import textwrap

st.title("Système de Gestion de Documents Intelligents")

# Directory to store uploaded files
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

def save_file(uploaded_file):
    file_path = UPLOAD_DIR / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def format_text(text):
    return textwrap.fill(text, width=80)

uploaded_files = st.file_uploader(label="Choisissez des fichiers PDF ou DOCX", type=["pdf", "docx"], accept_multiple_files=True)

if uploaded_files:
    index_ids = []
    for uploaded_file in uploaded_files:
        file_path = save_file(uploaded_file)
        st.write(f"Fichier enregistré sous {file_path}")

        # Loading the document (Indexing/Ingestion)
        index_id, _ = RagService.create_vector_store_index(file_path)
        st.write(f"Document indexé avec l'ID : {index_id}")
        index_ids.append(index_id)

    question = st.text_input("Posez une question sur les documents")
    if question:
        conversation_history = []
        response_generator, source_nodes = RagService.complete_chat(question, conversation_history, index_ids)
        
        st.write("Réponse :")
        response_text = ""
        for response in response_generator:
            response_text += response + " "
        st.write(format_text(response_text.strip()))
        
        st.write("Sources :")
        unique_sources = set()
        for source in source_nodes:
            source_text = source.get_text().strip()
            if source_text not in unique_sources:
                st.write(format_text(source_text))
                unique_sources.add(source_text)
