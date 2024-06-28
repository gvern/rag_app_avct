import os
import streamlit as st
from pkg.advanced_chatbot.services.rag_service import RagService
from pathlib import Path
import textwrap

st.set_page_config(layout="wide")
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

def handle_uploaded_files(files):
    index_ids = []
    for uploaded_file in files:
        try:
            file_path = save_file(uploaded_file)
            st.success(f"Fichier enregistré sous {file_path}")

            # Loading the document (Indexing/Ingestion)
            index_id, _ = RagService.create_vector_store_index(file_path)
            st.success(f"Document indexé avec l'ID : {index_id}")
            index_ids.append(index_id)

            # Generate and display summary
            summary = RagService.summarize_document_index(index_id)
            st.session_state.summaries[index_id] = summary
            st.write(f"Résumé pour {uploaded_file.name} :")
            st.write(format_text(summary))

            # Check and translate the first page if necessary
            language = RagService.detect_document_language(index_id)
            if language != 'fr':
                translation = RagService.translate_and_summarize_first_page_fr(index_id)
                st.session_state.translations[index_id] = translation
                st.write(f"Traduction de la première page pour {uploaded_file.name} :")
                st.write(format_text(translation))
        except Exception as e:
            st.error(f"Erreur lors du traitement de {uploaded_file.name}: {str(e)}")
    return index_ids

# Initialize session state for storing index IDs, conversation history, summaries, and translations
if 'index_ids' not in st.session_state:
    st.session_state.index_ids = []

if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

if 'summaries' not in st.session_state:
    st.session_state.summaries = {}

if 'translations' not in st.session_state:
    st.session_state.translations = {}

# UI Layout
st.header("Téléchargement des Fichiers")
uploaded_files = st.file_uploader("Choisissez des fichiers PDF ou DOCX", type=["pdf", "docx"], accept_multiple_files=True)

if uploaded_files:
    st.session_state.index_ids = handle_uploaded_files(uploaded_files)

st.header("Chat Interactif")
question = st.text_input("Posez une question sur les documents")
if st.button('Envoyer') and question:
    try:
        response_generator, source_nodes = RagService.complete_chat(question, st.session_state.conversation_history, st.session_state.index_ids)
        
        # Update conversation history
        st.session_state.conversation_history.append({
            'role': 'user',
            'content': question
        })
        for response in response_generator:
            st.session_state.conversation_history.append({
                'role': 'assistant',
                'content': response
            })
        
        st.subheader("Réponse :")
        response_text = ""
        for item in st.session_state.conversation_history:
            if item['role'] == 'assistant':
                response_text += item['content'] + " "
        st.write(format_text(response_text.strip()))
        
        st.subheader("Sources :")
        unique_sources = set()
        for source in source_nodes:
            source_text = source.get_text().strip()
            if source_text not in unique_sources:
                st.write(format_text(source_text))
                unique_sources.add(source_text)
    except Exception as e:
        st.error(f"Erreur lors de l'interaction avec le chatbot: {str(e)}")
