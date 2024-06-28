import os
import json
import streamlit as st
from pkg.advanced_chatbot.services.rag_service import RagService
from pathlib import Path
import textwrap
from datetime import datetime

st.set_page_config(layout="wide")
st.title("Système de Gestion de Documents Intelligents")

# Directory to store uploaded files and index configurations
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_FILE = Path("index_configs.json")

def load_configs():
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    return {}

def save_configs(configs):
    with open(CONFIG_FILE, "w") as f:
        json.dump(configs, f)

def save_file(uploaded_file):
    file_path = UPLOAD_DIR / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def format_text(text):
    return textwrap.fill(text, width=80)

def handle_uploaded_files(files):
    index_ids = []
    configs = load_configs()
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

            # Detect language
            language = RagService.detect_document_language(index_id)

            # Check and translate the first page if necessary
            if language != 'fr':
                translation = RagService.translate_and_summarize_first_page_fr(index_id)
                st.session_state.translations[index_id] = translation
                st.write(f"Traduction de la première page pour {uploaded_file.name} :")
                st.write(format_text(translation))
            
            # Save index configuration
            configs[index_id] = {
                "file_path": str(file_path),
                "summary": summary,
                "language": language
            }
            save_configs(configs)

        except Exception as e:
            st.error(f"Erreur lors du traitement de {uploaded_file.name}: {str(e)}")
    return index_ids

# Initialize session state for storing index IDs, conversation history, summaries, translations, and past conversations
if 'sessions' not in st.session_state:
    st.session_state.sessions = {}

if 'current_session' not in st.session_state:
    st.session_state.current_session = None

if 'index_ids' not in st.session_state:
    st.session_state.index_ids = []

if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

if 'summaries' not in st.session_state:
    st.session_state.summaries = {}

if 'translations' not in st.session_state:
    st.session_state.translations = {}

# Load existing configs
configs = load_configs()
for index_id, config in configs.items():
    st.session_state.summaries[index_id] = config["summary"]

# Sidebar for session management
st.sidebar.header("Gestion des Sessions")

# Function to save the current session
def save_current_session():
    if st.session_state.current_session:
        st.session_state.sessions[st.session_state.current_session] = {
            "index_ids": st.session_state.index_ids,
            "conversation_history": st.session_state.conversation_history,
            "summaries": st.session_state.summaries,
            "translations": st.session_state.translations
        }

# Function to create a new session with an automatic name
def create_new_session():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_session_name = f"Session {timestamp}"
    save_current_session()
    st.session_state.current_session = new_session_name
    st.session_state.index_ids = []
    st.session_state.conversation_history = []
    st.session_state.summaries = {}
    st.session_state.translations = {}
    st.session_state.sessions[new_session_name] = {
        "index_ids": [],
        "conversation_history": [],
        "summaries": {},
        "translations": {}
    }
    st.experimental_rerun()

# Button to create a new session
if st.sidebar.button("Créer une nouvelle session"):
    create_new_session()

# Display list of sessions in the sidebar
st.sidebar.subheader("Sessions existantes")
for session_name in st.session_state.sessions.keys():
    if st.sidebar.button(session_name):
        save_current_session()
        st.session_state.current_session = session_name
        session_data = st.session_state.sessions[session_name]
        st.session_state.index_ids = session_data["index_ids"]
        st.session_state.conversation_history = session_data["conversation_history"]
        st.session_state.summaries = session_data["summaries"]
        st.session_state.translations = session_data["translations"]
        st.experimental_rerun()

# Display past conversation if selected
st.sidebar.subheader("Conversation courante :")
if st.session_state.conversation_history:
    for i, entry in enumerate(st.session_state.conversation_history):
        role = "Utilisateur" if entry['role'] == 'user' else "Assistant"
        st.sidebar.write(f"{role} ({i+1}): {entry['content']}")

# Main UI Layout
st.header("Téléchargement des Fichiers")
uploaded_files = st.file_uploader("Choisissez des fichiers PDF ou DOCX", type=["pdf", "docx"], accept_multiple_files=True)

if uploaded_files:
    st.session_state.index_ids = handle_uploaded_files(uploaded_files)
    save_current_session()

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
        response_text = ""
        for response in response_generator:
            response_text += response + " "
        
        st.session_state.conversation_history.append({
            'role': 'assistant',
            'content': response_text.strip()
        })
        
        st.subheader("Réponse :")
        st.write(format_text(response_text.strip()))
        
        st.subheader("Sources :")
        unique_sources = set()
        for source in source_nodes:
            source_text = source.get_text().strip()
            if source_text not in unique_sources:
                st.write(format_text(source_text))
                unique_sources.add(source_text)
        save_current_session()
    except Exception as e:
        st.error(f"Erreur lors de l'interaction avec le chatbot: {str(e)}")
