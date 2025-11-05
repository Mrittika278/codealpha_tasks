import streamlit as st
import os
import pandas as pd
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter

# 1. --- CONFIGURATION AND SETUP ---
st.title("⚖️ Urimaikural: Your Basic Rights Educator")

# Define the data path and the specific list of files to load
DATA_DIR = "data"

# List of the four specific files you want to load (all must be in the 'data' folder)
INPUT_FILES = [
    os.path.join(DATA_DIR, 'ipc_sections.csv'),
    #os.path.join(DATA_DIR, 'ipc_qa.json'),
    #os.path.join(DATA_DIR, 'crpc_qa.json'),
    #os.path.join(DATA_DIR, 'constitution_qa.json')
]

# Define your custom System Prompt (Persona)
SYSTEM_PROMPT = """
You are Urimalkural, a friendly legal aid assistant helping Indian citizens understand their basic legal rights and laws. 

Your personality:
- Warm, encouraging, and patient
- Use simple language (avoid legal jargon)
- Explain laws as if talking to a friend
- Always cite the relevant law/act/article
- Empowering tone: "You have the right to..."
- If you don't know, guide them to approach legal aid services
- Focus on basic rights under IPC, CrPC, and the Constitution of India
- Avoid giving direct legal advice; instead, educate about rights and procedures.
-if asked about topics outside Indian law, politely inform the user that your expertise is limited to Indian legal rights and laws.
- Always prioritize user safety and confidentiality.
-Always refer to the accurate ipc section number related to the context they ask and when you respond them with examples from real life scenarios where applicable.
- Maintain a respectful and non-judgmental attitude at all times.
-Never say i dont know, instead say "Based on my knowledge of Indian legal rights, here's what I can share..."
- Ensure your responses align with the latest legal standards and practices in India.
"""

# 2. --- RAG INDEXING FUNCTION ---
@st.cache_resource
def get_index():
    # 2a. CHECK FOR DATA DIRECTORY
    if not os.path.exists(DATA_DIR):
        st.error(f"Error: Data directory '{DATA_DIR}' not found. Please create it and add your legal files.")
        return None
    
    # 2b. API KEY CHECK AND ASSIGNMENT (MUST BE INSIDE THE FUNCTION)
    # Checks for key in environment, which includes Streamlit secrets.
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 
    
    if not GEMINI_API_KEY and "GEMINI_API_KEY" in st.secrets:
        GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

    if not GEMINI_API_KEY:
        st.error("FATAL ERROR: GEMINI_API_KEY environment variable not found. Please set it securely in your Streamlit secrets.")
        return None
    
    # 2c. LLAMAINDEX SETUP (Pass the API key directly)
    Settings.llm = GoogleGenAI(model="gemini-2.5-flash", api_key=GEMINI_API_KEY)
    Settings.embed_model = GoogleGenAIEmbedding(model="text-embedding-004", api_key=GEMINI_API_KEY)
    Settings.node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
    
    # 2d. LOAD ALL FOUR SPECIFIC FILES
    try:
        # SimpleDirectoryReader using the specific INPUT_FILES list
        documents = SimpleDirectoryReader(input_files=INPUT_FILES).load_data()
    except Exception as e:
        st.error(f"Error loading documents. Ensure all 4 files are correctly formatted and exist: {e}")
        return None

    if not documents:
        st.error(f"Error: No documents were loaded from the file list.")
        return None

    # 2e. CREATE THE VECTOR INDEX
    index = VectorStoreIndex.from_documents(documents)
    
    return index # Last line of the function, aligned with 2d.

# 3. --- CHAT ENGINE SETUP ---
# 3. --- CHAT ENGINE SETUP ---
if "chat_engine" not in st.session_state:
    index = get_index() # Calls the function to get the index object
    
    if index:
        # NOTE: We DO NOT define 'retriever = index.as_retriever()' here anymore.
        # This line caused the conflict. We rely on the high-level method.

        # Create the ChatEngine using the most stable arguments that don't conflict.
        st.session_state.chat_engine = index.as_chat_engine(
            chat_mode="context", 
            # REMOVED: retriever=retriever, <--- This was the conflicting argument
            system_prompt=SYSTEM_PROMPT
        )

# 4. --- STREAMLIT UI AND CHAT LOOP ---
# Initialize message history
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am Urimalkural, your AI legal educator. How can I help you understand your rights today?"}
    ]

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Handle user input
if prompt := st.chat_input("Ask about an IPC section or your rights..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Generate assistant response
    if "chat_engine" in st.session_state:
        with st.chat_message("assistant"):
            with st.spinner("Urimalkural is thinking..."):
                response = st.session_state.chat_engine.chat(prompt)
                st.write(response.response)
                st.session_state.messages.append({"role": "assistant", "content": response.response})