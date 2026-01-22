import streamlit as st
import os
import pandas as pd
import requests
import re
import time
import numpy as np
from typing import List, Dict
from dotenv import load_dotenv

# Import terbaru sesuai standar LangChain v0.1+
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Catatan: Kita menggunakan sentence-transformers lokal untuk efisiensi biaya
from sentence_transformers import SentenceTransformer
import faiss

# 1. Load Environment Variables
load_dotenv()

# 2. Konfigurasi Halaman
st.set_page_config(page_title="Asisten POLTESA", page_icon="🎓", layout="centered")

# --- 3. SCREEN LOADER ---
if "loaded" not in st.session_state:
    placeholder = st.empty()
    with placeholder.container():
        st.markdown("""
            <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 70vh;">
                <h1 style="color: #0e1117; font-family: sans-serif;">🎓 Sivita</h1>
                <p style="color: #555;">Menyiapkan Asisten Virtual Poltesa...</p>
                <div class="loader"></div>
            </div>
            <style>
                .loader {
                    border: 4px solid #f3f3f3;
                    border-top: 4px solid #3498db;
                    border-radius: 50%;
                    width: 80px;
                    height: 80px;
                    animation: spin 1s linear infinite;
                    margin-top: 20px;
                }
                @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
            </style>
        """, unsafe_allow_html=True)
        time.sleep(3) # Loader 3 detik agar tidak terlalu lama
    placeholder.empty()
    st.session_state["loaded"] = True

# --- 4. INISIALISASI SESSION STATE ---
if "last_answer" not in st.session_state:
    st.session_state["last_answer"] = ""
if "last_duration" not in st.session_state:
    st.session_state["last_duration"] = 0
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []

# --- FUNGSI VALIDASI EMAIL ---
def is_valid_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@gmail\.com$'
    return re.match(pattern, email) is not None

# --- FUNGSI CHUNKING ---
def create_chunks(text: str, chunk_size: int = 800, chunk_overlap: int = 100) -> List[str]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return text_splitter.split_text(text)

# --- FUNGSI: AMBIL & PROSES DATA ---
def get_and_process_data():
    try:
        central_url = st.secrets["SHEET_CENTRAL_URL"]
        df_list = pd.read_csv(central_url)
        tab_names = df_list['NamaTab'].tolist()
        base_url = central_url.split('/export')[0]
        
        all_data = []
        for tab in tab_names:
            tab_url = f"{base_url}/gviz/tq?tqx=out:csv&sheet={tab.replace(' ', '%20')}"
            try:
                df = pd.read_csv(tab_url)
                text_data = f"### DATA {tab.upper()} ###\n"
                for idx, row in df.iterrows():
                    row_text = ", ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                    text_data += f"- {row_text}\n"
                all_data.append(text_data)
            except Exception: continue
        
        combined_text = "\n\n".join(all_data)
        chunks = create_chunks(combined_text)
        
        chunk_metadata = [{"chunk_id": i, "text": chunk, "source": "google_sheets"} for i, chunk in enumerate(chunks)]
        return chunk_metadata
    except Exception as e:
        st.error(f"Gagal memproses data: {e}")
        return []

# --- FUNGSI: BUAT VECTOR STORE ---
def create_vector_store(chunks: List[Dict]):
    try:
        # Menggunakan model multilingual yang ringan dan gratis
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        texts = [chunk["text"] for chunk in chunks]
        embeddings = model.encode(texts, normalize_embeddings=True)
        
        embeddings_array = np.array(embeddings).astype('float32')
        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatIP(dimension) 
        
        faiss.normalize_L2(embeddings_array)
        index.add(embeddings_array)
        
        return {"index": index, "chunks": chunks, "model": model}
    except Exception as e:
        st.error(f"Error Vector Store: {e}")
        return None

# --- FUNGSI: SEMANTIC SEARCH ---
def semantic_search(query: str, vector_store: Dict, top_k: int = 3):
    query_embedding = vector_store["model"].encode([query], normalize_embeddings=True)
    query_embedding = np.array(query_embedding).astype('float32')
    faiss.normalize_L2(query_embedding)
    
    distances, indices = vector_store["index"].search(query_embedding, top_k)
    return [vector_store["chunks"][idx]["text"] for idx in indices[0] if idx < len(vector_store["chunks"])]

# --- FUNGSI: SIMPAN LOG ---
def save_to_log(email, question, answer, duration):
    try:
        log_url = st.secrets["LOG_URL"]
        payload = {"email": email, "question": question, "answer": answer, "duration": f"{duration}s"}
        requests.post(log_url, json=payload, timeout=5)
    except: pass

# --- INITIAL LOAD ---
if "data_initialized" not in st.session_state:
    with st.spinner("📥 Sinkronisasi Database Poltesa..."):
        chunks = get_and_process_data()
        if chunks:
            vs = create_vector_store(chunks)
            st.session_state.vector_store = vs
            st.session_state.chunks = chunks
            st.session_state.data_initialized = True

# --- UI LOGIC ---
def clear_text():
    st.session_state["user_input"] = ""

def generate_response_with_rag(user_email, user_input):
    start_time = time.time()
    try:
        # Search relevant info
        relevant_chunks = semantic_search(user_input, st.session_state.vector_store)
        context = "\n".join(relevant_chunks)
        
        # OpenRouter / Gemini Config
        model = ChatOpenAI(
            model="google/gemini-2.0-flash-lite-001",
            openai_api_key=st.secrets["OPENROUTER_API_KEY"],
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.3
        )
        
        prompt = f"{st.secrets['SYSTEM_PROMPT']}\n\nCONTEXT:\n{context}\n\nPERTANYAAN: {user_input}\n\nJAWABAN:"
        
        response = model.invoke(prompt)
        duration = round(time.time() - start_time, 2)
        
        st.session_state["last_answer"] = response.content
        st.session_state["last_duration"] = duration
        save_to_log(user_email, user_input, response.content, duration)
                
    except Exception as e:
        st.error(f"Terjadi kendala teknis. Silakan coba lagi. ({e})")

# --- UI LAYOUT ---
st.markdown("""
    <style>
    .stTextArea textarea { border-radius: 12px; }
    .stTextInput input { border-radius: 12px; }
    .stButton button { border-radius: 25px; background-color: #3498db; color: white; }
    .duration-info { font-size: 0.8rem; color: gray; font-style: italic; }
    </style>
    """, unsafe_allow_html=True)

st.title("🎓 Asisten Virtual Poltesa")
st.caption("Sivita v1.0 - Berbasis Artificial Intelligence & RAG")

with st.form("chat_form"):
    email = st.text_input("Email Gmail:", placeholder="nama@gmail.com")
    query = st.text_area("Apa yang ingin Anda ketahui tentang Poltesa?", placeholder="Contoh: Apa saja jurusan di Poltesa?")
    
    c1, c2 = st.columns(2)
    with c1:
        btn_submit = st.form_submit_button("Tanyakan 🚀", use_container_width=True)
    with c2:
        btn_clear = st.form_submit_button("Reset", on_click=clear_text, use_container_width=True)

if btn_submit:
    if not is_valid_email(email):
        st.error("Gunakan alamat @gmail.com yang valid.")
    elif not query:
        st.warning("Pertanyaan tidak boleh kosong.")
    else:
        generate_response_with_rag(email, query)

if st.session_state["last_answer"]:
    st.chat_message("assistant").write(st.session_state["last_answer"])
    st.markdown(f'<p class="duration-info">Ditemukan dalam {st.session_state["last_duration"]} detik</p>', unsafe_allow_html=True)

# --- SIDEBAR INFO ---
with st.sidebar:
    st.header("Sistem Status")
    if st.session_state.get("data_initialized"):
        st.success(f"Database Aktif: {len(st.session_state.chunks)} Dokumen")
    st.info("Sivita menggunakan pencarian semantik untuk memahami konteks pertanyaan Anda melampaui sekadar kata kunci.")
