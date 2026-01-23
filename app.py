import streamlit as st
import os
import pandas as pd
import requests
import re
import time
import numpy as np
from typing import List, Dict
from dotenv import load_dotenv

# Import LangChain & AI
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer
import faiss

# 1. Load Environment Variables
load_dotenv()

# 2. Konfigurasi Halaman
st.set_page_config(page_title="Asisten POLTESA", page_icon="🎓", layout="centered")

# --- 3. FUNGSI LOGIKA & RAG ---

def is_valid_email(email):
    return re.match(r'^[a-zA-Z0-9._%+-]+@gmail\.com$', email) is not None

def clear_input_only():
    """Fungsi khusus hanya untuk menghapus teks di kolom pertanyaan."""
    st.session_state["user_query_input"] = ""
    # Jawaban tidak dihapus agar user tetap bisa membaca hasil sebelumnya

@st.cache_data(show_spinner=False)
def get_and_process_data():
    """Mengambil data dari Google Sheets dan memproses per baris."""
    try:
        central_url = st.secrets["SHEET_CENTRAL_URL"]
        df_list = pd.read_csv(central_url)
        tab_names = df_list['NamaTab'].tolist()
        base_url = central_url.split('/export')[0]
        
        all_chunks = []
        for tab in tab_names:
            tab_url = f"{base_url}/gviz/tq?tqx=out:csv&sheet={tab.replace(' ', '%20')}"
            try:
                df = pd.read_csv(tab_url)
                for idx, row in df.iterrows():
                    row_content = f"Data {tab}: " + ", ".join([f"{col} adalah {val}" for col, val in row.items() if pd.notna(val)])
                    all_chunks.append({"text": row_content, "source": tab})
            except Exception:
                continue
        return all_chunks
    except Exception as e:
        st.error(f"Gagal memuat Database: {e}")
        return []

def create_vector_store(chunks_data: List[Dict]):
    """Membangun Vector Database (FAISS)."""
    try:
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        texts = [c["text"] for c in chunks_data]
        embeddings = model.encode(texts, normalize_embeddings=True)
        
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings.astype('float32'))
        
        return {"index": index, "chunks": chunks_data, "model": model}
    except Exception as e:
        st.error(f"Gagal membangun Vector DB: {e}")
        return None

def semantic_search(query: str, vector_store: Dict, top_k: int = 5):
    """Mencari data paling relevan."""
    query_vec = vector_store["model"].encode([query], normalize_embeddings=True)
    distances, indices = vector_store["index"].search(query_vec.astype('float32'), top_k)
    
    results = []
    for idx in indices[0]:
        if idx < len(vector_store["chunks"]):
            results.append(vector_store["chunks"][idx]["text"])
    return results

def safe_log(email, query, answer, duration):
    """Mengirim log ke Google Sheets."""
    try:
        log_url = st.secrets["LOG_URL"]
        payload = {"email": email, "query": query, "answer": answer, "time": f"{duration}s"}
        requests.post(log_url, json=payload, timeout=10)
    except:
        pass

# --- 4. INISIALISASI SESSION STATE ---

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "last_answer" not in st.session_state:
    st.session_state["last_answer"] = ""
if "last_duration" not in st.session_state:
    st.session_state["last_duration"] = 0

# --- 5. SIDEBAR & SINKRONISASI ---

with st.sidebar:
    st.title("⚙️ Panel Sivita")
    if st.button("🔄 Sinkronkan Ulang Data"):
        st.cache_data.clear()
        st.session_state.vector_store = None
        st.rerun()
    
    if st.session_state.vector_store is None:
        with st.spinner("Mengunduh data Poltesa..."):
            raw_data = get_and_process_data()
            if raw_data:
                st.session_state.vector_store = create_vector_store(raw_data)
                st.success(f"✅ {len(raw_data)} data aktif.")

# --- 6. ANTARMUKA UTAMA ---

st.title("🎓 Asisten Virtual Poltesa (Sivita)")
st.markdown("<p style='margin-top: -20px; color: gray;'>Sivita v1.2 | RAG Technology</p>", unsafe_allow_html=True)

# Container Input
with st.container(border=True):
    email = st.text_input("Email Gmail Anda:", placeholder="nama@gmail.com")
    
    # Text area dengan KEY tetap untuk penghapusan
    user_query = st.text_area(
        "Apa yang ingin Anda tanyakan?", 
        placeholder="Tanyakan info kampus di sini...",
        key="user_query_input"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        btn_kirim = st.button("Kirim Pertanyaan 🚀", use_container_width=True)
    
    with col2:
        # Tombol Hapus hanya memanggil clear_input_only
        st.button("Hapus Pertanyaan 🗑️", on_click=clear_input_only, use_container_width=True)

    if btn_kirim:
        if not is_valid_email(email):
            st.error("Gunakan format email @gmail.com yang valid.")
        elif not user_query:
            st.warning("Masukkan pertanyaan Anda.")
        else:
            with st.spinner("Mencari jawaban..."):
                start_time = time.time()
                try:
                    # Proses RAG
                    context_list = semantic_search(user_query, st.session_state.vector_store)
                    context_text = "\n".join(context_list)
                    
                    llm = ChatOpenAI(
                        model="google/gemini-2.0-flash-lite-001",
                        openai_api_key=st.secrets["OPENROUTER_API_KEY"],
                        openai_api_base="https://openrouter.ai/api/v1",
                        temperature=0.1
                    )
                    
                    sys_prompt = st.secrets["SYSTEM_PROMPT"]
                    full_prompt = f"{sys_prompt}\n\nREFERENSI DATA:\n{context_text}\n\nPERTANYAAN: {user_query}"
                    
                    response = llm.invoke(full_prompt)
                    
                    # Simpan hasil jawaban ke session state
                    st.session_state["last_answer"] = response.content
                    st.session_state["last_duration"] = round(time.time() - start_time, 2)
                    
                    # Log data
                    safe_log(email, user_query, response.content, st.session_state["last_duration"])
                except Exception as e:
                    st.error(f"Gangguan teknis: {e}")

# Tampilan Jawaban (Tetap ada meskipun input pertanyaan dihapus)
if st.session_state["last_answer"]:
    st.markdown("---")
    with st.chat_message("assistant"):
        st.markdown(st.session_state["last_answer"])
    st.caption(f"⏱️ Selesai dalam {st.session_state['last_duration']} detik")

st.divider()
st.caption("Sivita - Virtual Assistant Poltesa @2026")
