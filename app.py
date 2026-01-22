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
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss

# 1. Load Environment Variables
load_dotenv()

# 2. Konfigurasi Halaman
st.set_page_config(page_title="Asisten POLTESA", page_icon="🎓", layout="wide")

# --- 3. FUNGSI LOGIKA RAG ---

def is_valid_email(email):
    return re.match(r'^[a-zA-Z0-9._%+-]+@gmail\.com$', email) is not None

@st.cache_data(show_spinner=False)
def get_and_process_data():
    """Mengambil data dari Google Sheets dan memproses per baris."""
    try:
        central_url = st.secrets["URL"]
        df_list = pd.read_csv(central_url)
        tab_names = df_list['NamaTab'].tolist()
        base_url = central_url.split('/export')[0]
        
        all_chunks = []
        for tab in tab_names:
            tab_url = f"{base_url}/gviz/tq?tqx=out:csv&sheet={tab.replace(' ', '%20')}"
            try:
                df = pd.read_csv(tab_url)
                # Memproses setiap baris sebagai satu kesatuan informasi (Penting untuk data Statistik)
                for idx, row in df.iterrows():
                    # Gabungkan kolom dan nilai menjadi kalimat deskriptif
                    row_content = f"Informasi dari {tab}: " + ", ".join([f"{col} adalah {val}" for col, val in row.items() if pd.notna(val)])
                    all_chunks.append({"text": row_content, "source": tab})
            except Exception:
                continue
        return all_chunks
    except Exception as e:
        st.error(f"Gagal memuat Google Sheets: {e}")
        return []

def create_vector_store(chunks_data: List[Dict]):
    """Memasukkan data ke dalam FAISS Vector Database."""
    try:
        # Menggunakan model multilingual agar paham konteks Indonesia
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        texts = [c["text"] for c in chunks_data]
        embeddings = model.encode(texts, normalize_embeddings=True)
        
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension) # Cosine Similarity
        faiss.normalize_L2(embeddings)
        index.add(embeddings.astype('float32'))
        
        return {"index": index, "chunks": chunks_data, "model": model}
    except Exception as e:
        st.error(f"Gagal membangun Vector DB: {e}")
        return None

def semantic_search(query: str, vector_store: Dict, top_k: int = 5):
    """Mencari data paling relevan menggunakan Vector Similarity."""
    query_vec = vector_store["model"].encode([query], normalize_embeddings=True)
    faiss.normalize_L2(query_vec)
    distances, indices = vector_store["index"].search(query_vec.astype('float32'), top_k)
    
    results = []
    for idx in indices[0]:
        if idx < len(vector_store["chunks"]):
            results.append(vector_store["chunks"][idx]["text"])
    return results

def safe_log(email, query, answer, duration):
    """Mengirim log ke Google Script dengan proteksi timeout."""
    try:
        log_url = st.secrets["LOG_URL"]
        payload = {"email": email, "query": query, "answer": answer, "time": f"{duration}s"}
        # Timeout ditambah ke 10 detik agar tidak mudah putus
        requests.post(log_url, json=payload, timeout=10)
    except:
        # Gagal log tidak boleh menghentikan jawaban AI ke user
        pass

# --- 4. INISIALISASI & SIDEBAR ---

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "last_answer" not in st.session_state:
    st.session_state["last_answer"] = ""

with st.sidebar:
    st.title("⚙️ Panel Kontrol")
    if st.button("🔄 Paksa Sinkronisasi Data"):
        st.cache_data.clear()
        st.session_state.vector_store = None
        st.rerun()
    
    st.divider()
    
    # Otomatis load data jika belum ada
    if st.session_state.vector_store is None:
        with st.spinner("Mensinkronkan Database..."):
            raw_data = get_and_process_data()
            if raw_data:
                st.session_state.vector_store = create_vector_store(raw_data)
                st.success(f"✅ {len(raw_data)} baris data aktif.")

    # Fitur Inspeksi (Untuk memastikan data dosen masuk)
    if st.session_state.vector_store:
        st.subheader("🔍 Cek Isi Database")
        test_word = st.text_input("Cari kata kunci di DB:", placeholder="Contoh: dosen")
        if test_word:
            matches = semantic_search(test_word, st.session_state.vector_store, top_k=3)
            for m in matches:
                st.caption(f"📍 {m}")

# --- 5. ANTARMUKA PENGGUNA ---

st.title("🎓 Asisten Virtual Poltesa (Sivita)")
st.markdown("---")

# Layout Form
with st.container(border=True):
    col_mail, col_empty = st.columns([2, 1])
    with col_mail:
        email = st.text_input("Gunakan Email Gmail Anda:", placeholder="contoh@gmail.com")
    
    user_query = st.text_area("Apa yang ingin Anda tanyakan?", placeholder="Tanyakan tentang jumlah mahasiswa, dosen, atau info kampus lainnya...")
    
    if st.button("Kirim Pertanyaan 🚀", use_container_width=True):
        if not is_valid_email(email):
            st.error("Email harus berakhiran @gmail.com")
        elif not user_query:
            st.warning("Mohon tuliskan pertanyaan Anda.")
        else:
            with st.spinner("Menganalisis data Poltesa..."):
                start_time = time.time()
                
                try:
                    # 1. Retrieval (Ambil Data Relevan)
                    context_list = semantic_search(user_query, st.session_state.vector_store)
                    context_text = "\n".join(context_list)
                    
                    # 2. Augmentation & Generation (LLM)
                    llm = ChatOpenAI(
                        model="google/gemini-2.0-flash-lite-001",
                        openai_api_key=st.secrets["OPENROUTER_API_KEY"],
                        openai_api_base="https://openrouter.ai/api/v1",
                        temperature=0.1 # Konsisten pada data
                    )
                    
                    sys_prompt = st.secrets["SYSTEM_PROMPT"]
                    full_prompt = f"{sys_prompt}\n\nREFERENSI DATA:\n{context_text}\n\nPERTANYAAN: {user_query}"
                    
                    response = llm.invoke(full_prompt)
                    st.session_state["last_answer"] = response.content
                    
                    # 3. Logging Aman
                    duration = round(time.time() - start_time, 2)
                    safe_log(email, user_query, response.content, duration)
                    
                except Exception as e:
                    st.error(f"Sivita mengalami gangguan teknis: {e}")

# Area Tampilan Jawaban
if st.session_state["last_answer"]:
    with st.chat_message("assistant"):
        st.write(st.session_state["last_answer"])
    st.caption("Jawaban ini dihasilkan berdasarkan data terbaru dari Google Sheets Poltesa.")

st.divider()
st.caption("Sivita v1.2 | Teknologi RAG (Retrieval-Augmented Generation)")
