import streamlit as st
import os
import pandas as pd
import requests
import re
import time
import numpy as np
from typing import List, Dict
from dotenv import load_dotenv

# Import LangChain v0.1+
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss

# 1. Load Environment Variables
load_dotenv()

# 2. Konfigurasi Halaman
st.set_page_config(page_title="Asisten POLTESA", page_icon="🎓", layout="wide")

# --- 3. FUNGSI UTAMA ---

def is_valid_email(email):
    return re.match(r'^[a-zA-Z0-9._%+-]+@gmail\.com$', email) is not None

def create_chunks(text: str):
    # Menggunakan separator yang lebih cerdas agar data tabel tidak terpotong sembarangan
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n### ", "\n\n", "\n", ". ", " "]
    )
    return text_splitter.split_text(text)

@st.cache_data(show_spinner=False)
def get_and_process_data():
    """Mengambil data dari Google Sheets dan memecahnya menjadi chunks."""
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
                # Ubah setiap baris menjadi teks deskriptif agar mudah dicari secara semantik
                for idx, row in df.iterrows():
                    row_content = f"Data {tab}: " + ", ".join([f"{col} adalah {val}" for col, val in row.items() if pd.notna(val)])
                    all_chunks.append({"text": row_content, "source": tab})
            except Exception:
                continue
        return all_chunks
    except Exception as e:
        st.error(f"Koneksi ke Google Sheets Gagal: {e}")
        return []

def create_vector_store(chunks_data: List[Dict]):
    """Memasukkan chunks ke dalam FAISS Vector Database."""
    try:
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        texts = [c["text"] for c in chunks_data]
        embeddings = model.encode(texts, normalize_embeddings=True)
        
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension) 
        faiss.normalize_L2(embeddings)
        index.add(embeddings.astype('float32'))
        
        return {"index": index, "chunks": chunks_data, "model": model}
    except Exception as e:
        st.error(f"Gagal memproses Vector Store: {e}")
        return None

def semantic_search(query: str, vector_store: Dict, top_k: int = 5):
    """Mencari data paling relevan berdasarkan makna kata."""
    query_vec = vector_store["model"].encode([query], normalize_embeddings=True)
    faiss.normalize_L2(query_vec)
    distances, indices = vector_store["index"].search(query_vec.astype('float32'), top_k)
    
    results = []
    for idx in indices[0]:
        if idx < len(vector_store["chunks"]):
            results.append(vector_store["chunks"][idx]["text"])
    return results

# --- 4. LOGIC INISIALISASI ---

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "last_answer" not in st.session_state:
    st.session_state["last_answer"] = ""

# Tombol Sinkronisasi di Sidebar
with st.sidebar:
    st.title("⚙️ Kontrol Database")
    if st.button("🔄 Sinkronkan Ulang Google Sheets"):
        st.cache_data.clear()
        st.session_state.vector_store = None
        st.rerun()
    
    st.divider()
    
    # Loader Data
    if st.session_state.vector_store is None:
        with st.spinner("Mengunduh data Poltesa..."):
            raw_data = get_and_process_data()
            if raw_data:
                st.session_state.vector_store = create_vector_store(raw_data)
                st.success(f"Berhasil memuat {len(raw_data)} baris data.")
    
    # Fitur Cek Data (Menjawab pertanyaan user: "Bagaimana cara tahu data sudah masuk?")
    if st.session_state.vector_store:
        st.subheader("🔍 Inspeksi Database")
        search_test = st.text_input("Tes cari kata di DB:", placeholder="Contoh: dosen")
        if search_test:
            matches = semantic_search(search_test, st.session_state.vector_store, top_k=3)
            for m in matches:
                st.caption(f"📍 {m}")

# --- 5. ANTARMUKA UTAMA ---

st.title("🎓 Asisten Virtual Poltesa (Sivita)")
st.info("Tanyakan informasi seputar jumlah mahasiswa, dosen, jurusan, atau fasilitas Poltesa.")

with st.container(border=True):
    email = st.text_input("Konfirmasi Email Gmail:", placeholder="user@gmail.com")
    user_query = st.text_area("Apa yang ingin Anda tanyakan?", placeholder="Contoh: Berapa jumlah dosen di Poltesa?")
    
    if st.button("Tanyakan ke Sivita 🚀", use_container_width=True):
        if not is_valid_email(email):
            st.error("Gunakan Gmail yang valid!")
        elif not user_query:
            st.warning("Masukkan pertanyaan Anda.")
        else:
            with st.spinner("Sivita sedang berpikir..."):
                start_time = time.time()
                
                # 1. Cari data di Vector DB
                context_list = semantic_search(user_query, st.session_state.vector_store)
                context_text = "\n".join(context_list)
                
                # 2. Kirim ke AI (OpenRouter)
                try:
                    llm = ChatOpenAI(
                        model="google/gemini-2.0-flash-lite-001",
                        openai_api_key=st.secrets["OPENROUTER_API_KEY"],
                        openai_api_base="https://openrouter.ai/api/v1",
                        temperature=0.1 # Rendah agar AI tidak mengarang
                    )
                    
                    system_prompt = st.secrets["SYSTEM_PROMPT"]
                    full_prompt = f"{system_prompt}\n\nREFERENSI DATA:\n{context_text}\n\nPERTANYAAN: {user_query}"
                    
                    response = llm.invoke(full_prompt)
                    st.session_state["last_answer"] = response.content
                    
                    # Simpan Log (Optional)
                    duration = round(time.time() - start_time, 2)
                    log_payload = {"email": email, "query": user_query, "answer": response.content, "time": duration}
                    requests.post(st.secrets["LOG_URL"], json=log_payload, timeout=2)
                    
                except Exception as e:
                    st.error(f"Gagal menghubungi otak AI: {e}")

# Tampilkan Jawaban
if st.session_state["last_answer"]:
    st.chat_message("assistant").write(st.session_state["last_answer"])

st.divider()
st.caption("Sivita v1.1 | Powered by RAG & Sentence-Transformers")
