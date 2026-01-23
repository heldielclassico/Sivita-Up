import streamlit as st
import os
import pandas as pd
import requests
import re
import time
import numpy as np
from typing import List, Dict, Tuple
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
    """Fungsi hapus chat: Menghapus tulisan di pertanyaan saja (jawaban tetap ada)."""
    st.session_state["user_query_input"] = ""

@st.cache_data(show_spinner=False)
def get_and_process_data() -> Tuple[List[Dict], str]:
    """Mengambil data & Prompt dari Google Sheets."""
    try:
        central_url = st.secrets["SHEET_CENTRAL_URL"]
        df_list = pd.read_csv(central_url)
        tab_names = df_list['NamaTab'].tolist()
        base_url = central_url.split('/export')[0]
        
        all_chunks = []
        fetched_prompt = "" 
        
        for tab in tab_names:
            tab_url = f"{base_url}/gviz/tq?tqx=out:csv&sheet={tab.replace(' ', '%20')}"
            try:
                df = pd.read_csv(tab_url)
                
                # JIKA TAB ADALAH 'Prompt', simpan ke variabel (Instruksi Sistem)
                if tab.lower() == 'prompt':
                    prompt_row = df[df['Nama'] == 'SYSTEM_PROMPT']
                    if not prompt_row.empty:
                        fetched_prompt = str(prompt_row.iloc[0]['Isi'])
                    continue
                
                # JIKA TAB ADALAH DATA (Diproses untuk Database FAISS)
                for idx, row in df.iterrows():
                    row_content = f"Data {tab}: " + ", ".join([f"{col} adalah {val}" for col, val in row.items() if pd.notna(val)])
                    all_chunks.append({"text": row_content, "source": tab})
            except Exception:
                continue
        return all_chunks, fetched_prompt
    except Exception as e:
        st.error(f"Gagal memuat Database: {e}")
        return [], ""

def create_vector_store(chunks_data: List[Dict]):
    """Membangun Vector Database menggunakan FAISS."""
    try:
        # Menggunakan model multilingual agar paham konteks Indonesia
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        texts = [c["text"] for c in chunks_data]
        embeddings = model.encode(texts, normalize_embeddings=True)
        
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension) # Cosine Similarity
        index.add(embeddings.astype('float32'))
        
        return {"index": index, "chunks": chunks_data, "model": model}
    except Exception as e:
        st.error(f"Gagal membangun Vector DB: {e}")
        return None

def semantic_search(query: str, vector_store: Dict, top_k: int = 5):
    """Mencari data paling relevan menggunakan Vector Similarity."""
    query_vec = vector_store["model"].encode([query], normalize_embeddings=True)
    distances, indices = vector_store["index"].search(query_vec.astype('float32'), top_k)
    
    results = []
    for idx in indices[0]:
        if idx < len(vector_store["chunks"]):
            results.append(vector_store["chunks"][idx]["text"])
    return results

def safe_log(email, query, answer, duration):
    """Mengirim log ke Google Script."""
    try:
        log_url = st.secrets["LOG_URL"]
        payload = {"email": email, "query": query, "answer": answer, "time": f"{duration}s"}
        requests.post(log_url, json=payload, timeout=10)
    except:
        pass

# --- 4. INISIALISASI SESSION STATE ---

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "dynamic_sys_prompt" not in st.session_state:
    st.session_state.dynamic_sys_prompt = ""
if "last_answer" not in st.session_state:
    st.session_state["last_answer"] = ""
if "last_duration" not in st.session_state:
    st.session_state["last_duration"] = 0

# --- 5. SIDEBAR & SINKRONISASI ---

with st.sidebar:
    st.title("⚙️ Panel Kontrol")
    if st.button("🔄 Sinkronkan Ulang Data & Prompt"):
        st.cache_data.clear()
        st.session_state.vector_store = None
        st.session_state.dynamic_sys_prompt = ""
        st.rerun()
    
    st.divider()
    
    # Otomatis load data jika belum ada
    if st.session_state.vector_store is None:
        with st.spinner("Mensinkronkan Database & Prompt..."):
            raw_data, dyn_prompt = get_and_process_data()
            if raw_data:
                st.session_state.vector_store = create_vector_store(raw_data)
                st.session_state.dynamic_sys_prompt = dyn_prompt
                st.success(f"✅ {len(raw_data)} baris data & Prompt aktif.")

# --- 6. ANTARMUKA PENGGUNA ---

st.title("🎓 Asisten Virtual Poltesa (Sivita)")
st.markdown("<p style='margin-top: -20px; color: gray;'>Sivita v1.3 | RAG & Cloud Prompt</p>", unsafe_allow_html=True)

# Layout Form
with st.container(border=True):
    email = st.text_input("Gunakan Email Gmail Anda:", placeholder="contoh@gmail.com")
    
    # Text area menggunakan key agar bisa dihapus secara programatik
    user_query = st.text_area(
        "Apa yang ingin Anda tanyakan?", 
        placeholder="Tanyakan tentang jumlah mahasiswa, dosen, atau info kampus lainnya...",
        key="user_query_input"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        btn_kirim = st.button("Kirim Pertanyaan 🚀", use_container_width=True)
    with col2:
        # Tombol hapus hanya membersihkan kolom pertanyaan
        st.button("Hapus Pertanyaan 🗑️", on_click=clear_input_only, use_container_width=True)
    
    if btn_kirim:
        if not is_valid_email(email):
            st.error("Email harus berakhiran @gmail.com")
        elif not user_query:
            st.warning("Mohon tuliskan pertanyaan Anda.")
        else:
            with st.spinner("Menganalisis data Poltesa..."):
                start_time = time.time()
                try:
                    # 1. Retrieval (Ambil Data Relevan dari FAISS)
                    context_list = semantic_search(user_query, st.session_state.vector_store)
                    context_text = "\n".join(context_list)
                    
                    # 2. Augmentation & Generation (LLM)
                    llm = ChatOpenAI(
                        model="google/gemini-2.0-flash-lite-001",
                        openai_api_key=st.secrets["OPENROUTER_API_KEY"],
                        openai_api_base="https://openrouter.ai/api/v1",
                        temperature=0.1 
                    )
                    
                    # Gunakan Prompt dari Google Sheets (Jika kosong, gunakan default)
                    sys_prompt = st.session_state.get("dynamic_sys_prompt", "Anda adalah Sivita, asisten Poltesa.")
                    
                    full_prompt = f"{sys_prompt}\n\nREFERENSI DATA:\n{context_text}\n\nPERTANYAAN: {user_query}"
                    
                    response = llm.invoke(full_prompt)
                    
                    # Simpan hasil jawaban ke state
                    st.session_state["last_answer"] = response.content
                    st.session_state["last_duration"] = round(time.time() - start_time, 2)
                    
                    # 3. Logging Aman
                    safe_log(email, user_query, response.content, st.session_state["last_duration"])
                    
                except Exception as e:
                    st.error(f"Sivita mengalami gangguan teknis: {e}")

# Area Tampilan Jawaban (Akan tetap tampil meski pertanyaan dihapus)
if st.session_state["last_answer"]:
    st.markdown("---")
    with st.chat_message("assistant"):
        st.markdown(st.session_state["last_answer"])
    st.caption(f"⏱️ Pencarian selesai dalam {st.session_state['last_duration']} detik")

st.divider()
st.caption("Sivita - Virtual Assistant Poltesa @2026")
