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
    """Hanya menghapus teks pertanyaan di kolom input."""
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
        full_instructions = []
        
        for tab in tab_names:
            tab_url = f"{base_url}/gviz/tq?tqx=out:csv&sheet={tab.replace(' ', '%20')}"
            try:
                df = pd.read_csv(tab_url)
                if tab.lower() == 'prompt':
                    if 'Isi' in df.columns:
                        full_instructions = df['Isi'].dropna().astype(str).tolist()
                    continue
                
                for idx, row in df.iterrows():
                    row_content = f"Data {tab}: " + ", ".join([f"{col} adalah {val}" for col, val in row.items() if pd.notna(val)])
                    all_chunks.append({"text": row_content, "source": tab})
            except Exception:
                continue
        
        final_prompt = "\n".join(full_instructions) if full_instructions else "Anda adalah Sivita."
        return all_chunks, final_prompt
    except Exception as e:
        st.error(f"Gagal memuat Database: {e}")
        return [], ""

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
    """Mencari referensi data paling relevan."""
    query_vec = vector_store["model"].encode([query], normalize_embeddings=True)
    distances, indices = vector_store["index"].search(query_vec.astype('float32'), top_k)
    results = [vector_store["chunks"][idx]["text"] for idx in indices[0] if idx < len(vector_store["chunks"])]
    return results

# --- FUNGSI: SIMPAN LOG (DIREVISI) ---
def save_to_log(email, question, answer="", duration=0):
    """Mengirim data log ke Google Apps Script."""
    try:
        log_url = st.secrets["LOG_URL"]
        # Menggunakan key 'query' agar terbaca di kolom pertanyaan Google Sheets
        payload = {
            "email": email,
            "query": question, 
            "answer": answer,
            "time": f"{duration} detik"
        }
        requests.post(log_url, json=payload, timeout=10)
    except Exception as e:
        # Log error internal (opsional)
        pass

# --- 4. INISIALISASI & SIDEBAR ---

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "dynamic_sys_prompt" not in st.session_state:
    st.session_state.dynamic_sys_prompt = ""
if "last_answer" not in st.session_state:
    st.session_state["last_answer"] = ""
if "last_duration" not in st.session_state:
    st.session_state["last_duration"] = 0

with st.sidebar:
    st.title("⚙️ Panel Sivita")
    if st.button("🔄 Sinkronkan Ulang Data & Prompt"):
        st.cache_data.clear()
        st.session_state.vector_store = None
        st.rerun()
    
    if st.session_state.vector_store is None:
        with st.spinner("Mensinkronkan Data & Instruksi..."):
            raw_data, dyn_prompt = get_and_process_data()
            if raw_data:
                st.session_state.vector_store = create_vector_store(raw_data)
                st.session_state.dynamic_sys_prompt = dyn_prompt
                st.success(f"✅ Database & {len(dyn_prompt.splitlines())} baris instruksi aktif.")

# --- 5. UI UTAMA ---

st.title("🎓 Asisten Virtual Poltesa (Sivita)")
st.markdown("<p style='margin-top: -20px; color: gray;'>Sivita v1.3 | Modular Prompt System</p>", unsafe_allow_html=True)

with st.container(border=True):
    email = st.text_input("Email Gmail Anda:", placeholder="nama@gmail.com")
    user_query = st.text_area("Apa yang ingin Anda tanyakan?", placeholder="Tanyakan info kampus...", key="user_query_input")
    
    col1, col2 = st.columns(2)
    with col1:
        btn_kirim = st.button("Kirim Pertanyaan 🚀", use_container_width=True)
    with col2:
        st.button("Hapus Pertanyaan 🗑️", on_click=clear_input_only, use_container_width=True)

    if btn_kirim:
        if not is_valid_email(email):
            st.error("Gunakan email @gmail.com")
        elif not user_query:
            st.warning("Tuliskan pertanyaan.")
        elif st.session_state.vector_store is None:
            st.error("Database belum siap. Silakan klik sinkronisasi.")
        else:
            with st.spinner("Mencari jawaban..."):
                start_time = time.time()
                try:
                    context_list = semantic_search(user_query, st.session_state.vector_store)
                    context_text = "\n".join(context_list)
                    
                    llm = ChatOpenAI(
                        model="google/gemini-2.0-flash-lite-001",
                        openai_api_key=st.secrets["OPENROUTER_API_KEY"],
                        openai_api_base="https://openrouter.ai/api/v1",
                        temperature=0.1
                    )
                    
                    full_prompt = f"{st.session_state.dynamic_sys_prompt}\n\nREFERENSI DATA:\n{context_text}\n\nPERTANYAAN: {user_query}"
                    
                    response = llm.invoke(full_prompt)
                    st.session_state["last_answer"] = response.content
                    st.session_state["last_duration"] = round(time.time() - start_time, 2)
                    
                    # Memanggil fungsi log yang sudah diperbaiki
                    save_to_log(email, user_query, response.content, st.session_state["last_duration"])
                except Exception as e:
                    st.error(f"Error: {e}")

if st.session_state["last_answer"]:
    st.markdown("---")
    with st.chat_message("assistant"):
        st.markdown(st.session_state["last_answer"])
    st.caption(f"⏱️ Selesai dalam {st.session_state['last_duration']} detik")

st.divider()
st.caption("Sivita - Virtual Assistant Poltesa @2026")
