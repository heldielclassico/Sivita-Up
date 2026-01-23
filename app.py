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

# --- KODE CSS UNTUK TAMPILAN STANDAR (FLOW NORMAL) ---
st.markdown(f"""
    <style>
    /* Sembunyikan elemen dekoratif bawaan Streamlit */
    header[data-testid="stHeader"] {{ display: none !important; }}
    footer {{ visibility: hidden !important; }}
    #MainMenu {{ visibility: hidden !important; }}
    .stAppDeployButton {{ display: none !important; }}

    /* Atur margin konten agar rapi */
    .block-container {{
        max-width: 800px;
        padding-top: 2rem;
        padding-bottom: 5rem;
    }}

    /* Styling Answer Box */
    .answer-box {{
        padding: 25px;
        background-color: #f8f9fa;
        border-radius: 15px;
        border: 1px solid #e0e0e0;
        line-height: 1.7;
        color: #1f2937;
        margin-bottom: 20px;
    }}

    /* Area Input Wrapper untuk pemisah visual */
    .input-section {{
        padding: 20px;
        background-color: #ffffff;
        border-top: 1px solid #eee;
        margin-top: 30px;
    }}
    </style>
    """, unsafe_allow_html=True)

# --- 3. FUNGSI LOGIKA ---

def is_valid_email(email):
    return re.match(r'^[a-zA-Z0-9._%+-]+@gmail\.com$', email) is not None

def clear_input_only():
    st.session_state["user_query_input"] = ""

def clear_answer_only():
    st.session_state["last_answer"] = ""
    st.session_state["last_duration"] = 0

@st.cache_data(show_spinner=False)
def get_and_process_data() -> Tuple[List[Dict], str]:
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
                    if 'Isi' in df.columns: full_instructions = df['Isi'].dropna().astype(str).tolist()
                    continue
                for _, row in df.iterrows():
                    content = f"Data {tab}: " + ", ".join([f"{c} adalah {v}" for c, v in row.items() if pd.notna(v)])
                    all_chunks.append({"text": content, "source": tab})
            except Exception: continue
        return all_chunks, "\n".join(full_instructions) if full_instructions else "Anda adalah Sivita."
    except Exception: return [], ""

def create_vector_store(chunks_data: List[Dict]):
    try:
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        texts = [c["text"] for c in chunks_data]
        embeddings = model.encode(texts, normalize_embeddings=True)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings.astype('float32'))
        return {"index": index, "chunks": chunks_data, "model": model}
    except Exception: return None

def semantic_search(query: str, vector_store: Dict):
    query_vec = vector_store["model"].encode([query], normalize_embeddings=True)
    _, indices = vector_store["index"].search(query_vec.astype('float32'), 5)
    return [vector_store["chunks"][idx]["text"] for idx in indices[0] if idx < len(vector_store["chunks"])]

# --- 4. INISIALISASI ---
for key, val in [("vector_store", None), ("dynamic_sys_prompt", ""), ("last_answer", ""), ("last_duration", 0)]:
    if key not in st.session_state: st.session_state[key] = val

if st.session_state.vector_store is None:
    raw_data, dyn_prompt = get_and_process_data()
    if raw_data:
        st.session_state.vector_store = create_vector_store(raw_data)
        st.session_state.dynamic_sys_prompt = dyn_prompt

# --- 5. RENDER UI ---

# BAGIAN 1: ATAS (STATIS)
st.markdown("<h1 style='text-align: center;'>🎓 Sivita Poltesa</h1>", unsafe_allow_html=True)
with st.expander("⚙️ Konfigurasi Akun & Data", expanded=False):
    email = st.text_input("Email Gmail:", placeholder="nama@gmail.com")
    if st.button("🔄 Perbarui Data", use_container_width=True):
        st.cache_data.clear()
        st.session_state.vector_store = None
        st.rerun()

st.divider()

# BAGIAN 2: JAWABAN (TENGAH)
if st.session_state["last_answer"]:
    st.markdown("### 🤖 Respon Sivita")
    st.markdown(f'<div class="answer-box">{st.session_state["last_answer"]}</div>', unsafe_allow_html=True)
    st.caption(f"⏱️ Diproses dalam {st.session_state['last_duration']} detik")
    st.button("Hapus Jawaban ✨", on_click=clear_answer_only)
else:
    st.info("Halo! Silakan ajukan pertanyaan seputar kampus Poltesa pada kolom di bawah.")

# BAGIAN 3: INPUT (BAWAH)
st.markdown('<div class="input-section">', unsafe_allow_html=True)
user_query = st.text_area(
    "Apa yang ingin Anda ketahui?", 
    placeholder="Contoh: Bagaimana cara daftar ulang?", 
    key="user_query_input", 
    height=120
)

col_send, col_clear = st.columns([1.5, 1])
with col_send:
    btn_kirim = st.button("Kirim Pertanyaan 🚀", use_container_width=True, type="primary")
with col_clear:
    st.button("Bersihkan Teks 🗑️", on_click=clear_input_only, use_container_width=True)

st.markdown("<br><p style='text-align: center; color: #888; font-size: 0.8rem;'>Sivita Virtual Assistant Poltesa @2026</p>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- 6. LOGIKA BACKEND ---
if btn_kirim:
    if not is_valid_email(email):
        st.error("Masukkan email @gmail.com yang valid untuk melanjutkan.")
    elif not user_query:
        st.warning("Silakan ketik pertanyaan terlebih dahulu.")
    else:
        with st.spinner("Sivita sedang mencari jawaban..."):
            start_time = time.time()
            try:
                context = "\n".join(semantic_search(user_query, st.session_state.vector_store))
                llm = ChatOpenAI(
                    model="google/gemini-2.0-flash-lite-001",
                    openai_api_key=st.secrets["OPENROUTER_API_KEY"],
                    openai_api_base="https://openrouter.ai/api/v1",
                    temperature=0.1
                )
                full_prompt = f"{st.session_state.dynamic_sys_prompt}\n\nREFERENSI:\n{context}\n\nPERTANYAAN: {user_query}"
                res = llm.invoke(full_prompt)
                st.session_state["last_answer"] = res.content
                st.session_state["last_duration"] = round(time.time() - start_time, 2)
                st.rerun()
            except Exception as e:
                st.error(f"Terjadi kesalahan teknis: {e}")
