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

# --- KODE CSS UNTUK LAYOUT KAKU (FIXED) ---
st.markdown(f"""
    <style>
    /* 1. Sembunyikan elemen bawaan Streamlit */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    .stAppDeployButton {{display: none;}}

    /* 2. Kunci layar utama agar tidak bisa scroll */
    .stApp {{
        overflow: hidden;
        height: 100vh;
    }}

    /* 3. Header Tetap di Atas */
    .fixed-header {{
        position: fixed;
        top: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 100%;
        max-width: 800px;
        background-color: white;
        z-index: 1000;
        padding: 10px 20px;
        border-bottom: 1px solid #eee;
    }}

    /* 4. Area Tengah (Hanya Bagian Ini yang Bisa Scroll) */
    .scrollable-container {{
        position: absolute;
        top: 160px; /* Jarak dari atas (disesuaikan dengan tinggi header) */
        bottom: 240px; /* Jarak dari bawah (disesuaikan dengan tinggi footer) */
        left: 50%;
        transform: translateX(-50%);
        width: 100%;
        max-width: 750px;
        overflow-y: auto;
        padding: 10px 20px;
    }}

    /* 5. Footer Tetap di Bawah */
    .fixed-footer {{
        position: fixed;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 100%;
        max-width: 800px;
        background-color: white;
        z-index: 1000;
        padding: 15px 20px;
        border-top: 1px solid #eee;
    }}

    /* Styling Answer Box */
    .answer-box {{
        padding: 20px;
        background-color: #f8f9fa;
        border-radius: 12px;
        border: 1px solid #e0e0e0;
        margin-bottom: 10px;
    }}
    </style>
    """, unsafe_allow_html=True)

# --- 3. FUNGSI LOGIKA & RAG ---

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
                    if 'Isi' in df.columns:
                        full_instructions = df['Isi'].dropna().astype(str).tolist()
                    continue
                for idx, row in df.iterrows():
                    row_content = f"Data {tab}: " + ", ".join([f"{col} adalah {val}" for col, val in row.items() if pd.notna(val)])
                    all_chunks.append({"text": row_content, "source": tab})
            except Exception: continue
        final_prompt = "\n".join(full_instructions) if full_instructions else "Anda adalah Sivita."
        return all_chunks, final_prompt
    except Exception as e:
        return [], ""

def create_vector_store(chunks_data: List[Dict]):
    try:
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        texts = [c["text"] for c in chunks_data]
        embeddings = model.encode(texts, normalize_embeddings=True)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings.astype('float32'))
        return {"index": index, "chunks": chunks_data, "model": model}
    except Exception: return None

def semantic_search(query: str, vector_store: Dict, top_k: int = 5):
    query_vec = vector_store["model"].encode([query], normalize_embeddings=True)
    distances, indices = vector_store["index"].search(query_vec.astype('float32'), top_k)
    return [vector_store["chunks"][idx]["text"] for idx in indices[0] if idx < len(vector_store["chunks"])]

def save_to_log(email, question, answer="", duration=0):
    try:
        requests.post(st.secrets["LOG_URL"], json={"email": email, "question": question, "answer": answer, "duration": f"{duration} detik"}, timeout=5)
    except Exception: pass

# --- 4. INISIALISASI ---

for key, val in [("vector_store", None), ("dynamic_sys_prompt", ""), ("last_answer", ""), ("last_duration", 0)]:
    if key not in st.session_state: st.session_state[key] = val

if st.session_state.vector_store is None:
    raw_data, dyn_prompt = get_and_process_data()
    if raw_data:
        st.session_state.vector_store = create_vector_store(raw_data)
        st.session_state.dynamic_sys_prompt = dyn_prompt

# --- 5. RENDER UI ---

# BAGIAN 1: HEADER (FIXED)
st.markdown('<div class="fixed-header">', unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; margin-bottom: 5px;'>🎓 Sivita Poltesa</h2>", unsafe_allow_html=True)
with st.expander("⚙️ Konfigurasi", expanded=False):
    email = st.text_input("Email Gmail:", placeholder="nama@gmail.com")
    if st.button("🔄 Sinkronkan Data", use_container_width=True):
        st.cache_data.clear()
        st.session_state.vector_store = None
        st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

# BAGIAN 2: AREA TENGAH (SCROLLABLE)
st.markdown('<div class="scrollable-container">', unsafe_allow_html=True)
if st.session_state["last_answer"]:
    st.markdown("### 🤖 Jawaban")
    st.markdown(f'<div class="answer-box">{st.session_state["last_answer"]}</div>', unsafe_allow_html=True)
    st.caption(f"⏱️ Respon: {st.session_state['last_duration']} detik")
    st.button("Hapus Jawaban ✨", on_click=clear_answer_only)
else:
    st.info("Halo! Ada yang bisa saya bantu hari ini?")
st.markdown('</div>', unsafe_allow_html=True)

# BAGIAN 3: FOOTER (FIXED)
st.markdown('<div class="fixed-footer">', unsafe_allow_html=True)
user_query = st.text_area("Input", placeholder="Ketik pertanyaan...", key="user_query_input", height=80, label_visibility="collapsed")

col_send, col_clear = st.columns([2, 1])
with col_send:
    btn_kirim = st.button("Kirim 🚀", use_container_width=True, type="primary")
with col_clear:
    st.button("Bersihkan 🗑️", on_click=clear_input_only, use_container_width=True)

st.markdown("<div style='text-align: center; color: #999; font-size: 0.75rem; margin-top: 10px;'>Sivita Virtual Assistant Poltesa @2026</div>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- 6. LOGIKA ---
if btn_kirim:
    if not is_valid_email(email):
        st.error("Gunakan email @gmail.com")
    elif not user_query:
        st.warning("Tulis pertanyaan.")
    else:
        with st.spinner("Berpikir..."):
            start_time = time.time()
            try:
                context = "\n".join(semantic_search(user_query, st.session_state.vector_store))
                llm = ChatOpenAI(model="google/gemini-2.0-flash-lite-001", openai_api_key=st.secrets["OPENROUTER_API_KEY"], openai_api_base="https://openrouter.ai/api/v1", temperature=0.1)
                res = llm.invoke(f"{st.session_state.dynamic_sys_prompt}\n\nDATA:\n{context}\n\nPERTANYAAN: {user_query}")
                st.session_state["last_answer"], st.session_state["last_duration"] = res.content, round(time.time() - start_time, 2)
                save_to_log(email, user_query, res.content, st.session_state["last_duration"])
                st.rerun()
            except Exception as e: st.error(f"Error: {e}")
