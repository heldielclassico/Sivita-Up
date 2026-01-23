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

# --- KODE CSS UNTUK STRUKTUR 3 BAGIAN (HEADER TETAP, TENGAH SCROLL, FOOTER TETAP) ---
st.markdown(f"""
    <style>
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    .stAppDeployButton {{display: none;}}

    /* Reset margin aplikasi agar full ke layar */
    .main .block-container {{
        max-width: 800px;
        padding-top: 0rem;
        padding-bottom: 0rem;
    }}

    /* --- BAGIAN 1: HEADER (TETAP DI ATAS) --- */
    .fixed-header {{
        position: fixed;
        top: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 100%;
        max-width: 800px;
        background-color: white;
        z-index: 1000;
        padding: 20px 20px 10px 20px;
        border-bottom: 2px solid #f1f1f1;
    }}

    /* --- BAGIAN 2: AREA JAWABAN (DAPAT DI-SCROLL) --- */
    .scrollable-content {{
        margin-top: 200px; /* Menghindari tumpang tindih dengan header */
        margin-bottom: 300px; /* Menghindari tumpang tindih dengan footer */
        padding: 10px;
    }}
    
    .answer-box {{
        padding: 20px;
        background-color: #f8f9fa;
        border-radius: 12px;
        border: 1px solid #e0e0e0;
        line-height: 1.6;
        color: #31333F;
    }}

    /* --- BAGIAN 3: AREA INPUT (TETAP DI BAWAH) --- */
    .fixed-footer {{
        position: fixed;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 100%;
        max-width: 800px;
        background-color: white;
        z-index: 1000;
        padding: 15px 20px 10px 20px;
        border-top: 2px solid #f1f1f1;
        box-shadow: 0 -5px 15px rgba(0,0,0,0.05);
    }}

    /* Spasi antar kolom */
    [data-testid="stHorizontalBlock"] {{
        gap: 10px !important;
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
        st.error(f"Gagal memuat Database: {e}")
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
    results = [vector_store["chunks"][idx]["text"] for idx in indices[0] if idx < len(vector_store["chunks"])]
    return results

def save_to_log(email, question, answer="", duration=0):
    try:
        log_url = st.secrets["LOG_URL"]
        payload = {"email": email, "question": question, "answer": answer, "duration": f"{duration} detik"}
        requests.post(log_url, json=payload, timeout=5)
    except Exception: pass

# --- 4. INISIALISASI ---

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "dynamic_sys_prompt" not in st.session_state:
    st.session_state.dynamic_sys_prompt = ""
if "last_answer" not in st.session_state:
    st.session_state["last_answer"] = ""
if "last_duration" not in st.session_state:
    st.session_state["last_duration"] = 0

if st.session_state.vector_store is None:
    with st.spinner("Mensinkronkan Data..."):
        raw_data, dyn_prompt = get_and_process_data()
        if raw_data:
            st.session_state.vector_store = create_vector_store(raw_data)
            st.session_state.dynamic_sys_prompt = dyn_prompt

# --- 5. RENDER UI ---

# BAGIAN 1: HEADER (JUDUL & KONFIGURASI) - FIXED TOP
st.markdown('<div class="fixed-header">', unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; margin-bottom: 10px;'>🎓 Sivita Poltesa</h1>", unsafe_allow_html=True)
with st.expander("⚙️ Konfigurasi Email & Sinkronisasi", expanded=False):
    email = st.text_input("Email Gmail Anda:", placeholder="nama@gmail.com")
    if st.button("🔄 Sinkronkan Ulang Data", use_container_width=True):
        st.cache_data.clear()
        st.session_state.vector_store = None
        st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

# BAGIAN 2: JAWABAN - SCROLLABLE CONTENT
st.markdown('<div class="scrollable-content">', unsafe_allow_html=True)
if st.session_state["last_answer"]:
    st.markdown("### 🤖 Jawaban Sivita")
    ans_html = f'<div class="answer-box">{st.session_state["last_answer"]}</div>'
    st.markdown(ans_html, unsafe_allow_html=True)
    st.caption(f"⏱️ Waktu Respon: {st.session_state['last_duration']} detik")
    st.button("Hapus Jawaban ✨", on_click=clear_answer_only)
else:
    st.info("Halo! Saya Sivita. Ada yang bisa saya bantu hari ini?")
st.markdown('</div>', unsafe_allow_html=True)

# BAGIAN 3: INPUT & FOOTER - FIXED BOTTOM
st.markdown('<div class="fixed-footer">', unsafe_allow_html=True)
user_query = st.text_area(
    "Label Tersembunyi", 
    placeholder="Ketik pertanyaan Anda di sini...", 
    key="user_query_input", 
    height=100,
    label_visibility="collapsed"
)

col_send, col_clear = st.columns([1.5, 1])
with col_send:
    btn_kirim = st.button("Kirim Pertanyaan 🚀", use_container_width=True, type="primary")
with col_clear:
    st.button("Hapus Pertanyaan 🗑️", on_click=clear_input_only, use_container_width=True)

st.divider()
st.markdown("<div style='text-align: center; color: #888; font-size: 0.85rem; padding-bottom: 10px;'>Sivita Virtual Assistant Poltesa @2026</div>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- 6. LOGIKA PENGIRIMAN ---
if btn_kirim:
    if not is_valid_email(email):
        st.error("Harap masukkan email @gmail.com yang valid.")
    elif not user_query:
        st.warning("Pertanyaan tidak boleh kosong.")
    else:
        with st.spinner("Sedang memproses..."):
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
                full_prompt = f"{st.session_state.dynamic_sys_prompt}\n\nDATA REFERENSI:\n{context_text}\n\nPERTANYAAN: {user_query}"
                response = llm.invoke(full_prompt)
                st.session_state["last_answer"] = response.content
                st.session_state["last_duration"] = round(time.time() - start_time, 2)
                save_to_log(email, user_query, response.content, st.session_state["last_duration"])
                st.rerun()
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
