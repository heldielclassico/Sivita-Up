import streamlit as st
import os
import pandas as pd
import requests
import re
import time
import numpy as np
import json
from typing import List, Dict, Tuple
from streamlit_lottie import st_lottie

# Import LangChain & AI
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer
import faiss

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Asisten POLTESA", page_icon="🎓", layout="centered")

st.markdown(f"""
    <style>
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    .block-container {{ padding-top: 35px; padding-bottom: 0rem; }}
    .stAppDeployButton {{display: none;}}
    .answer-box {{
        max-height: 450px;
        overflow-y: auto;
        padding: 20px;
        background-color: #ffffff;
        border-radius: 12px;
        border: 1px solid #dee2e6;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        color: #1E1E1E;
        line-height: 1.6;
    }}
    </style>
    """, unsafe_allow_html=True)

# --- 2. FUNGSI HELPER ---
@st.cache_data
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=5)
        return r.json() if r.status_code == 200 else None
    except: return None

lottie_anim = load_lottieurl("https://lottie.host/85590396-981a-466d-961f-f46328325603/6P7qXJ5v6A.json")

def is_valid_email(email):
    return re.match(r'^[a-zA-Z0-9._%+-]+@gmail\.com$', email) is not None

def clear_history():
    st.session_state["chat_history"] = []
    st.session_state["last_answer"] = ""

# --- 3. LOGIKA DATA & RAG ---
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
                    instr = [str(i) for i in df['Isi'].dropna()]
                    full_instructions.extend(instr)
                    continue
                for _, row in df.iterrows():
                    content = f"TAB {tab.upper()}: " + " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                    all_chunks.append({"text": content, "source": tab})
            except: continue
        return all_chunks, "\n".join(full_instructions)
    except: return [], ""

def create_vector_store(chunks_data: List[Dict]):
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    texts = [c["text"] for c in chunks_data]
    embeddings = model.encode(texts, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype('float32'))
    return {"index": index, "chunks": chunks_data, "model": model}

def semantic_search(query: str, vector_store: Dict, top_k: int = 15):
    # top_k dinaikkan ke 15 agar aturan 'JadwalKu' tidak tenggelam oleh data Jadwal Dosen
    query_vec = vector_store["model"].encode([query], normalize_embeddings=True)
    _, indices = vector_store["index"].search(query_vec.astype('float32'), top_k)
    return [vector_store["chunks"][idx]["text"] for idx in indices[0] if idx < len(vector_store["chunks"])]

# --- 4. INISIALISASI STATE ---
if "vector_store" not in st.session_state:
    raw_data, dyn_prompt = get_and_process_data()
    if raw_data:
        st.session_state.vector_store = create_vector_store(raw_data)
        st.session_state.dynamic_sys_prompt = dyn_prompt
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "last_answer" not in st.session_state: st.session_state.last_answer = ""

# --- 5. UI UTAMA ---
st.markdown("<h1 style='text-align: center;'>🎓 Sivita AI</h1>", unsafe_allow_html=True)

with st.container(border=True):
    email = st.text_input("Email Gmail Anda:", placeholder="nama@gmail.com")
    if st.button("🔄 Sinkron Ulang Data"):
        st.cache_data.clear()
        st.session_state.clear()
        st.rerun()

    if st.session_state.last_answer:
        st.markdown(f'<div class="answer-box"><strong>🤖 Sivita:</strong><br>{st.session_state.last_answer}</div>', unsafe_allow_html=True)

    user_query = st.text_area("Masukkan pertanyaan atau kata kunci (Contoh: JadwalKu):", height=100)
    btn_kirim = st.button("Kirim 🚀", type="primary", use_container_width=True)

    if btn_kirim:
        if not is_valid_email(email):
            st.error("Email tidak valid!")
        elif not user_query:
            st.warning("Pertanyaan kosong.")
        else:
            with st.spinner("Sivita sedang berpikir..."):
                try:
                    # 1. Ambil konteks (Top_K 15 agar aturan sistem ikut terbawa)
                    context_list = semantic_search(user_query, st.session_state.vector_store, top_k=15)
                    context_text = "\n".join(context_list)
                    
                    # 2. Tambahkan Logika Penegas agar JadwalKu tidak tertukar dengan Jadwal Dosen
                    # Kami masukkan aturan ini ke dalam prompt sistem secara dinamis
                    sys_rules = (
                        f"{st.session_state.dynamic_sys_prompt}\n"
                        "INSTRUKSI KHUSUS:\n"
                        "- Jika user mengetik 'JadwalKu' (huruf besar/kecil tidak masalah), Anda WAJIB memberikan link Google Form untuk input jadwal.\n"
                        "- Jika user bertanya 'Jadwal [Nama Dosen]', cari datanya di TAB JADWALDOSEN.\n"
                        "- JANGAN memberikan link form input jika user hanya menanyakan jadwal dosen tertentu.\n"
                        "- Jangan sebutkan 'Google Sheets' sesuai aturan Sivita."
                    )

                    full_prompt = (
                        f"{sys_rules}\n\n"
                        f"DATA REFERENSI:\n{context_text}\n\n"
                        f"PERTANYAAN: {user_query}"
                    )
                    
                    llm = ChatOpenAI(
                        model="google/gemini-2.0-flash-lite-001",
                        openai_api_key=st.secrets["OPENROUTER_API_KEY"],
                        openai_api_base="https://openrouter.ai/api/v1",
                        temperature=0.0 # Temperature 0 agar AI sangat patuh dan tidak berimprovisasi
                    )
                    
                    response = llm.invoke(full_prompt)
                    st.session_state.last_answer = response.content
                    st.session_state.chat_history.append({"u": user_query, "b": response.content})
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
