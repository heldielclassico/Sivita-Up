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

# --- KODE CSS AGRESIF UNTUK MENGHAPUS ELEMEN PUTIH ---
st.markdown(f"""
    <style>
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    
    .block-container {{
        padding-top: 5px;
        padding-bottom: 220px; 
    }}

    /* Container utama panel melayang */
    div[data-testid="stVerticalBlock"] > div:has(div.floating-anchor) {{
        position: fixed;
        bottom: 50px;
        left: 50%;
        transform: translateX(-50%);
        width: 95%;
        max-width: 730px; 
        background-color: #ffffff;
        padding: 10px 12px;
        border: 1px solid #e0e0e0;
        border-radius: 20px;
        z-index: 999;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        overflow: visible !important;
    }}

    /* HAPUS ELEMEN PUTIH LONJONG (Sangat Agresif) */
    div[data-testid="stFormSubmitButton"], 
    div[data-testid="stWidgetLabel"],
    .stTextArea label,
    div[data-baseweb="base-input"] + div {{
        display: none !important;
        height: 0px !important;
        margin: 0px !important;
        padding: 0px !important;
    }}

    /* Menghilangkan border dan shadow default textarea */
    .stTextArea textarea {{
        border: none !important;
        background-color: transparent !important;
        padding-right: 110px !important; 
        resize: none !important;
        font-size: 16px !important;
        min-height: 80px !important;
        box-shadow: none !important;
    }}

    /* Tombol melayang di pojok kanan bawah */
    div[data-testid="column"]:has(button) {{
        position: absolute !important;
        right: 15px !important;
        bottom: 15px !important;
        z-index: 1001 !important;
        width: auto !important;
    }}
    
    [data-testid="stHorizontalBlock"] {{
        display: flex !important;
        gap: 6px !important; 
        flex-direction: row !important;
    }}

    /* Style tombol bulat */
    .stButton > button {{
        border-radius: 50px !important;
        padding: 0px 8px !important;
        height: 38px !important;
        min-width: 45px !important;
        border: 1px solid #f0f0f0 !important;
        background-color: white !important;
    }}

    button[kind="primary"] {{
        background-color: #ff4b4b !important;
        color: white !important;
        border: none !important;
    }}

    .stAppDeployButton {{display: none;}}
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
                    if 'Isi' in df.columns:
                        full_instructions = df['Isi'].dropna().astype(str).tolist()
                    continue
                for idx, row in df.iterrows():
                    row_content = f"Data {tab}: " + ", ".join([f"{col} adalah {val}" for col, val in row.items() if pd.notna(val)])
                    all_chunks.append({"text": row_content, "source": tab})
            except Exception: continue
        final_prompt = "\n".join(full_instructions) if full_instructions else "Anda adalah Sivita."
        return all_chunks, final_prompt
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

def semantic_search(query: str, vector_store: Dict, top_k: int = 5):
    query_vec = vector_store["model"].encode([query], normalize_embeddings=True)
    distances, indices = vector_store["index"].search(query_vec.astype('float32'), top_k)
    return [vector_store["chunks"][idx]["text"] for idx in indices[0] if idx < len(vector_store["chunks"])]

def save_to_log(email, question, answer="", duration=0):
    try:
        log_url = st.secrets["LOG_URL"]
        payload = {"email": email, "question": question, "answer": answer, "duration": f"{duration} detik"}
        requests.post(log_url, json=payload, timeout=5)
    except Exception: pass

# --- 4. INISIALISASI ---

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
    with st.spinner("Mensinkronkan Data..."):
        raw_data, dyn_prompt = get_and_process_data()
        if raw_data:
            st.session_state.vector_store = create_vector_store(raw_data)
            st.session_state.dynamic_sys_prompt = dyn_prompt

if "last_answer" not in st.session_state: st.session_state["last_answer"] = ""
if "last_duration" not in st.session_state: st.session_state["last_duration"] = 0

# --- 5. UI UTAMA ---

st.markdown("<h1 style='text-align: center; margin-top: -40px; margin-bottom: -15px;'>🎓 Asisten Virtual Poltesa (Sivita)</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray; margin-top: 0px; margin-bottom: 15px;'>Sivita v1.3 | Fixed UI</p>", unsafe_allow_html=True)

email = st.text_input("Email Gmail Anda:", placeholder="nama@gmail.com")
if st.button("🔄 Sinkronkan Ulang Data", use_container_width=True):
    st.cache_data.clear()
    st.session_state.vector_store = None
    st.rerun()

if st.session_state["last_answer"]:
    st.markdown("---")
    with st.chat_message("assistant"):
        st.markdown(st.session_state["last_answer"])
    col_info, col_clear = st.columns([2, 1])
    with col_info: st.caption(f"⏱️ {st.session_state['last_duration']} detik")
    with col_clear: st.button("Hapus Jawaban ✨", on_click=clear_answer_only, use_container_width=True)
    st.markdown("---")

# --- PANEL INPUT (Hapus paksa elemen pengganggu) ---
with st.container():
    st.markdown('<div class="floating-anchor"></div>', unsafe_allow_html=True)
    
    # Text area tanpa label
    user_query = st.text_area(
        "hidden_label", 
        placeholder="Tanyakan sesuatu pada Sivita...", 
        key="user_query_input", 
        label_visibility="collapsed"
    )
    
    # Tombol Berdekatan
    c1, c2 = st.columns([1, 1])
    with c1:
        st.button("🗑️", on_click=clear_input_only)
    with c2:
        btn_kirim = st.button("🚀", type="primary")

    if btn_kirim:
        if not is_valid_email(email):
            st.error("Gunakan email @gmail.com")
        elif user_query:
            with st.spinner("..."):
                start_time = time.time()
                try:
                    context_list = semantic_search(user_query, st.session_state.vector_store)
                    llm = ChatOpenAI(
                        model="google/gemini-2.0-flash-lite-001",
                        openai_api_key=st.secrets["OPENROUTER_API_KEY"],
                        openai_api_base="https://openrouter.ai/api/v1",
                        temperature=0.1
                    )
                    full_p = f"{st.session_state.dynamic_sys_prompt}\n\nDATA:\n{chr(10).join(context_list)}\n\nQ: {user_query}"
                    response = llm.invoke(full_p)
                    st.session_state["last_answer"] = response.content
                    st.session_state["last_duration"] = round(time.time() - start_time, 2)
                    save_to_log(email, user_query, response.content, st.session_state["last_duration"])
                    st.rerun()
                except Exception as e: st.error(f"Error: {e}")

st.caption("Sivita Poltesa @2026")

ganti tampilan text area nya yang lebih modern
