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

# --- KODE CSS UNTUK TAMPILAN MENGAMBANG PERMANEN (FIXED BOTTOM) ---
st.markdown(f"""
    <style>
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    
    /* Memberikan ruang kosong di bawah agar konten jawaban tidak tertutup input floating */
    .main .block-container {{
        padding-bottom: 280px;
        padding-top: 10px;
    }}
    
    [data-testid="stHorizontalBlock"] {{
        gap: 5px !important;
    }}
    
    .stAppDeployButton {{display: none;}}

    /* --- STYLE AREA JAWABAN SCROLLABLE --- */
    .answer-box {{
        max-height: 400px;
        overflow-y: auto;
        padding: 15px;
        background-color: #f8f9fa;
        border-radius: 12px;
        border: 1px solid #e0e0e0;
        margin-bottom: 15px;
        line-height: 1.6;
        color: #31333F;
    }}

    /* --- STYLE AREA INPUT MENGAMBANG (FIXED AT BOTTOM) --- */
    .custom-input-group {{
        position: fixed;
        bottom: 25px;
        left: 50%;
        transform: translateX(-50%);
        width: 100%;
        max-width: 730px; /* Menyesuaikan lebar layout centered Streamlit */
        z-index: 9999;
        padding: 18px;
        background-color: #ffffff;
        border: 1px solid #d1d5db;
        border-radius: 20px;
        box-shadow: 0 -10px 25px rgba(0,0,0,0.1);
    }}
    
    /* Responsif untuk perangkat mobile */
    @media (max-width: 768px) {{
        .custom-input-group {{
            width: 92%;
            bottom: 15px;
        }}
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
            except Exception:
                continue
        
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
    except Exception as e:
        st.error(f"Gagal membangun Vector DB: {e}")
        return None

def semantic_search(query: str, vector_store: Dict, top_k: int = 5):
    query_vec = vector_store["model"].encode([query], normalize_embeddings=True)
    distances, indices = vector_store["index"].search(query_vec.astype('float32'), top_k)
    results = [vector_store["chunks"][idx]["text"] for idx in indices[0] if idx < len(vector_store["chunks"])]
    return results

def save_to_log(email, question, answer="", duration=0):
    try:
        log_url = st.secrets["LOG_URL"]
        payload = {
            "email": email,
            "question": question,
            "answer": answer,
            "duration": f"{duration} detik"
        }
        requests.post(log_url, json=payload, timeout=5)
    except Exception as e:
        print(f"Log Error: {e}")

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

# --- 5. UI UTAMA ---

st.markdown("<h1 style='text-align: center; margin-top: -30px;'>🎓 Sivita Poltesa</h1>", unsafe_allow_html=True)

with st.expander("⚙️ Konfigurasi Email & Data", expanded=False):
    email = st.text_input("Email Gmail:", placeholder="nama@gmail.com")
    if st.button("🔄 Sinkronkan Ulang", use_container_width=True):
        st.cache_data.clear()
        st.session_state.vector_store = None
        st.rerun()

# AREA TAMPILAN JAWABAN
if st.session_state["last_answer"]:
    st.markdown("### 🤖 Jawaban")
    # Menggunakan string tunggal untuk mencegah kebocoran tag HTML
    ans_html = f'<div class="answer-box">{st.session_state["last_answer"]}</div>'
    st.markdown(ans_html, unsafe_allow_html=True)
    st.caption(f"⏱️ Selesai dalam {st.session_state['last_duration']} detik")
    st.button("Hapus Jawaban ✨", on_click=clear_answer_only)

# --- WRAPPER INPUT FLOATING (STAY AT BOTTOM) ---
st.markdown('<div class="custom-input-group">', unsafe_allow_html=True)

user_query = st.text_area(
    "Tanya Sivita:", 
    placeholder="Ketik pertanyaan Anda di sini...", 
    key="user_query_input", 
    height=90, 
    label_visibility="collapsed"
)

col_send, col_clear = st.columns([2, 1])
with col_send:
    btn_kirim = st.button("Kirim 🚀", use_container_width=True, type="primary")
with col_clear:
    st.button("Bersihkan 🗑️", on_click=clear_input_only, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# --- LOGIKA BACKEND ---
if btn_kirim:
    if not is_valid_email(email):
        st.error("Gunakan email @gmail.com")
    elif not user_query:
        st.warning("Silakan tulis pertanyaan.")
    else:
        with st.spinner("Sivita sedang berpikir..."):
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
                full_prompt = f"{st.session_state.dynamic_sys_prompt}\n\nDATA:\n{context_text}\n\nPERTANYAAN: {user_query}"
                response = llm.invoke(full_prompt)
                
                st.session_state["last_answer"] = response.content
                st.session_state["last_duration"] = round(time.time() - start_time, 2)
                save_to_log(email, user_query, response.content, st.session_state["last_duration"])
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

st.divider()
st.caption("Sivita Virtual Assistant @2026")
