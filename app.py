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

# 1. Load Environment Variables (Disesuaikan dengan st.secrets)

# 2. Konfigurasi Halaman
st.set_page_config(page_title="Asisten POLTESA", page_icon="🎓", layout="centered")

# --- KODE UNTUK MENGHILANGKAN MENU, FOOTER, DAN ICON GITHUB ---
st.markdown(f"""
    <style>
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    
    .block-container {{
        padding-top: 35px;
        padding-bottom: 0rem;
    }}
    
    [data-testid="stHorizontalBlock"] {{
        gap: 5px !important;
    }}
    
    .stAppDeployButton {{display: none;}}

    .answer-box {{
        max-height: 350px;
        overflow-y: auto;
        padding: 15px;
        background-color: #f8f9fa;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin-bottom: 10px;
        line-height: 1.6;
        color: #31333F;
    }}
    </style>
    """, unsafe_allow_html=True)

# --- FUNGSI ANIMASI LOTTIE ---
@st.cache_data
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

lottie_searching = load_lottieurl("https://lottie.host/85590396-981a-466d-961f-f46328325603/6P7qXJ5v6A.json")

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

# --- 4. INISIALISASI STATE ---

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "dynamic_sys_prompt" not in st.session_state:
    st.session_state.dynamic_sys_prompt = ""
if "last_answer" not in st.session_state:
    st.session_state["last_answer"] = ""
if "last_duration" not in st.session_state:
    st.session_state["last_duration"] = 0

# --- 5. LOGIKA SINKRONISASI (FULL SCREEN LOADING) ---

if st.session_state.vector_store is None:
    loading_screen = st.empty()
    with loading_screen.container():
        st.markdown("<br><br><br><br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if lottie_searching:
                st_lottie(lottie_searching, height=280, key="initial_sync_anim")
            st.markdown("<h3 style='text-align: center; color: #007bff;'>Mensinkronkan Data...</h3>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; color: gray;'>Sivita sedang menyiapkan informasi terbaru untuk Anda.</p>", unsafe_allow_html=True)
        
        raw_data, dyn_prompt = get_and_process_data()
        if raw_data:
            st.session_state.vector_store = create_vector_store(raw_data)
            st.session_state.dynamic_sys_prompt = dyn_prompt
            time.sleep(1.5)
            st.rerun()
    st.stop()

# --- 6. UI UTAMA ---

st.markdown("<h1 style='text-align: center; margin-top: -40px; margin-bottom: -23px;'>🎓 Asisten Virtual Poltesa (Sivita)</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray; margin-bottom: 15px;'>Sivita v1.3 | Modular Prompt System</p>", unsafe_allow_html=True)

with st.container(border=True):
    email = st.text_input("Email Gmail Anda:", placeholder="nama@gmail.com")
    
    if st.button("🔄 Sinkronkan Ulang Data", use_container_width=True):
        st.cache_data.clear()
        st.session_state.vector_store = None
        st.rerun()

    # Tampilan Jawaban Terakhir
    if st.session_state["last_answer"]:
        full_answer_html = (
            f'<div class="answer-box" style="margin-top: 15px;">'
            f'<div style="font-weight: bold; color: #007bff; margin-bottom: 8px;">🤖 Jawaban Sivita:</div>'
            f'{st.session_state["last_answer"]}'
            f'</div>'
        )
        st.markdown(full_answer_html, unsafe_allow_html=True)
        st.caption(f"⏱️ Selesai dalam {st.session_state['last_duration']} detik")
        st.button("Hapus Jawaban ✨", on_click=clear_answer_only, use_container_width=True)

    # Area Input Pertanyaan
    with st.container(border=True):
        user_query = st.text_area("Apa yang ingin Anda tanyakan?", placeholder="Tanyakan info kampus...", key="user_query_input", height=150)
        col_send, col_del_q = st.columns([1.5, 1])
        with col_send:
            btn_kirim = st.button("Kirim Pertanyaan 🚀", use_container_width=True, type="primary")
        with col_del_q:
            st.button("Hapus Pertanyaan 🗑️", on_click=clear_input_only, use_container_width=True)

    # Logika Pengiriman Pertanyaan dengan Modal Status
    if btn_kirim:
        if not is_valid_email(email):
            st.error("Gunakan email @gmail.com")
        elif not user_query:
            st.warning("Tuliskan pertanyaan.")
        else:
            # LANGKAH 1: Hapus jawaban lama dari layar segera
            st.session_state["last_answer"] = ""
            st.session_state["last_duration"] = 0
            
            # LANGKAH 2: Tampilkan status/modal pencarian
            with st.status("🔍 Sivita sedang mencari data...", expanded=True) as status:
                st.write("Menganalisis pertanyaan...")
                if lottie_searching:
                    st_lottie(lottie_searching, height=150, key="query_loading")
                
                start_time = time.time()
                try:
                    # Proses Pencarian
                    context_list = semantic_search(user_query, st.session_state.vector_store)
                    context_text = "\n".join(context_list)
                    
                    status.write("Merumuskan jawaban terbaik...")
                    
                    llm = ChatOpenAI(
                        model="google/gemini-2.0-flash-lite-001",
                        openai_api_key=st.secrets["OPENROUTER_API_KEY"],
                        openai_api_base="https://openrouter.ai/api/v1",
                        temperature=0.1
                    )
                    
                    full_prompt = f"{st.session_state.dynamic_sys_prompt}\n\nREFERENSI DATA:\n{context_text}\n\nPERTANYAAN: {user_query}"
                    response = llm.invoke(full_prompt)
                    
                    # Simpan hasil ke state
                    st.session_state["last_answer"] = response.content
                    st.session_state["last_duration"] = round(time.time() - start_time, 2)
                    
                    # Simpan Log
                    save_to_log(email, user_query, response.content, st.session_state["last_duration"])
                    
                    status.update(label="✅ Selesai!", state="complete", expanded=False)
                    time.sleep(0.5)
                    st.rerun() # Refresh untuk memunculkan jawaban di area teks
                    
                except Exception as e:
                    status.update(label="❌ Terjadi kesalahan", state="error")
                    st.error(f"Kesalahan: {e}")

st.caption("Sivita - Virtual Assistant Poltesa @2026")
