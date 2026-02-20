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

# 1. Load Environment Variables (Opsional)
# from dotenv import load_dotenv
# load_dotenv()

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
        max-height: 400px;
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

def clear_history():
    st.session_state["chat_history"] = []
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

# 1. Load Environment Variables (Opsional)
# from dotenv import load_dotenv
# load_dotenv()

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
        max-height: 400px;
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

def clear_history():
    st.session_state["chat_history"] = []
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
                
              # Di dalam loop fungsi get_and_process_data
                    for idx, row in df.iterrows():
              # Tambahkan nama tab di awal string agar AI tahu konteksnya
                    row_content = f"DATABASE {tab.upper()}: " + ", ".join([f"{col} adalah {val}" for col, val in row.items() if pd.notna(val)])
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
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# --- 5. LOGIKA SINKRONISASI ---

if st.session_state.vector_store is None:
    loading_screen = st.empty()
    with loading_screen.container():
        st.markdown("<br><br><br><br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if lottie_searching:
                st_lottie(lottie_searching, height=280, key="initial_sync_anim")
            st.markdown("<h3 style='text-align: center; color: #007bff;'>Mensinkronkan Data...</h3>", unsafe_allow_html=True)
        
        raw_data, dyn_prompt = get_and_process_data()
        if raw_data:
            st.session_state.vector_store = create_vector_store(raw_data)
            st.session_state.dynamic_sys_prompt = dyn_prompt
            time.sleep(1.5)
            st.rerun()
    st.stop()

# --- 6. UI UTAMA ---

st.markdown("<h1 style='text-align: center; margin-top: -40px; margin-bottom: -23px;'>🎓 Asisten Virtual Poltesa (Sivita)</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray; margin-bottom: 15px;'>Sivita v1.4 | Memory Integrated</p>", unsafe_allow_html=True)

with st.container(border=True):
    email = st.text_input("Email Gmail Anda:", placeholder="nama@gmail.com")
    
    col_sync, col_reset = st.columns(2)
    with col_sync:
        if st.button("🔄 Sinkron Data", use_container_width=True):
            st.cache_data.clear()
            st.session_state.vector_store = None
            st.rerun()
    with col_reset:
        if st.button("🧹 Hapus Chat", on_click=clear_history, use_container_width=True):
            pass

    placeholder_animasi = st.empty()

    # Tampilan Jawaban
    if st.session_state["last_answer"]:
        with st.container():
            st.markdown(f'<div class="answer-box"><strong>🤖 Sivita:</strong><br>{st.session_state["last_answer"]}</div>', unsafe_allow_html=True)
            st.caption(f"⏱️ {st.session_state['last_duration']} detik")

    # Input Area
    with st.container(border=True):
        user_query = st.text_area("Pertanyaan Anda:", placeholder="Tanyakan apa saja...", key="user_query_input", height=100)
        col_send, col_del = st.columns([2, 1])
        with col_send:
            btn_kirim = st.button("Kirim 🚀", use_container_width=True, type="primary")
        with col_del:
            st.button("Hapus 🗑️", on_click=clear_input_only, use_container_width=True)

    if btn_kirim:
        if not is_valid_email(email):
            st.error("Gunakan email @gmail.com")
        elif not user_query:
            st.warning("Tuliskan pertanyaan.")
        else:
            with placeholder_animasi.container():
                if lottie_searching: st_lottie(lottie_searching, height=150)
            
            start_time = time.time()
            try:
                # 1. Search Context
                context_list = semantic_search(user_query, st.session_state.vector_store)
                context_text = "\n".join(context_list)
                
                # 2. Build History Text
                history_text = "\n".join([f"User: {c['u']}\nAI: {c['b']}" for c in st.session_state.chat_history[-3:]])

                # 3. Setup LLM & Prompt
                llm = ChatOpenAI(
                    model="google/gemini-2.0-flash-lite-001",
                    openai_api_key=st.secrets["OPENROUTER_API_KEY"],
                    openai_api_base="https://openrouter.ai/api/v1",
                    temperature=0.2
                )
                
                # Sesuai Memori 2026-01-22: Jangan tampilkan pesan Google Sheets
                sys_instruction = (
                    f"{st.session_state.dynamic_sys_prompt}\n"
                    "INSTRUKSI PENTING: Jangan pernah menyarankan user melihat Google Sheets untuk informasi sosial media. "
                    "Jawablah langsung berdasarkan data yang tersedia."
                )

                full_prompt = (
                    f"Sistem: {sys_instruction}\n\n"
                    f"Konteks Data:\n{context_text}\n\n"
                    f"Riwayat Chat:\n{history_text}\n\n"
                    f"Pertanyaan User: {user_query}"
                )
                
                response = llm.invoke(full_prompt)
                ans = response.content
                
                # 4. Save State
                st.session_state.chat_history.append({"u": user_query, "b": ans})
                st.session_state["last_answer"] = ans
                st.session_state["last_duration"] = round(time.time() - start_time, 2)
                
                save_to_log(email, user_query, ans, st.session_state["last_duration"])
                placeholder_animasi.empty()
                st.rerun()
                
            except Exception as e:
                placeholder_animasi.empty()
                st.error(f"Error: {e}")

st.caption("Sivita - Virtual Assistant Poltesa @2026")

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
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# --- 5. LOGIKA SINKRONISASI ---

if st.session_state.vector_store is None:
    loading_screen = st.empty()
    with loading_screen.container():
        st.markdown("<br><br><br><br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if lottie_searching:
                st_lottie(lottie_searching, height=280, key="initial_sync_anim")
            st.markdown("<h3 style='text-align: center; color: #007bff;'>Mensinkronkan Data...</h3>", unsafe_allow_html=True)
        
        raw_data, dyn_prompt = get_and_process_data()
        if raw_data:
            st.session_state.vector_store = create_vector_store(raw_data)
            st.session_state.dynamic_sys_prompt = dyn_prompt
            time.sleep(1.5)
            st.rerun()
    st.stop()

# --- 6. UI UTAMA ---

st.markdown("<h1 style='text-align: center; margin-top: -40px; margin-bottom: -23px;'>🎓 Asisten Virtual Poltesa (Sivita)</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray; margin-bottom: 15px;'>Sivita v1.4 | Memory Integrated</p>", unsafe_allow_html=True)

with st.container(border=True):
    email = st.text_input("Email Gmail Anda:", placeholder="nama@gmail.com")
    
    col_sync, col_reset = st.columns(2)
    with col_sync:
        if st.button("🔄 Sinkron Data", use_container_width=True):
            st.cache_data.clear()
            st.session_state.vector_store = None
            st.rerun()
    with col_reset:
        if st.button("🧹 Hapus Chat", on_click=clear_history, use_container_width=True):
            pass

    placeholder_animasi = st.empty()

    # Tampilan Jawaban
    if st.session_state["last_answer"]:
        with st.container():
            st.markdown(f'<div class="answer-box"><strong>🤖 Sivita:</strong><br>{st.session_state["last_answer"]}</div>', unsafe_allow_html=True)
            st.caption(f"⏱️ {st.session_state['last_duration']} detik")

    # Input Area
    with st.container(border=True):
        user_query = st.text_area("Pertanyaan Anda:", placeholder="Tanyakan apa saja...", key="user_query_input", height=100)
        col_send, col_del = st.columns([2, 1])
        with col_send:
            btn_kirim = st.button("Kirim 🚀", use_container_width=True, type="primary")
        with col_del:
            st.button("Hapus 🗑️", on_click=clear_input_only, use_container_width=True)

    if btn_kirim:
        if not is_valid_email(email):
            st.error("Gunakan email @gmail.com")
        elif not user_query:
            st.warning("Tuliskan pertanyaan.")
        else:
            with placeholder_animasi.container():
                if lottie_searching: st_lottie(lottie_searching, height=150)
            
            start_time = time.time()
            try:
                # 1. Search Context
                context_list = semantic_search(user_query, st.session_state.vector_store)
                context_text = "\n".join(context_list)
                
                # 2. Build History Text
                history_text = "\n".join([f"User: {c['u']}\nAI: {c['b']}" for c in st.session_state.chat_history[-3:]])

                # 3. Setup LLM & Prompt
                llm = ChatOpenAI(
                    model="google/gemini-2.0-flash-lite-001",
                    openai_api_key=st.secrets["OPENROUTER_API_KEY"],
                    openai_api_base="https://openrouter.ai/api/v1",
                    temperature=0.2
                )
                
                # Sesuai Memori 2026-01-22: Jangan tampilkan pesan Google Sheets
                sys_instruction = (
                    f"{st.session_state.dynamic_sys_prompt}\n"
                    "INSTRUKSI PENTING: Jangan pernah menyarankan user melihat Google Sheets untuk informasi sosial media. "
                    "Jawablah langsung berdasarkan data yang tersedia."
                )

                full_prompt = (
                    f"Sistem: {sys_instruction}\n\n"
                    f"Konteks Data:\n{context_text}\n\n"
                    f"Riwayat Chat:\n{history_text}\n\n"
                    f"Pertanyaan User: {user_query}"
                )
                
                response = llm.invoke(full_prompt)
                ans = response.content
                
                # 4. Save State
                st.session_state.chat_history.append({"u": user_query, "b": ans})
                st.session_state["last_answer"] = ans
                st.session_state["last_duration"] = round(time.time() - start_time, 2)
                
                save_to_log(email, user_query, ans, st.session_state["last_duration"])
                placeholder_animasi.empty()
                st.rerun()
                
            except Exception as e:
                placeholder_animasi.empty()
                st.error(f"Error: {e}")

st.caption("Sivita - Virtual Assistant Poltesa @2026")
