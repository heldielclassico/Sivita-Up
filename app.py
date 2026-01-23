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

# --- KODE CSS MINIMALIS (HANYA UNTUK MEMBERSIHKAN UI) ---
st.markdown(f"""
    <style>
    /* Hilangkan elemen bawaan Streamlit */
    header[data-testid="stHeader"] {{ display: none !important; }}
    footer {{ visibility: hidden !important; }}
    #MainMenu {{ visibility: hidden !important; }}
    .stAppDeployButton {{ display: none !important; }}

    /* Atur margin agar rapi di tengah */
    .block-container {{
        max-width: 800px;
        padding-top: 2rem;
    }}
    
    /* Area teks jawaban agar tetap terlihat berbeda */
    .stAlert {{
        border-radius: 12px;
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

# Bagian Judul
st.title("🎓 Sivita Poltesa")

# Bagian Konfigurasi
with st.expander("⚙️ Konfigurasi", expanded=False):
    email = st.text_input("Email Gmail:", placeholder="nama@gmail.com")
    if st.button("🔄 Perbarui Data", use_container_width=True):
        st.cache_data.clear()
        st.session_state.vector_store = None
        st.rerun()

st.divider()

# Bagian Jawaban
if st.session_state["last_answer"]:
    st.subheader("🤖 Jawaban Sivita")
    st.info(st.session_state["last_answer"])
    st.caption(f"⏱️ Waktu proses: {st.session_state['last_duration']} detik")
    st.button("Hapus Jawaban ✨", on_click=clear_answer_only)
else:
    st.write("Silakan ajukan pertanyaan di bawah.")

st.divider()

# Bagian Input (Tanpa bungkusan DIV)
user_query = st.text_area(
    "Apa yang ingin Anda tanyakan?", 
    placeholder="Ketik pertanyaan di sini...", 
    key="user_query_input", 
    height=150
)

col_send, col_clear = st.columns([1.5, 1])
with col_send:
    btn_kirim = st.button("Kirim Pertanyaan 🚀", use_container_width=True, type="primary")
with col_clear:
    st.button("Bersihkan 🗑️", on_click=clear_input_only, use_container_width=True)

st.markdown("---")
st.caption("Sivita Virtual Assistant Poltesa @2026")

# --- 6. LOGIKA BACKEND ---
if btn_kirim:
    if not is_valid_email(email):
        st.error("Gunakan email @gmail.com")
    elif not user_query:
        st.warning("Tulis pertanyaan Anda.")
    else:
        with st.spinner("Sivita sedang mencari data..."):
            start_time = time.time()
            try:
                context = "\n".join(semantic_search(user_query, st.session_state.vector_store))
                llm = ChatOpenAI(
                    model="google/gemini-2.0-flash-lite-001",
                    openai_api_key=st.secrets["OPENROUTER_API_KEY"],
                    openai_api_base="https://openrouter.ai/api/v1",
                    temperature=0.1
                )
                full_prompt = f"{st.session_state.dynamic_sys_prompt}\n\nDATA:\n{context}\n\nPERTANYAAN: {user_query}"
                res = llm.invoke(full_prompt)
                st.session_state["last_answer"] = res.content
                st.session_state["last_duration"] = round(time.time() - start_time, 2)
                st.rerun()
            except Exception as e:
                st.error(f"Kesalahan: {e}")
