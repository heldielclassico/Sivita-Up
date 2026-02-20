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

# CSS untuk UI yang bersih dan responsif
st.markdown(f"""
    <style>
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    .block-container {{ padding-top: 35px; padding-bottom: 0rem; }}
    .stAppDeployButton {{display: none;}}
    
    /* Box Jawaban */
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

# --- 2. FUNGSI HELPER & ANIMASI ---
@st.cache_data
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=5)
        return r.json() if r.status_code == 200 else None
    except: return None

lottie_anim = load_lottieurl("https://lottie.host/85590396-981a-466d-961f-f46328325603/6P7qXJ5v6A.json")

def is_valid_email(email):
    return re.match(r'^[a-zA-Z0-9._%+-]+@gmail\.com$', email) is not None

def clear_input_only():
    st.session_state["user_query_input"] = ""

def clear_history():
    st.session_state["chat_history"] = []
    st.session_state["last_answer"] = ""
    st.session_state["last_duration"] = 0

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
                    # Filter pesan terlarang agar tidak masuk ke sistem prompt
                    forbidden_phrase = "tabel google sheets"
                    instr = [str(i) for i in df['Isi'].dropna() if forbidden_phrase not in str(i).lower()]
                    full_instructions.extend(instr)
                    continue
                
                for _, row in df.iterrows():
                    # Tambahkan Nama Tab di depan data agar pencarian lebih akurat (Penting untuk data Tendik)
                    content = f"DATABASE {tab.upper()}: " + " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                    all_chunks.append({"text": content, "source": tab})
            except: continue
            
        final_prompt = "\n".join(full_instructions) if full_instructions else "Anda adalah Sivita."
        return all_chunks, final_prompt
    except Exception as e:
        st.error(f"Koneksi Database Gagal: {e}")
        return [], ""

def create_vector_store(chunks_data: List[Dict]):
    try:
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        texts = [c["text"] for c in chunks_data]
        embeddings = model.encode(texts, normalize_embeddings=True)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings.astype('float32'))
        return {"index": index, "chunks": chunks_data, "model": model}
    except Exception as e:
        st.error(f"Gagal Inisialisasi AI: {e}")
        return None

def semantic_search(query: str, vector_store: Dict, top_k: int = 15):
    # top_k dinaikkan ke 8 agar data statistik tidak terlewat oleh data lain
    query_vec = vector_store["model"].encode([query], normalize_embeddings=True)
    _, indices = vector_store["index"].search(query_vec.astype('float32'), top_k)
    return [vector_store["chunks"][idx]["text"] for idx in indices[0] if idx < len(vector_store["chunks"])]

def save_to_log(email, question, answer="", duration=0):
    try:
        log_url = st.secrets["LOG_URL"]
        payload = {"email": email, "question": question, "answer": answer, "duration": f"{duration} detik"}
        requests.post(log_url, json=payload, timeout=5)
    except: pass

# --- 4. INISIALISASI STATE ---
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
    st.session_state.dynamic_sys_prompt = ""
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "last_answer" not in st.session_state: st.session_state.last_answer = ""
if "last_duration" not in st.session_state: st.session_state.last_duration = 0

# --- 5. LOADING SCREEN (SINKRONISASI) ---
if st.session_state.vector_store is None:
    with st.container():
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        if lottie_anim: st_lottie(lottie_anim, height=250)
        st.markdown("<h3 style='text-align: center; color: #007bff;'>Sinkronisasi Data Poltesa...</h3>", unsafe_allow_html=True)
        
        raw_data, dyn_prompt = get_and_process_data()
        if raw_data:
            st.session_state.vector_store = create_vector_store(raw_data)
            st.session_state.dynamic_sys_prompt = dyn_prompt
            time.sleep(1)
            st.rerun()
    st.stop()

# --- 6. UI UTAMA ---
st.markdown("<h1 style='text-align: center; margin-top: -30px;'>🎓 Sivita AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Virtual Assistant POLTESA - Versi Memori Aktif</p>", unsafe_allow_html=True)

with st.container(border=True):
    email = st.text_input("Email Gmail Anda:", placeholder="nama@gmail.com")
    
    col_sync, col_reset = st.columns(2)
    with col_sync:
        if st.button("🔄 Sinkron Ulang", use_container_width=True):
            st.cache_data.clear()
            st.session_state.vector_store = None
            st.rerun()
    with col_reset:
        if st.button("🧹 Hapus Chat", on_click=clear_history, use_container_width=True): pass

    placeholder_animasi = st.empty()

    # Tampilan Jawaban Terakhir
    if st.session_state.last_answer:
        st.markdown(f'<div class="answer-box"><strong>🤖 Sivita:</strong><br>{st.session_state.last_answer}</div>', unsafe_allow_html=True)
        st.caption(f"⏱️ Selesai dalam {st.session_state.last_duration} detik")

    # Area Input Pertanyaan
    with st.container(border=True):
        user_query = st.text_area("Apa yang ingin Anda tanyakan?", placeholder="Contoh: Berapa jumlah tendik?", key="user_query_input", height=120)
        col_send, col_del_q = st.columns([1.5, 1])
        with col_send:
            btn_kirim = st.button("Kirim Pertanyaan 🚀", use_container_width=True, type="primary")
        with col_del_q:
            st.button("Hapus Teks 🗑️", on_click=clear_input_only, use_container_width=True)

    # Logika Pengolahan Jawaban
    if btn_kirim:
        if not is_valid_email(email):
            st.error("Gunakan email @gmail.com")
        elif not user_query:
            st.warning("Silakan tulis pertanyaan Anda.")
        else:
            with placeholder_animasi.container():
                if lottie_anim: st_lottie(lottie_anim, height=150)
                st.markdown("<p style='text-align: center;'>Sivita sedang mencari di database...</p>", unsafe_allow_html=True)
            
            start_time = time.time()
            try:
                # 1. Ambil Konteks Data
                context_list = semantic_search(user_query, st.session_state.vector_store)
                context_text = "\n".join(context_list)
                
                # 2. Ambil Riwayat Percakapan (3 terakhir)
                history_text = "\n".join([f"User: {c['u']}\nSivita: {c['b']}" for c in st.session_state.chat_history[-3:]])
                
                # 3. Prompt Khusus (Aturan 2026-01-22 & Perbaikan Tendik)
                sys_rules = (
                    f"{st.session_state.dynamic_sys_prompt}\n"
                    "Aturan Penting:\n"
                    "- JANGAN PERNAH menyuruh user melihat Google Sheets untuk info sosial media.\n"
                    "- Jika ditanya jumlah tendik, dosen, atau mahasiswa, cari datanya di bagian DATABASE STATISTIK.\n"
                    "- Gunakan sapaan ramah 'Sobat Poltesa'.\n"
                )

                full_prompt = (
                    f"Sistem: {sys_rules}\n\n"
                    f"DATA REFERENSI:\n{context_text}\n\n"
                    f"RIWAYAT CHAT:\n{history_text}\n\n"
                    f"PERTANYAAN USER: {user_query}"
                )
                
                # 4. Panggil Gemini (OpenRouter)
                llm = ChatOpenAI(
                    model="google/gemini-2.0-flash-lite-001",
                    openai_api_key=st.secrets["OPENROUTER_API_KEY"],
                    openai_api_base="https://openrouter.ai/api/v1",
                    temperature=0.2
                )
                
                response = llm.invoke(full_prompt)
                ans_content = response.content
                
                # 5. Simpan ke History dan Log
                st.session_state.chat_history.append({"u": user_query, "b": ans_content})
                st.session_state.last_answer = ans_content
                st.session_state.last_duration = round(time.time() - start_time, 2)
                
                save_to_log(email, user_query, ans_content, st.session_state.last_duration)
                placeholder_animasi.empty()
                st.rerun()
                
            except Exception as e:
                placeholder_animasi.empty()
                st.error(f"Terjadi kesalahan: {e}")

st.caption("Sivita - Virtual Assistant Poltesa @2026")
