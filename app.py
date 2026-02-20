import streamlit as st
import pandas as pd
import requests
import re
import time
import numpy as np
from typing import List, Dict, Tuple
from streamlit_lottie import st_lottie
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer
import faiss

# --- 1. KONFIGURASI HALAMAN & UI ---
st.set_page_config(page_title="Asisten POLTESA", page_icon="🎓", layout="centered")

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    .stAppDeployButton {display: none;}
    .answer-box {
        padding: 20px; background-color: #ffffff; border-radius: 12px;
        border: 1px solid #dee2e6; margin-bottom: 15px; line-height: 1.6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. FUNGSI LOGIKA DATA (DATABASE) ---
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
                # Jika tab PROMPT, simpan sebagai instruksi dasar
                if tab.lower() == 'prompt':
                    full_instructions.extend(df['Isi'].dropna().astype(str).tolist())
                    continue
                
                # Olah data per baris
                for _, row in df.iterrows():
                    # Gabungkan kolom menjadi satu teks konteks yang kaya informasi
                    content = f"SUMBER {tab.upper()}: " + " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                    all_chunks.append({"text": content, "source": tab})
            except: continue
            
        return all_chunks, "\n".join(full_instructions)
    except Exception as e:
        st.error(f"Gagal memuat Database: {e}")
        return [], ""

def create_vector_store(chunks_data: List[Dict]):
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    texts = [c["text"] for c in chunks_data]
    embeddings = model.encode(texts, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype('float32'))
    return {"index": index, "chunks": chunks_data, "model": model}

def semantic_search(query: str, vector_store: Dict, top_k: int = 10):
    # top_k=10 memastikan aturan sistem (seperti JadwalKu) ikut terambil
    query_vec = vector_store["model"].encode([query], normalize_embeddings=True)
    _, indices = vector_store["index"].search(query_vec.astype('float32'), top_k)
    return [vector_store["chunks"][idx]["text"] for idx in indices[0] if idx < len(vector_store["chunks"])]

# --- 3. INISIALISASI SESSION STATE ---
if "vector_store" not in st.session_state:
    with st.spinner("Mensinkronkan Data dari Google Sheets..."):
        raw_chunks, sys_prompt = get_and_process_data()
        if raw_chunks:
            st.session_state.vector_store = create_vector_store(raw_chunks)
            st.session_state.dynamic_sys_prompt = sys_prompt
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "last_answer" not in st.session_state: st.session_state.last_answer = ""

# --- 4. UI UTAMA ---
st.title("🎓 Sivita AI")
email = st.sidebar.text_input("Email Gmail:", placeholder="nama@gmail.com")

if st.sidebar.button("🔄 Sinkron Ulang Data"):
    st.cache_data.clear()
    st.session_state.clear()
    st.rerun()

# Display Jawaban Terakhir
if st.session_state.last_answer:
    st.markdown(f'<div class="answer-box"><strong>🤖 Sivita:</strong><br>{st.session_state.last_answer}</div>', unsafe_allow_html=True)

# Input Pertanyaan
query = st.chat_input("Tanyakan apa saja atau ketik 'JadwalKu'...")

if query:
    if not re.match(r'^[a-zA-Z0-9._%+-]+@gmail\.com$', email or ""):
        st.error("Masukkan email @gmail.com yang valid di sidebar!")
    else:
        with st.spinner("Mencari informasi..."):
            try:
                # 1. Cari Konteks (AI akan menemukan baris 'JadwalKu' di sini)
                context_list = semantic_search(query, st.session_state.vector_store)
                context_text = "\n".join(context_list)
                
                # 2. Ambil Riwayat
                history_text = "\n".join([f"U: {c['u']}\nA: {c['b']}" for c in st.session_state.chat_history[-3:]])
                
                # 3. Prompting (Mengarahkan AI untuk mematuhi aturan di Konteks)
                # Aturan Sivita 2026-01-22: Jangan sebut Google Sheets
                sys_instruction = (
                    f"{st.session_state.dynamic_sys_prompt}\n"
                    "INSTRUKSI UTAMA:\n"
                    "- Gunakan 'KONTEKS DATA' sebagai satu-satunya rujukan.\n"
                    "- Jika ada instruksi khusus mengenai kata kunci tertentu (seperti JadwalKu) di dalam data, jalankan instruksi tersebut dengan tepat.\n"
                    "- Jangan pernah menyebutkan 'tabel Google Sheets' dalam jawaban Anda."
                )

                prompt = f"{sys_instruction}\n\nKONTEKS DATA:\n{context_text}\n\nRIWAYAT:\n{history_text}\n\nUSER: {query}"
                
                # 4. Eksekusi LLM
                llm = ChatOpenAI(
                    model="google/gemini-2.0-flash-lite-001",
                    openai_api_key=st.secrets["OPENROUTER_API_KEY"],
                    openai_api_base="https://openrouter.ai/api/v1",
                    temperature=0.1
                )
                
                response = llm.invoke(prompt)
                answer = response.content
                
                # 5. Simpan State
                st.session_state.chat_history.append({"u": query, "b": answer})
                st.session_state.last_answer = answer
                st.rerun()
                
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")

st.caption("Sivita - Database-Driven AI Poltesa @2026")
