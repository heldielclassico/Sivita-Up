import streamlit as st
import os
import pandas as pd
import requests
import re
import time
import numpy as np
from typing import List, Dict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
import faiss
from sentence_transformers import SentenceTransformer

# 1. Load Environment Variables
load_dotenv()

# 2. Konfigurasi Halaman
st.set_page_config(page_title="Asisten POLTESA", page_icon="🎓")

# --- 3. SCREEN LOADER (Muncul 5 detik saat pertama kali buka) ---
if "loaded" not in st.session_state:
    placeholder = st.empty()
    with placeholder.container():
        st.markdown("""
            <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 70vh;">
                <h1 style="color: #0e1117; font-family: sans-serif;">🎓 Sivita</h1>
                <p style="color: #555;">Menyiapkan Asisten Virtual Poltesa...</p>
                <div class="loader"></div>
            </div>
            <style>
                .loader {
                    border: 4px solid #f3f3f3;
                    border-top: 4px solid #3498db;
                    border-radius: 50%;
                    width: 80px;
                    height: 80px;
                    animation: spin 1s linear infinite;
                    margin-top: 20px;
                }
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
            </style>
        """, unsafe_allow_html=True)
        time.sleep(5)
    placeholder.empty()
    st.session_state["loaded"] = True

# --- 4. INISIALISASI SESSION STATE ---
if "last_answer" not in st.session_state:
    st.session_state["last_answer"] = ""
if "last_duration" not in st.session_state:
    st.session_state["last_duration"] = 0
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []

# --- FUNGSI VALIDASI EMAIL ---
def is_valid_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@gmail\.com$'
    return re.match(pattern, email) is not None

# --- FUNGSI CHUNKING ---
def create_chunks(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """Membagi teks menjadi chunk yang lebih kecil"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", ", ", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return chunks

# --- FUNGSI: AMBIL & PROSES DATA DENGAN CHUNKING ---
def get_and_process_data():
    """Ambil data dari Google Sheets dan proses dengan chunking"""
    try:
        central_url = st.secrets["SHEET_CENTRAL_URL"]
        df_list = pd.read_csv(central_url)
        tab_names = df_list['NamaTab'].tolist()
        base_url = central_url.split('/export')[0]
        
        all_data = []
        for tab in tab_names:
            tab_url = f"{base_url}/gviz/tq?tqx=out:csv&sheet={tab.replace(' ', '%20')}"
            try:
                df = pd.read_csv(tab_url)
                # Konversi DataFrame menjadi teks terstruktur
                text_data = f"### {tab.upper()} ###\n"
                for idx, row in df.iterrows():
                    row_text = ", ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                    text_data += f"Baris {idx+1}: {row_text}\n"
                all_data.append(text_data)
            except Exception as e:
                print(f"Error loading sheet {tab}: {e}")
                continue
        
        # Gabungkan semua data
        combined_text = "\n\n".join(all_data)
        
        # Buat chunks
        chunks = create_chunks(combined_text, chunk_size=800, chunk_overlap=100)
        
        # Simpan metadata untuk setiap chunk
        chunk_metadata = []
        for i, chunk in enumerate(chunks):
            chunk_metadata.append({
                "chunk_id": i,
                "text": chunk,
                "source": "google_sheets"
            })
        
        return chunk_metadata
        
    except Exception as e:
        print(f"Error in get_and_process_data: {e}")
        return []

# --- FUNGSI: BUAT VECTOR EMBEDDINGS ---
def create_vector_store(chunks: List[Dict]):
    """Buat vector embeddings untuk semantic search"""
    try:
        # Pilihan 1: Gunakan OpenAI Embeddings (berbayar, lebih akurat)
        # embeddings_model = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
        # texts = [chunk["text"] for chunk in chunks]
        # embeddings = embeddings_model.embed_documents(texts)
        
        # Pilihan 2: Gunakan SentenceTransformer lokal (gratis, offline)
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        texts = [chunk["text"] for chunk in chunks]
        embeddings = model.encode(texts, normalize_embeddings=True)
        
        # Konversi ke numpy array
        embeddings_array = np.array(embeddings).astype('float32')
        
        # Buat FAISS index
        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner Product untuk cosine similarity
        
        # Normalisasi embeddings untuk cosine similarity
        faiss.normalize_L2(embeddings_array)
        index.add(embeddings_array)
        
        return {
            "index": index,
            "chunks": chunks,
            "dimension": dimension
        }
        
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return None

# --- FUNGSI: SEMANTIC SEARCH ---
def semantic_search(query: str, vector_store: Dict, top_k: int = 3) -> List[str]:
    """Cari chunk yang paling relevan dengan query"""
    try:
        # Encode query
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        query_embedding = model.encode([query], normalize_embeddings=True)
        query_embedding = np.array(query_embedding).astype('float32')
        
        # Normalisasi
        faiss.normalize_L2(query_embedding)
        
        # Search
        distances, indices = vector_store["index"].search(query_embedding, top_k)
        
        # Ambil chunk yang relevan
        relevant_chunks = []
        for idx, score in zip(indices[0], distances[0]):
            if idx < len(vector_store["chunks"]):
                chunk = vector_store["chunks"][idx]
                relevant_chunks.append({
                    "text": chunk["text"],
                    "score": float(score)
                })
        
        # Urutkan berdasarkan skor
        relevant_chunks.sort(key=lambda x: x["score"], reverse=True)
        
        return [chunk["text"] for chunk in relevant_chunks]
        
    except Exception as e:
        print(f"Error in semantic search: {e}")
        return []

# --- FUNGSI: SIMPAN LOG ---
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

# --- INISIALISASI DATA (Saat pertama kali run) ---
if "data_initialized" not in st.session_state:
    with st.spinner("📥 Memuat dan memproses data Poltesa..."):
        chunks = get_and_process_data()
        if chunks:
            vector_store = create_vector_store(chunks)
            st.session_state.vector_store = vector_store
            st.session_state.chunks = chunks
            st.session_state.data_initialized = True
            st.success(f"✅ Data berhasil diproses: {len(chunks)} chunks")
        else:
            st.error("❌ Gagal memuat data")

# --- FUNGSI: HAPUS CHAT ---
def clear_text():
    st.session_state["user_input"] = ""

# --- FUNGSI: GENERATE RESPONSE DENGAN RAG ---
def generate_response_with_rag(user_email, user_input):
    start_time = time.time()
    
    try:
        # 1. SEMANTIC SEARCH: Cari chunk yang relevan
        relevant_chunks = []
        if st.session_state.vector_store:
            relevant_chunks = semantic_search(user_input, st.session_state.vector_store, top_k=3)
        
        # 2. Siapkan context dari chunk yang relevan
        context = ""
        if relevant_chunks:
            context = "INFORMASI RELEVAN:\n"
            for i, chunk in enumerate(relevant_chunks):
                context += f"\n--- Chunk {i+1} ---\n{chunk}\n"
        else:
            # Fallback: Gunakan beberapa chunk pertama
            context = "INFORMASI UMUM:\n"
            for i, chunk in enumerate(st.session_state.chunks[:3]):
                context += f"\n--- Informasi {i+1} ---\n{chunk['text']}\n"
        
        # 3. Siapkan prompt dengan RAG
        instruction = st.secrets["SYSTEM_PROMPT"]
        api_key_secret = st.secrets["OPENROUTER_API_KEY"]
        
        final_prompt = f"""{instruction}

{context}

PERTANYAAN USER: {user_input}

PERINTAH:
1. Jawab hanya berdasarkan informasi di atas
2. Jika informasi tidak cukup, katakan "Maaf, informasi ini belum tersedia dalam database Poltesa"
3. Berikan jawaban yang jelas dan terstruktur
4. Gunakan bahasa Indonesia yang formal

JAWABAN:"""
        
        # 4. Generate response dengan LLM
        model = ChatOpenAI(
            model="google/gemini-2.0-flash-lite-001",
            openai_api_key=api_key_secret,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.3,
            max_tokens=1000
        )
        
        response = model.invoke(final_prompt)
        end_time = time.time()
        duration = round(end_time - start_time, 2)
        
        if response and response.content:
            st.session_state["last_answer"] = response.content
            st.session_state["last_duration"] = duration
            
            # Log dengan informasi chunk yang digunakan
            chunk_info = f"Chunks used: {len(relevant_chunks)}"
            save_to_log(user_email, user_input, response.content, duration)
        else:
            st.warning("AI tidak dapat merumuskan jawaban.")
                
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "402" in error_msg:
            st.error("Mohon Maaf Kami sedang mengalami Gangguan Teknis. \n\n Web : https://poltesa.ac.id/")
        else:
            st.error(f"Terjadi kesalahan teknis: {e}")

# --- CSS KUSTOM ---
st.markdown("""
    <style>
    .stTextArea textarea { border-radius: 10px; }
    .stTextInput input { border-radius: 10px; }
    .stButton button { border-radius: 20px; }
    .stForm { margin-bottom: 0px !important; }
    .stCaption {
        margin-top: -15px !important;
        padding-top: 0px !important;
    }
    .duration-info {
        font-size: 0.75rem;
        color: #9ea4a9;
        margin-top: 2px;
        margin-bottom: 15px;
        font-style: italic;
    }
    .chunk-info {
        font-size: 0.7rem;
        color: #666;
        background-color: #f8f9fa;
        padding: 8px;
        border-radius: 5px;
        margin-bottom: 10px;
        border-left: 4px solid #3498db;
    }
    </style>
    """, unsafe_allow_html=True)

# --- HEADER ---
st.title("🎓 Asisten Virtual Poltesa (Sivita)")
st.markdown("<p style='margin-top: -20px; font-size: 0.8em; color: gray;'>Versi Beta V1.0 dengan RAG System</p>", unsafe_allow_html=True)

# --- INFO DATA ---
if st.session_state.get("data_initialized"):
    st.markdown(f"""
    <div class="chunk-info">
    📊 <strong>Database Status:</strong> {len(st.session_state.chunks)} chunks data tersedia<br>
    🔍 <strong>Search Method:</strong> Semantic Search dengan embedding
    </div>
    """, unsafe_allow_html=True)

# --- 5. UI FORM ---
with st.form("chat_form", clear_on_submit=False):
    user_email = st.text_input(
        "Masukan GMail Wajib (Format: nama@gmail.com):", 
        placeholder="contoh@gmail.com",
        key="user_email"
    )
    
    user_text = st.text_area(
        "Tanyakan sesuatu tentang Poltesa:",
        placeholder="Halo! Saya Sivita, ada yang bisa saya bantu?",
        key="user_input" 
    )
    
    col1, col2 = st.columns([1, 1]) 
    
    with col1:
        submitted = st.form_submit_button("Kirim", use_container_width=True)
    with col2:
        st.form_submit_button("Hapus Chat", on_click=clear_text, use_container_width=True)
    
    if submitted:
        if not user_email:
            st.error("Alamat email wajib diisi!")
        elif not is_valid_email(user_email):
            st.error("Format email salah! Harus menggunakan @gmail.com")
        elif user_text.strip() == "":
            st.warning("Mohon masukkan pertanyaan terlebih dahulu.")
        else:
            if not st.session_state.get("data_initialized"):
                st.warning("⚠️ Data belum siap. Silakan refresh halaman.")
            else:
                with st.spinner("🔍 Mencari informasi relevan..."):
                    generate_response_with_rag(user_email, user_text)

# --- BAGIAN HASIL ---
if st.session_state["last_answer"]:
    st.chat_message("assistant").markdown(st.session_state["last_answer"])
    st.markdown(f'<p class="duration-info">⏱️ Pencarian selesai dalam {st.session_state["last_duration"]} detik</p>', unsafe_allow_html=True)

# --- SIDEBAR UNTUK MONITORING ---
with st.sidebar:
    st.header("📈 RAG Monitoring")
    
    if st.session_state.get("data_initialized"):
        st.metric("Total Chunks", len(st.session_state.chunks))
        
        if st.button("🔍 Test Semantic Search"):
            test_query = st.text_input("Query test:", "jadwal kuliah")
            if test_query:
                relevant = semantic_search(test_query, st.session_state.vector_store, top_k=2)
                if relevant:
                    st.write("**Chunks ditemukan:**")
                    for i, chunk in enumerate(relevant):
                        st.text_area(f"Chunk {i+1}", chunk[:300] + "...", height=100)
    
    st.markdown("---")
    st.caption("**Chunking Strategy:**")
    st.markdown("""
    - Chunk size: 800 karakter
    - Overlap: 100 karakter
    - Embedding: multilingual-MiniLM
    - Top-k retrieval: 3 chunks
    """)

# Footer
st.caption("Sivita - Sistem Informasi Virtual Asisten Poltesa @2026 | Powered by RAG System")
