# File: ingest.py

import os
import toml
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

def load_secrets():
    """Memuat kunci API dari file .streamlit/secrets.toml"""
    secrets_path = os.path.join(".streamlit", "secrets.toml")
    if not os.path.exists(secrets_path):
        raise FileNotFoundError(f"File rahasia tidak ditemukan di {secrets_path}.")
    
    secrets = toml.load(secrets_path)
    # Memastikan PINECONE_API_KEY dan HUGGINGFACEHUB_API_TOKEN ada
    if "PINECONE_API_KEY" not in secrets or "HUGGINGFACEHUB_API_TOKEN" not in secrets:
        raise KeyError("Kunci API PINECONE_API_KEY dan HUGGINGFACEHUB_API_TOKEN harus ada di secrets.toml.")
        
    return secrets['PINECONE_API_KEY'], secrets['HUGGINGFACEHUB_API_TOKEN']

def main():
    """
    Fungsi utama untuk memuat, memproses, dan mengunggah data PDF ke Pinecone.
    """
    print("Memulai proses ingestion data...")

    # --- 1. Memuat Kunci API ---
    try:
        pinecone_api_key, hf_token = load_secrets()
        print("Kunci API berhasil dimuat.")
    except (FileNotFoundError, KeyError) as e:
        print(f"Error: {e}")
        return

    # --- SET ENVIRONMENT VARIABLES (PERBAIKAN KRUSIAL) ---
    # Setting environment variables di sini memastikan library Sentence Transformers/Hugging Face
    # dapat menemukan token untuk mengunduh model.
    os.environ["PINECONE_API_KEY"] = pinecone_api_key
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token # Tambahan untuk kompatibilitas
    # ---------------------------------------------------
    
    # --- 2. Memuat Dokumen PDF ---
    file_path = os.path.join("drug_data", "dataset_obat_new.pdf")

    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' tidak ditemukan. Pastikan nama folder dan file sudah benar.")
        return
        
    print(f"Memuat dokumen dari '{file_path}'...")
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print(f"Dokumen berhasil dimuat. Total halaman: {len(documents)}")

    print("Memecah dokumen menjadi potongan-potongan teks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs_split = text_splitter.split_documents(documents)
    print(f"Dokumen berhasil dipecah menjadi {len(docs_split)} potongan.")

    embedding_model_name = 'multi-qa-mpnet-base-dot-v1'
    print(f"Menginisialisasi model embedding: '{embedding_model_name}'...")
    # Model embedding akan menggunakan token HF dari environment variable
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    # --- 5. Menghubungkan ke Pinecone dan Melakukan Upsert ---
    index_name = "drug-chatbot-index"
    print(f"Menghubungkan ke indeks Pinecone: '{index_name}' dan memulai proses upsert...")
    
    try:
        # Menghapus parameter pinecone_api_key karena sudah disetel di os.environ
        PineconeVectorStore.from_documents(
            documents=docs_split,
            embedding=embeddings,
            index_name=index_name,
        )
        print("\n==================================================")
        print(f"✅ Sukses! Data telah berhasil dimasukkan ke indeks '{index_name}'.")
        print("Anda sekarang bisa menjalankan aplikasi chatbot utama (app.py).")
        print("==================================================")
        
    except Exception as e:
        print(f"Terjadi kesalahan saat proses upsert ke Pinecone: {e}")

if __name__ == "__main__":
    main()