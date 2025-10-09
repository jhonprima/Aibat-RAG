import streamlit as st
from pinecone import Pinecone as PineconeClient
import os

# --- Import yang relevan ---
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_cohere import ChatCohere
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# --- --- --- --- --- --- --- --- --- --- --- ---
# --- BAGIAN 1: KONFIGURASI TAMPILAN (UI) ---
# --- --- --- --- --- --- --- --- --- --- --- ---

st.set_page_config(
    page_title="AIbat | Sahabat Pencari Obat",
    page_icon="🌿",
    layout="wide"
)

# CSS Kustom untuk tema "Sahabat Pencari Obat" yang segar dan modern
st.markdown("""
<style>
    /* Mengubah font utama menjadi lebih modern dan mudah dibaca */
    html, body, [class*="css"]  {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji";
    }
    
    /* Menghilangkan header default Streamlit */
    .st-emotion-cache-18ni7ap {
        display: none;
    }
    
    /* Kustomisasi area chat utama */
    .st-emotion-cache-1c7y2kd {
        background-color: #FFFFFF;
    }

    /* Bubble chat pengguna (kanan) - Warna biru yang ramah */
    [data-testid="chat-bubble-user"] {
        background-color: #E0F7FA;
    }

    /* Bubble chat asisten (kiri) - Warna netral dan bersih */
    [data-testid="chat-bubble-assistant"] {
        background-color: #F1F3F4;
    }

    /* Sidebar - Warna hijau alami yang menenangkan */
    .st-emotion-cache-6q9sum {
        background-color: #E8F5E9;
    }

    /* Tombol logout di sidebar */
    .st-emotion-cache-7ym5gk {
        border-radius: 12px;
        border: 2px solid #388E3C;
        color: #388E3C;
    }
    .st-emotion-cache-7ym5gk:hover {
        background-color: #C8E6C9;
        color: #2E7D32;
        border: 2px solid #2E7D32;
    }
</style>
""", unsafe_allow_html=True)

# --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# --- BAGIAN 2: LOGIKA BACKEND CHATBOT ---
# --- --- --- --- --- --- --- --- --- --- --- --- --- ---

try:
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
    COHERE_API_KEY = st.secrets["COHERE_API_KEY"]
    INDEX_NAME = "drug-chatbot-index"
    EMBEDDING_MODEL_NAME = 'multi-qa-mpnet-base-dot-v1'
except KeyError as e:
    st.error(f"Kunci API hilang dari secrets.toml: {e}. Pastikan semua kunci sudah terisi.")
    st.stop()

@st.cache_resource
def setup_rag():
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        pc = PineconeClient(api_key=PINECONE_API_KEY)
        index = pc.Index(INDEX_NAME)
        vectorstore = PineconeVectorStore(
            index=index,
            embedding=embeddings,
            text_key="text"
        )
        llm = ChatCohere(
            cohere_api_key=COHERE_API_KEY,
            model="command-a-03-2025",
            temperature=0.3
        )
        
        # --- PROMPT TAHAN BANTING (ROBUST) ---
        prompt = ChatPromptTemplate.from_template("""
        Anda adalah "AIbat", seorang sahabat apoteker yang cerdas, empatik, dan sangat berpengetahuan tentang obat-obatan Indonesia.
        Gunakan gaya bahasa yang ramah, sopan, dan mudah dimengerti.

        PETUNJUK UTAMA:
        1. FOKUS KETAT PADA KONTEKS: Jawab pertanyaan pengguna HANYA berdasarkan informasi yang ada di bagian KONTEKS.
        2. ANALISIS INPUT: Analisis pertanyaan pengguna {input} untuk menentukan kategorinya.
        3. JAGA KEAMANAN: Selalu ingat bahwa Anda adalah AI informasi, bukan pengganti saran medis profesional.

        ATURAN RESPON:
        - KATEGORI OBAT: Jika informasi tersedia, rangkum data dari konteks.
        - KATEGORI KETERBATASAN DATA (Out-of-Context): Jika KONTEKS sama sekali tidak mengandung informasi yang relevan, jawab: "Mohon maaf, saya tidak menemukan informasi spesifik mengenai hal tersebut di dalam dokumen obat-obatan saya."
        - KATEGORI NON-TOPIK (Off-topic, Sapaan, dsb): Jika pertanyaan {input} di luar topik obat-obatan atau hanya sapaan sederhana (misal: 'halo', 'bagaimana kabar'), jawablah dengan sopan dan alihkan kembali ke topik. Contoh: "Halo! Senang bertemu Anda. Fokus saya adalah informasi obat. Ada yang bisa saya bantu terkait obat-obatan?"

        ---
        Konteks Pencarian:
        {context}
        ---
        Pertanyaan Pengguna: {input}
        
        Jawaban AIbat:
        """)

        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        return retrieval_chain
        
    except Exception as e:
        st.error(f"Gagal menyiapkan RAG: {e}.")
        st.stop()

# --- --- --- --- --- --- --- --- --- --- --- ---
# --- BAGIAN 3: TAMPILAN UTAMA (UI) APLIKASI ---
# --- --- --- --- --- --- --- --- --- --- --- ---

# --- SIDEBAR ---
with st.sidebar:
    st.header("Tentang AIbat")
    st.info("AIbat adalah sahabat AI Anda untuk menjelajahi dunia obat-obatan.")
    st.warning("⚠️ Informasi ini bukan pengganti nasihat medis.")
    if "username" in st.session_state and st.session_state.username:
        st.markdown(f"Selamat datang, **{st.session_state.username}**!")
        if st.button("Logout", use_container_width=True):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()

# --- FUNGSI HALAMAN CHAT ---
def show_chat_page():
    st.header("Cari Obat, Untuk Selamat.")
    st.title("AIbat, Sahabat Pencari Obat Anda 🌿")
    st.divider()

    qa_chain = setup_rag()

    # Inisialisasi riwayat chat
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": f"Halo, {st.session_state.username}! Ada pertanyaan seputar obat yang bisa saya bantu hari ini?"}]

    # Tampilkan pesan dari riwayat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Tanyakan tentang dosis, efek samping, dll..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("AIbat sedang berpikir dan mencari..."):
                try:
                    # Menambahkan nama pengguna ke input untuk prompt
                    response = qa_chain.invoke({"input": prompt, "username": st.session_state.username})
                    answer = response["answer"]
                    
                    source_documents = response.get('context', [])
                    sources_list = []
                    
                    # Logika untuk menampilkan Sumber Informasi
                    if source_documents and "tidak menemukan informasi" not in answer:
                        for doc in source_documents:
                            source_file = doc.metadata.get('source', 'Tidak diketahui')
                            sources_list.append(f"- {os.path.basename(source_file)}")
                    
                    sources_unique = "\n".join(sorted(list(set(sources_list))))
                    
                    if sources_unique:
                        full_response = f"{answer}\n\n---\n*Sumber Informasi:*\n`{sources_unique}`"
                    else:
                        full_response = answer

                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    st.markdown(full_response)
                except Exception as e:
                    st.error(f"Terjadi kesalahan: {e}")

# --- FUNGSI HALAMAN INPUT NAMA ---
def show_name_input_page():
    st.title("Selamat Datang di AIbat 🌿")
    st.subheader("Sahabat Cerdas Anda untuk Informasi Obat")
    with st.form("name_form"):
        username = st.text_input("Siapa nama Anda?")
        submitted = st.form_submit_button("Mulai Berbicara dengan AIbat")
        if submitted and username.strip():
            st.session_state.username = username.strip()
            st.rerun()

# --- LOGIKA UTAMA ---
if "username" not in st.session_state or not st.session_state.username:
    show_name_input_page()
else:
    show_chat_page()