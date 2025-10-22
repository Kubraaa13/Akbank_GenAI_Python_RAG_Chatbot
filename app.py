import streamlit as st
import os


st.set_page_config(
    page_title="Akbank RAG Python Chatbot 🐍", 
    layout="wide", 
    initial_sidebar_state="expanded" 
)


from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

# --- AYARLAR VE GÜVENLİK ---
LLM_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "models/text-embedding-004"


DATA_FILE_PATH = os.path.join(
    os.path.dirname(__file__), 
    "python_kod_aciklamalari.txt"
)


st.write(f"⚠️ DEBUG: Kesinleştirilmiş Mutlak Yol: {DATA_FILE_PATH}")


# --- RAG FONKSİYONLARI ---


def load_and_chunk_data(file_path):
    """Veri dosyasını yükler ve RAG için parçalara böler."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
    except FileNotFoundError:
        
        st.error(f"HATA: Veri dosyası ({file_path}) bulunamadı. Lütfen dosyanın GitHub'a yüklendiğinden emin olun.")
        return []

    text_splitter = CharacterTextSplitter(
        separator="\nKOD-", 
        chunk_size=3000,
        chunk_overlap=0 
    )
    texts = text_splitter.split_text(raw_text)
    return [f"KOD-{t.strip()}" for t in texts if t.strip()]

@st.cache_resource 
def create_rag_chain():
    """RAG zincirini kurar (Veri yükleme, Embedding ve ChromaDB)."""
    
    
    if "GOOGLE_API_KEY" not in st.secrets:
        st.error("Lütfen Streamlit ayarlarında 'Secrets' bölümüne GOOGLE_API_KEY'i ekleyin.")
        st.warning("Örnek: GOOGLE_API_KEY=\"AIzaSy...\"")
        return None
    
    
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"] 

    
    knowledge_base = load_and_chunk_data(DATA_FILE_PATH) 
    if not knowledge_base:
        return None

    
    try:
        embedding_model = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
        vectorstore = Chroma.from_texts(
            texts=knowledge_base, 
            embedding=embedding_model, 
            persist_directory="./chroma_db"
        )
    except Exception as e:
        st.error(f"RAG Kurulum Hatası: API erişiminde sorun oluştu. Detay: {e}")
        st.warning("Lütfen API kotanızı kontrol edin.")
        return None

    
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL, 
        temperature=0.2,
        convert_system_message_to_human=True 
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=vectorstore.as_retriever()
    )
    return qa_chain


# --- STREAMLIT WEB ARAYÜZÜ ANA FONKSİYONU ---

def main():
    st.title("🐍 Python Kod Rehberi Chatbot")
    st.subheader("Gemini Destekli RAG Uygulaması")
    st.markdown("---")

    qa_chain = create_rag_chain()

    if qa_chain is None:
        st.warning("Chatbot başlatılamadı. Lütfen yukarıdaki hataları giderin.")
        return

    st.success("Chatbot Kullanıma Hazır!")

    user_query = st.text_input("Python kodları hakkında bir soru sorun (Örn: 'Sınıf nasıl tanımlanır?')")

    if user_query:
        with st.spinner("Cevap Üretiliyor..."):
            try:
                result = qa_chain.invoke({"query": user_query})
                response_text = result.get('result', 'Cevap alınamadı.')

                st.markdown("### 🤖 Bot Cevabı:")
                st.markdown(response_text) 
                
            except Exception as e:
                st.error(f"Sorgulama Sırasında Hata: API'ye bağlanılamadı veya kod çalışmadı. Detay: {e}")

if __name__ == '__main__':
    main()