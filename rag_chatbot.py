import os
from dotenv import load_dotenv 
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA


load_dotenv()

# 1. AYARLAR
LLM_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "models/text-embedding-004" 
DATA_FILE_PATH = "python_kod_aciklamalari.txt" # TXT dosyanızın adı

# 2. VERİ YÜKLEME VE PARÇALAMA (Chunking)
def load_and_chunk_data(file_path):
    """TXT dosyasını yükler ve RAG için parçalara (chunks) böler."""
    print(f"Veri yükleniyor ve parçalanıyor: {file_path}")
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
    except FileNotFoundError:
        print(f"HATA: Dosya bulunamadı: {file_path}. Lütfen dosyayı bu klasöre ekleyin.")
        return []

    text_splitter = CharacterTextSplitter(
        separator="\nKOD-", 
        chunk_size=3000,
        chunk_overlap=0 
    )
    texts = text_splitter.split_text(raw_text)
    
    
    cleaned_texts = [f"KOD-{t.strip()}" for t in texts if t.strip()]
    
    return cleaned_texts

# 3. EMBEDDING VE VEKTÖR VERİ TABANI OLUŞTURMA
def create_vector_store(texts):
    """Metin parçalarını vektörleştirir ve ChromaDB'de saklar."""
    print("Vektörleştirme (Embedding) yapılıyor...")
    
    
    embedding_model = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    
    
    vectorstore = Chroma.from_texts(
        texts=texts, 
        embedding=embedding_model, 
        persist_directory="./chroma_db"
    )
    print("Vektör veritabanı hazırlandı.")
    return vectorstore


# 4. SORGULAMA VE YANIT ÜRETME (run_chatbot fonksiyonunun içinde)
def run_chatbot(vectorstore):
    # ... (Diğer tanımlar) ...
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0.2,
        convert_system_message_to_human=True
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )

    while True:
        query = input("Siz: ")
        if query.lower() == 'exit':
            print("Güle güle!")
            break
        
    
        
        try:
            
            result = qa_chain.invoke({"query": query}) 
            
            
            response_text = result.get('result', 'Cevap alınamadı.')
            
            print(f"\nBot: {response_text}\n")
        
        except Exception as e:
            
            print(f"\nBOT HATA: Sorgu sırasında bir hata oluştu. Detay: {e}\n")


# ... (Diğer kodlar) ...

# 5. ANA ÇALIŞMA BLOKU
if __name__ == "__main__":
    if "GOOGLE_API_KEY" not in os.environ:
        print("\n!! HATA: GOOGLE_API_KEY bulunamadı. Lütfen .env dosyanızı kontrol edin. !!\n")
    else:
        # 1. Veri Yükleme
        knowledge_base = load_and_chunk_data(DATA_FILE_PATH)
        if knowledge_base:
            # 2. Vektör Veri Tabanı Oluşturma
            db = create_vector_store(knowledge_base)
            # 3. Chatbot'u Çalıştırma
            run_chatbot(db)