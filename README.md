# 🐍 Python Kod Rehberi Chatbot (Gemini Destekli RAG Uygulaması)
Python kodları hakkında bilgi veren ve örnekler sunan, Gemini API ve RAG mimarisi ile geliştirilmiş chatbot projesi.

## 🚀 Canlı Uygulama Linki
[https://akbankgenaipythonragchatbot-qwe9egrpuvfbxg2fazt9ug.streamlit.app/]
 
 Bu projenin amacı RAG mimarisi temelli bir yapay zeka sohbet robotu geliştirmektir. Geliştirilen chatbot, temel ve orta düzey Python kod parçacıkları ve kavramları hakkında kullanıcı sorularına, harici bir bilgi kaynağından (veri seti) bilgi çekerek doğru, ilgili ve çalışan kod örnekleri ile zenginleştirilmiş yanıtlar vermeyi hedefler. 

## Veri Seti Hakkında Bilgi:                                                                                
Veri Seti Konusu: Temel Python programlama yapıları, Nesne Tabanlı Programlama (OOP), hata yönetimi ve popüler harici kütüphane (NumPy) kullanımı gibi 8 farklı ana başlıkta kod örnekleri ve açıklamaları.                                                                                              

## Hazırlanış Metodolijisi: 
Veri seti, her biri kendi içinde bütüncül bir bilgi parçası (chunk) oluşturacak şekilde tasarlanmıştır. Her parça, konu başlığı, teorik tanım, çalışan Python kod bloğu ve kodun çıktısı/açıklaması bileşenlerini içerir. Bu yapı, RAG sisteminin sorgulara en alakalı, bütün halseki bilgiyi çekebilmesi için kritiktir.       

### 🛠 Kullanılan Yöntemler ve Çözüm Mimarisi

Proje, standart bir RAG mimarisini takip etmektedir. [cite_start]Çözüm mimarisi detayları bu adımda yer alacaktır[cite: 23].

| Aşama | Kullanılacak Teknolojiler (Örnek) | Açıklama |
| :--- | :--- | :--- |
| **Büyük Dil Modeli (LLM)** | [cite_start]**Gemini API** [cite: 33, 42] | Yanıt üretimi (Generation) için kullanılacaktır. |
| **RAG Pipeline Çatısı** | [cite_start]LangChain veya Haystack [cite: 44] | Vektörleştirme, arama ve LLM çağrısı gibi adımları yönetecek ana iskelet. |
| **Vektör Veri Tabanı** | [cite_start]ChromaDB, FAISS veya Pinecone [cite: 43] | Veri setinin vektörleştirilmiş halinin depolandığı, hızlı anlamsal arama sağlayan veritabanı. |
| **Embedding Model** | [cite_start]Google'ın Embedding Modeli veya açık kaynaklı bir model [cite: 43] | Metinleri anlamsal vektörlere dönüştürmek için kullanılacaktır. |
