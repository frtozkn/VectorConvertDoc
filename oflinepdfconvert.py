import pdfplumber
from markdownify import markdownify
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 1. PDF Dosyasını Oku
def read_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text.strip()

# 2. Markdown'a Çevir
def convert_to_markdown(text):
    return markdownify(text)

# 3. Çevrimdışı Model Yükle ve Vektörleştir
def convert_to_vector(text):
    model = SentenceTransformer("offline_model")  # Çevrimdışı modeli yükle
    vector = model.encode([text])[0]
    return vector

# 4. FAISS İndex Oluştur
def create_faiss_index(vector):
    dimension = len(vector)
    index = faiss.IndexFlatL2(dimension)
    vector = np.array([vector], dtype=np.float32)
    index.add(vector)
    return index

# Kullanım
file_path = "files/test.pdf"  # PDF dosyanın yolu
text = read_pdf(file_path)
markdown_text = convert_to_markdown(text)
vector = convert_to_vector(markdown_text)
index = create_faiss_index(vector)

# Vektörü Kaydet
faiss.write_index(index, "vector.index")

# Markdown Dosyasını Kaydet
with open("outputpdf.md", "w", encoding="utf-8") as f:
    f.write(markdown_text)

print("PDF başarıyla Markdown'a çevrildi ve vektör olarak kaydedildi.")
