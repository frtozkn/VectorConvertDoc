from docx import Document
from markdownify import markdownify
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 1. DOCX Dosyasını Oku
def read_docx(file_path):
    doc = Document(file_path)
    text = "\n".join([p.text for p in doc.paragraphs])
    return text

# 2. Markdown'a Çevir
def convert_to_markdown(text):
    return markdownify(text)

# 3. Metni Vektörleştir
def convert_to_vector(text):
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Küçük ve hızlı bir model
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
file_path = "files/test.docx"  # DOCX dosyanın yolu
text = read_docx(file_path)
markdown_text = convert_to_markdown(text)
vector = convert_to_vector(markdown_text)
index = create_faiss_index(vector)

# Vektörü Kaydet
faiss.write_index(index, "vector.index")

print("DOCX başarıyla Markdown'a çevrildi ve vektör olarak kaydedildi.")

with open("output.md", "w", encoding="utf-8") as f:
    f.write(markdown_text)
