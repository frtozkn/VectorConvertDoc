from sentence_transformers import SentenceTransformer
import os

# Modeli indir ve kaydet
model_name = "all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)
model.save("offline_model")  # Modeli kaydet
print("Model başarıyla indirildi ve kaydedildi.");