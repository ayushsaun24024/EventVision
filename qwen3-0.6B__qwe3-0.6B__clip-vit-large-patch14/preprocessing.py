import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
import faiss

class ArticleFolderDataset(Dataset):
    def __init__(self, json_data):
        self.items = list(json_data.items())

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        aid, art = self.items[idx]
        title = art.get("title", "")
        content = art.get("content", "")
        text = f"Title: {title}\nContent: {content}"
        return text, aid


def generate_embeddings(model, texts, batch_size):
    out = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch = texts[i:i+batch_size]
        emb = model.encode(batch, convert_to_numpy=True)
        out.append(emb)
    return np.vstack(out)


def build_faiss_index(emb_matrix):
    emb = emb_matrix.astype("float32")
    faiss.normalize_L2(emb)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return index

def process_articles(json_data, model_name, output_dir, batch_size):
    dataset = ArticleFolderDataset(json_data)
    texts = [dataset[i][0] for i in range(len(dataset))]
    ids = [dataset[i][1] for i in range(len(dataset))]

    model = SentenceTransformer(model_name, device="cuda", trust_remote_code=True)
    model.half()

    emb_matrix = generate_embeddings(model, texts, batch_size)

    np.save(os.path.join(output_dir, "database_embeddings_Qwen3.npy"), emb_matrix)
    np.save(os.path.join(output_dir, "database_article_ids_Qwen3.npy"), np.array(ids))

    index = build_faiss_index(emb_matrix)
    faiss.write_index(index, os.path.join(output_dir, "database_faiss_index_Qwen3.bin"))

if __name__ == "__main__":
    DATASET_JSON = "./database.json"
    OUTPUT_DIR = "./eventa_embeddings_Qwen3_test"
    MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
    BATCH_SIZE = 4

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(DATASET_JSON, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    process_articles(json_data, MODEL_NAME, OUTPUT_DIR, BATCH_SIZE)
