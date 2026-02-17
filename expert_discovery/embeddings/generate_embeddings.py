import json
import os
import sys
from pathlib import Path

import mysql.connector
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[1]
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
load_dotenv(BASE_DIR / ".env")
sys.path.append(str(BASE_DIR))


def get_db_config():
    return {
        "host": os.getenv("DB_HOST", "127.0.0.1"),
        "user": os.getenv("DB_USER", "root"),
        "password": os.getenv("DB_PASSWORD", ""),
        "port": int(os.getenv("DB_PORT", "3306")),
        "database": os.getenv("DB_NAME", "expert_discovery"),
    }


def fetch_rows(cursor, query):
    cursor.execute(query)
    return cursor.fetchall()


def encode_texts(model, texts, batch_size=64):
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def main():
    config = get_db_config()
    conn = mysql.connector.connect(**config)
    cursor = conn.cursor()

    questions = fetch_rows(cursor, "SELECT question_id, title, body FROM questions")
    answers = fetch_rows(cursor, "SELECT answer_id, user_id, answer_text FROM answers")
    users = fetch_rows(cursor, "SELECT user_id FROM users")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    question_ids = [row[0] for row in questions]
    question_texts = [f"{row[1]}\n{row[2]}" for row in questions]
    question_embeddings = encode_texts(model, question_texts)

    answer_ids = [row[0] for row in answers]
    answer_user_ids = [row[1] for row in answers]
    answer_texts = [row[2] for row in answers]
    answer_embeddings = encode_texts(model, answer_texts)

    user_ids = [row[0] for row in users]
    user_vectors = {uid: [] for uid in user_ids}
    for idx, user_id in enumerate(answer_user_ids):
        user_vectors[user_id].append(answer_embeddings[idx])

    user_embeddings = []
    for user_id in user_ids:
        vectors = user_vectors.get(user_id) or []
        if vectors:
            user_embeddings.append(np.mean(vectors, axis=0))
        else:
            user_embeddings.append(np.zeros(answer_embeddings.shape[1], dtype=np.float32))

    np.save(EMBEDDINGS_DIR / "question_embeddings.npy", question_embeddings.astype(np.float32))
    np.save(EMBEDDINGS_DIR / "answer_embeddings.npy", answer_embeddings.astype(np.float32))
    np.save(EMBEDDINGS_DIR / "user_embeddings.npy", np.vstack(user_embeddings).astype(np.float32))

    index = {
        "question_ids": question_ids,
        "answer_ids": answer_ids,
        "answer_user_ids": answer_user_ids,
        "user_ids": user_ids,
    }
    with open(EMBEDDINGS_DIR / "embedding_index.json", "w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2)

    cursor.close()
    conn.close()
    print("Embeddings generated.")


if __name__ == "__main__":
    main()
