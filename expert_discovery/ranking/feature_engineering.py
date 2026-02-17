import json
import os
import sys
from pathlib import Path

import mysql.connector
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[1]
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
GRAPH_DIR = BASE_DIR / "graph"
load_dotenv(BASE_DIR / ".env")
sys.path.append(str(BASE_DIR))

from expert_discovery.utils.similarity import (
    cosine_similarity,
    jaccard_similarity,
    min_max_normalize,
    weighted_score,
)


def get_db_config():
    return {
        "host": os.getenv("DB_HOST", "127.0.0.1"),
        "user": os.getenv("DB_USER", "root"),
        "password": os.getenv("DB_PASSWORD", ""),
        "port": int(os.getenv("DB_PORT", "3306")),
        "database": os.getenv("DB_NAME", "expert_discovery"),
    }


def load_embeddings():
    with open(EMBEDDINGS_DIR / "embedding_index.json", "r", encoding="utf-8") as handle:
        index = json.load(handle)
    question_embeddings = np.load(EMBEDDINGS_DIR / "question_embeddings.npy")
    answer_embeddings = np.load(EMBEDDINGS_DIR / "answer_embeddings.npy")
    user_embeddings = np.load(EMBEDDINGS_DIR / "user_embeddings.npy")
    return index, question_embeddings, answer_embeddings, user_embeddings


def load_graph_scores():
    graph_path = GRAPH_DIR / "graph_metrics.json"
    if not graph_path.exists():
        return {}
    with open(graph_path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {int(user_id): metrics.get("graph_score", 0.0) for user_id, metrics in raw.items()}


def fetch_user_stats(cursor):
    cursor.execute("SELECT user_id, reputation, accepted_answers, total_answers, activity_score FROM users")
    return cursor.fetchall()


def fetch_user_upvotes(cursor):
    cursor.execute("SELECT user_id, SUM(upvotes) FROM answers GROUP BY user_id")
    return {row[0]: row[1] or 0 for row in cursor.fetchall()}


def fetch_user_tags(cursor):
    cursor.execute(
        "SELECT a.user_id, t.tag_name "
        "FROM answers a "
        "JOIN question_tags qt ON a.question_id = qt.question_id "
        "JOIN tags t ON qt.tag_id = t.tag_id"
    )
    tag_map = {}
    for user_id, tag_name in cursor.fetchall():
        tag_map.setdefault(user_id, {}).setdefault(tag_name, 0)
        tag_map[user_id][tag_name] += 1
    return tag_map


def build_user_features(conn):
    cursor = conn.cursor()
    user_stats = fetch_user_stats(cursor)
    upvotes = fetch_user_upvotes(cursor)
    user_tags = fetch_user_tags(cursor)

    reputation = {row[0]: float(row[1]) for row in user_stats}
    accepted = {row[0]: float(row[2]) for row in user_stats}
    total_answers = {row[0]: float(row[3]) for row in user_stats}
    activity = {row[0]: float(row[4]) for row in user_stats}
    upvote_scores = {uid: float(upvotes.get(uid, 0)) for uid in reputation}

    norm_reputation = min_max_normalize(reputation)
    norm_accepted = min_max_normalize(accepted)
    norm_total = min_max_normalize(total_answers)
    norm_upvotes = min_max_normalize(upvote_scores)
    norm_activity = min_max_normalize(activity)

    expertise = {
        uid: (norm_reputation[uid] + norm_accepted[uid] + norm_total[uid] + norm_upvotes[uid]) / 4
        for uid in reputation
    }

    cursor.close()
    return {
        "expertise": expertise,
        "activity": norm_activity,
        "user_tags": user_tags,
        "reputation": reputation,
    }


def get_user_embeddings_map(index, user_embeddings):
    return {uid: user_embeddings[idx] for idx, uid in enumerate(index["user_ids"])}


def rank_users(question_text, question_tags, top_k=3):
    config = get_db_config()
    conn = mysql.connector.connect(**config)

    index, _, _, user_embeddings = load_embeddings()
    user_embeddings_map = get_user_embeddings_map(index, user_embeddings)
    graph_scores = load_graph_scores()

    model = SentenceTransformer("all-MiniLM-L6-v2")
    question_embedding = model.encode([question_text], normalize_embeddings=True)[0]

    features = build_user_features(conn)

    weights = {
        "semantic": 0.4,
        "tag": 0.3,
        "expertise": 0.2,
        "activity": 0.1,
        "graph": 0.1,
    }

    results = []
    for user_id, embedding in user_embeddings_map.items():
        semantic = cosine_similarity(question_embedding, embedding)
        user_tag_set = set(features["user_tags"].get(user_id, {}).keys())
        tag_sim = jaccard_similarity(set(question_tags), user_tag_set)
        expertise_score = features["expertise"].get(user_id, 0.0)
        activity_score = features["activity"].get(user_id, 0.0)
        graph_score = graph_scores.get(user_id, 0.0)

        score = weighted_score(
            {
                "semantic": semantic,
                "tag": tag_sim,
                "expertise": expertise_score,
                "activity": activity_score,
                "graph": graph_score,
            },
            weights,
        )

        results.append(
            {
                "user_id": user_id,
                "score": score,
                "semantic": semantic,
                "tag_similarity": tag_sim,
                "expertise": expertise_score,
                "activity": activity_score,
                "graph_score": graph_score,
            }
        )

    results.sort(key=lambda x: x["score"], reverse=True)
    conn.close()
    return results[:top_k], features


def build_training_data(negatives_per_positive=3):
    config = get_db_config()
    conn = mysql.connector.connect(**config)
    cursor = conn.cursor()

    index, question_embeddings, _, user_embeddings = load_embeddings()
    question_id_to_idx = {qid: idx for idx, qid in enumerate(index["question_ids"])}
    user_embeddings_map = get_user_embeddings_map(index, user_embeddings)
    graph_scores = load_graph_scores()

    cursor.execute("SELECT question_id, title, body FROM questions")
    questions = cursor.fetchall()

    cursor.execute("SELECT question_id, user_id FROM answers")
    answer_pairs = cursor.fetchall()
    question_to_users = {}
    for question_id, user_id in answer_pairs:
        question_to_users.setdefault(question_id, set()).add(user_id)

    cursor.execute("SELECT question_id, tag_name FROM question_tags qt JOIN tags t ON qt.tag_id = t.tag_id")
    question_tags = {}
    for question_id, tag_name in cursor.fetchall():
        question_tags.setdefault(question_id, set()).add(tag_name)

    features = build_user_features(conn)

    rows = []
    for question_id, title, body in questions:
        pos_users = question_to_users.get(question_id, set())
        if not pos_users:
            continue
        all_users = set(user_embeddings_map.keys())
        neg_users = list(all_users - pos_users)
        sampled_neg = random_sample(neg_users, negatives_per_positive * len(pos_users))
        selected_users = list(pos_users) + sampled_neg

        q_idx = question_id_to_idx.get(question_id)
        if q_idx is None:
            continue
        q_emb = question_embeddings[q_idx]
        tags = question_tags.get(question_id, set())

        for user_id in selected_users:
            semantic = cosine_similarity(q_emb, user_embeddings_map[user_id])
            user_tag_set = set(features["user_tags"].get(user_id, {}).keys())
            tag_sim = jaccard_similarity(tags, user_tag_set)
            expertise_score = features["expertise"].get(user_id, 0.0)
            activity_score = features["activity"].get(user_id, 0.0)
            graph_score = graph_scores.get(user_id, 0.0)
            label = 1 if user_id in pos_users else 0

            rows.append(
                {
                    "question_id": question_id,
                    "user_id": user_id,
                    "semantic": semantic,
                    "tag_similarity": tag_sim,
                    "expertise": expertise_score,
                    "activity": activity_score,
                    "graph_score": graph_score,
                    "label": label,
                }
            )

    cursor.close()
    conn.close()
    return rows


def random_sample(items, count):
    if count <= 0:
        return []
    if len(items) <= count:
        return items
    return list(np.random.choice(items, size=count, replace=False))
