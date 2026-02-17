import json
import os
import pickle
import sys
from pathlib import Path

import mysql.connector
import networkx as nx
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[1]
GRAPH_DIR = BASE_DIR / "graph"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)
load_dotenv(BASE_DIR / ".env")
sys.path.append(str(BASE_DIR))

from expert_discovery.utils.similarity import min_max_normalize


def get_db_config():
    return {
        "host": os.getenv("DB_HOST", "127.0.0.1"),
        "user": os.getenv("DB_USER", "root"),
        "password": os.getenv("DB_PASSWORD", ""),
        "port": int(os.getenv("DB_PORT", "3306")),
        "database": os.getenv("DB_NAME", "expert_discovery"),
    }


def main():
    config = get_db_config()
    conn = mysql.connector.connect(**config)
    cursor = conn.cursor()

    cursor.execute("SELECT user_id FROM users")
    users = [row[0] for row in cursor.fetchall()]

    cursor.execute("SELECT question_id FROM questions")
    questions = [row[0] for row in cursor.fetchall()]

    cursor.execute("SELECT tag_id, tag_name FROM tags")
    tags = cursor.fetchall()

    cursor.execute("SELECT question_id, tag_id FROM question_tags")
    question_tags = cursor.fetchall()

    cursor.execute("SELECT question_id, user_id, upvotes, accepted_flag FROM answers")
    answers = cursor.fetchall()

    graph = nx.DiGraph()

    for user_id in users:
        graph.add_node(f"user_{user_id}", node_type="user")
    for question_id in questions:
        graph.add_node(f"question_{question_id}", node_type="question")
    for tag_id, tag_name in tags:
        graph.add_node(f"tag_{tag_id}", node_type="tag", tag_name=tag_name)

    for question_id, tag_id in question_tags:
        graph.add_edge(f"question_{question_id}", f"tag_{tag_id}", edge_type="has_tag")

    for question_id, user_id, upvotes, accepted_flag in answers:
        weight = 1 + upvotes + (5 if accepted_flag else 0)
        graph.add_edge(
            f"user_{user_id}",
            f"question_{question_id}",
            edge_type="answered",
            weight=weight,
        )

    user_tag_weights = {}
    question_to_tags = {}
    for question_id, tag_id in question_tags:
        question_to_tags.setdefault(question_id, []).append(tag_id)

    for question_id, user_id, _, _ in answers:
        for tag_id in question_to_tags.get(question_id, []):
            key = (user_id, tag_id)
            user_tag_weights[key] = user_tag_weights.get(key, 0) + 1

    for (user_id, tag_id), weight in user_tag_weights.items():
        graph.add_edge(
            f"user_{user_id}",
            f"tag_{tag_id}",
            edge_type="expertise",
            weight=weight,
        )

    pagerank = nx.pagerank(graph, weight="weight")
    degree = nx.degree_centrality(graph.to_undirected())
    betweenness = nx.betweenness_centrality(graph.to_undirected())

    user_scores = {}
    for user_id in users:
        node_key = f"user_{user_id}"
        user_scores[user_id] = {
            "pagerank": pagerank.get(node_key, 0.0),
            "degree": degree.get(node_key, 0.0),
            "betweenness": betweenness.get(node_key, 0.0),
        }

    normalized_pr = min_max_normalize({uid: val["pagerank"] for uid, val in user_scores.items()})
    normalized_deg = min_max_normalize({uid: val["degree"] for uid, val in user_scores.items()})
    normalized_bet = min_max_normalize({uid: val["betweenness"] for uid, val in user_scores.items()})

    for user_id, values in user_scores.items():
        graph_score = (
            normalized_pr.get(user_id, 0.0)
            + normalized_deg.get(user_id, 0.0)
            + normalized_bet.get(user_id, 0.0)
        ) / 3
        values["graph_score"] = graph_score

    with open(GRAPH_DIR / "graph_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(user_scores, handle, indent=2)

    with open(GRAPH_DIR / "graph.gpickle", "wb") as handle:
        pickle.dump(graph, handle, protocol=pickle.HIGHEST_PROTOCOL)

    cursor.close()
    conn.close()
    print("Graph metrics generated.")


if __name__ == "__main__":
    main()
