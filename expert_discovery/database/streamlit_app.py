import json
import os
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mysql.connector
import networkx as nx
import pandas as pd
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parents[1]
GRAPH_PATH = BASE_DIR / "graph" / "graph.gpickle"
load_dotenv(BASE_DIR / ".env")
sys.path.append(str(BASE_DIR.parent))

from expert_discovery.ranking.feature_engineering import rank_users


def get_db_config():
    return {
        "host": os.getenv("DB_HOST", "127.0.0.1"),
        "user": os.getenv("DB_USER", "root"),
        "password": os.getenv("DB_PASSWORD", ""),
        "port": int(os.getenv("DB_PORT", "3306")),
        "database": os.getenv("DB_NAME", "expert_discovery"),
    }


def fetch_user_profiles(user_ids):
    if not user_ids:
        return {}
    config = get_db_config()
    conn = mysql.connector.connect(**config)
    cursor = conn.cursor()
    format_ids = ",".join([str(uid) for uid in user_ids])
    cursor.execute(
        f"SELECT user_id, username, reputation, activity_score FROM users WHERE user_id IN ({format_ids})"
    )
    profiles = {row[0]: row[1:] for row in cursor.fetchall()}
    cursor.close()
    conn.close()
    return profiles


@st.cache_data(show_spinner=False)
def load_question_embeddings():
    embeddings_dir = BASE_DIR / "embeddings"
    index_path = embeddings_dir / "embedding_index.json"
    questions_path = embeddings_dir / "question_embeddings.npy"
    if not index_path.exists() or not questions_path.exists():
        return None, None
    with open(index_path, "r", encoding="utf-8") as handle:
        index = json.load(handle)
    question_embeddings = np.load(questions_path)
    return index, question_embeddings


@st.cache_data(show_spinner=False)
def fetch_question_metadata():
    config = get_db_config()
    conn = mysql.connector.connect(**config)
    cursor = conn.cursor()
    cursor.execute("SELECT question_id, title FROM questions")
    question_rows = cursor.fetchall()
    cursor.execute(
        "SELECT qt.question_id, t.tag_name "
        "FROM question_tags qt "
        "JOIN tags t ON qt.tag_id = t.tag_id"
    )
    tag_rows = cursor.fetchall()
    cursor.close()
    conn.close()

    question_map = {row[0]: row[1] for row in question_rows}
    tag_map = {}
    for question_id, tag_name in tag_rows:
        tag_map.setdefault(question_id, []).append(tag_name)
    return question_map, tag_map


def fetch_top_answers(question_ids, limit_per_question=2):
    if not question_ids:
        return {}
    config = get_db_config()
    conn = mysql.connector.connect(**config)
    cursor = conn.cursor()
    format_ids = ",".join([str(qid) for qid in question_ids])
    cursor.execute(
        "SELECT a.question_id, a.answer_text, a.upvotes, a.accepted_flag, u.username "
        "FROM answers a "
        "JOIN users u ON a.user_id = u.user_id "
        f"WHERE a.question_id IN ({format_ids}) "
        "ORDER BY a.question_id, a.accepted_flag DESC, a.upvotes DESC"
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    answer_map = {}
    for question_id, answer_text, upvotes, accepted_flag, username in rows:
        answer_map.setdefault(question_id, [])
        if len(answer_map[question_id]) >= limit_per_question:
            continue
        answer_map[question_id].append(
            {
                "answer_text": answer_text,
                "upvotes": upvotes,
                "accepted": bool(accepted_flag),
                "username": username,
            }
        )
    return answer_map


@st.cache_data(show_spinner=False)
def fetch_activity_timeline(user_ids):
    if not user_ids:
        return pd.DataFrame()
    config = get_db_config()
    conn = mysql.connector.connect(**config)
    cursor = conn.cursor()
    format_ids = ",".join([str(uid) for uid in user_ids])
    cursor.execute(
        "SELECT a.user_id, DATE(q.timestamp) AS day, COUNT(*) "
        "FROM answers a "
        "JOIN questions q ON a.question_id = q.question_id "
        f"WHERE a.user_id IN ({format_ids}) "
        "GROUP BY a.user_id, day "
        "ORDER BY day"
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=["user_id", "day", "count"])


@st.cache_data(show_spinner=False)
def load_graph_metrics():
    metrics_path = BASE_DIR / "graph" / "graph_metrics.json"
    if not metrics_path.exists():
        return {}
    with open(metrics_path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {int(user_id): metrics for user_id, metrics in raw.items()}


def render_tag_heatmap(user_ids, features):
    tag_counts = {}
    for user_id in user_ids:
        user_tags = features["user_tags"].get(user_id, {})
        for tag, count in user_tags.items():
            tag_counts[tag] = tag_counts.get(tag, 0) + count

    top_tags = [tag for tag, _ in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]]
    if not top_tags:
        st.info("No tag data available for selected experts.")
        return

    matrix = []
    labels = []
    for user_id in user_ids:
        labels.append(f"user_{user_id}")
        user_tags = features["user_tags"].get(user_id, {})
        matrix.append([user_tags.get(tag, 0) for tag in top_tags])

    data = np.array(matrix, dtype=float)
    if data.max() > 0:
        data = data / data.max()

    fig, ax = plt.subplots(figsize=(8, 3 + 0.3 * len(user_ids)))
    im = ax.imshow(data, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(top_tags)))
    ax.set_xticklabels(top_tags, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Tag Overlap Heatmap (Normalized)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig)


def render_influence_radar(user_ids):
    metrics = load_graph_metrics()
    if not metrics:
        st.info("Graph metrics not found. Run graph/build_graph.py first.")
        return
    if not user_ids:
        st.info("No experts selected.")
        return

    user_id = user_ids[0]
    user_metrics = metrics.get(user_id, {})
    if not user_metrics:
        st.info("Graph metrics missing for selected expert.")
        return

    labels = ["pagerank", "degree", "betweenness", "graph_score"]
    values = [
        float(user_metrics.get("pagerank", 0.0)),
        float(user_metrics.get("degree", 0.0)),
        float(user_metrics.get("betweenness", 0.0)),
        float(user_metrics.get("graph_score", 0.0)),
    ]

    max_val = max(values) if max(values) > 0 else 1.0
    values = [val / max_val for val in values]
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(5, 5))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values, color="#ff7f0e", linewidth=2)
    ax.fill(angles, values, color="#ff7f0e", alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title(f"Influence Radar: user_{user_id}")
    ax.set_ylim(0, 1)
    st.pyplot(fig)


def render_graph(user_ids):
    if not GRAPH_PATH.exists():
        st.info("Graph file not found. Run graph/build_graph.py first.")
        return
    with open(GRAPH_PATH, "rb") as handle:
        graph = pickle.load(handle)
    nodes = [f"user_{uid}" for uid in user_ids]
    related_nodes = set(nodes)
    for node in nodes:
        related_nodes.update(graph.successors(node))
        related_nodes.update(graph.predecessors(node))
    subgraph = graph.subgraph(related_nodes)

    node_colors = []
    for node in subgraph.nodes:
        node_type = subgraph.nodes[node].get("node_type", "other")
        if node_type == "user":
            node_colors.append("#1f77b4")
        elif node_type == "question":
            node_colors.append("#ff7f0e")
        elif node_type == "tag":
            node_colors.append("#2ca02c")
        else:
            node_colors.append("#7f7f7f")

    plt.figure(figsize=(9, 6))
    pos = nx.spring_layout(subgraph, k=0.7, seed=7)
    nx.draw_networkx_nodes(subgraph, pos, node_size=420, node_color=node_colors)
    nx.draw_networkx_edges(subgraph, pos, alpha=0.35, width=1.2)
    nx.draw_networkx_labels(subgraph, pos, font_size=8)
    plt.axis("off")
    st.pyplot(plt.gcf())


def main():
    st.set_page_config(page_title="Expert Discovery", layout="wide")
    st.title("Expert Discovery")
    st.caption("Semantic matching + graph influence for community Q&A expert search.")

    with st.sidebar:
        st.header("Query")
        question_text = st.text_area("Question", height=140)
        tags_input = st.text_input("Tags (comma-separated)")
        top_k = st.slider("Top K", min_value=3, max_value=10, value=3)
        similar_k = st.slider("Similar Questions", min_value=3, max_value=10, value=5)
        run_btn = st.button("Find Experts")

    if run_btn:
        if not question_text.strip():
            st.warning("Enter a question.")
            return

        question_tags = [tag.strip() for tag in tags_input.split(",") if tag.strip()]
        results, features = rank_users(question_text, question_tags, top_k=top_k)

        user_ids = [entry["user_id"] for entry in results]
        profiles = fetch_user_profiles(user_ids)

        output_rows = []
        tag_summary = {}
        for entry in results:
            user_id = entry["user_id"]
            username, reputation, activity_score = profiles.get(user_id, ("unknown", 0, 0))
            tag_dict = features["user_tags"].get(user_id, {})
            sorted_tags = sorted(tag_dict, key=tag_dict.get, reverse=True)
            top_tags = ", ".join(sorted_tags[:5])
            for tag in sorted_tags[:5]:
                tag_summary[tag] = tag_summary.get(tag, 0) + 1
            output_rows.append(
                {
                    "Username": username,
                    "Match Score %": round(entry["score"] * 100, 2),
                    "Semantic": round(entry["semantic"] * 100, 2),
                    "Tag Similarity": round(entry["tag_similarity"] * 100, 2),
                    "Expertise": round(entry["expertise"] * 100, 2),
                    "Expertise Tags": top_tags,
                    "Reputation": reputation,
                    "Graph Influence": round(entry["graph_score"] * 100, 2),
                }
            )

        df = pd.DataFrame(output_rows)
        left, right = st.columns([1.2, 1])

        with left:
            st.subheader("Top Experts")
            st.dataframe(df, use_container_width=True)

            st.subheader("Score Breakdown")
            score_df = df[["Username", "Match Score %", "Semantic", "Tag Similarity", "Expertise", "Graph Influence"]]
            score_df = score_df.set_index("Username")
            st.bar_chart(score_df)

        with right:
            st.subheader("Top Tags Across Experts")
            if tag_summary:
                tag_df = (
                    pd.DataFrame(
                        [{"Tag": tag, "Count": count} for tag, count in tag_summary.items()]
                    )
                    .sort_values("Count", ascending=False)
                    .head(10)
                )
                st.bar_chart(tag_df.set_index("Tag"))
            else:
                st.info("No tags found for selected experts.")

            st.subheader("Graph Connections")
            render_graph(user_ids)

        st.subheader("Most Similar Questions")
        index, question_embeddings = load_question_embeddings()
        if index is None or question_embeddings is None:
            st.info("Question embeddings not found. Run embeddings/generate_embeddings.py first.")
        else:
            model = SentenceTransformer("all-MiniLM-L6-v2")
            query_embedding = model.encode([question_text], normalize_embeddings=True)[0]
            scores = np.dot(question_embeddings, query_embedding)
            top_indices = np.argsort(scores)[-similar_k:][::-1]
            question_map, tag_map = fetch_question_metadata()
            similar_rows = []
            similar_ids = []
            for idx in top_indices:
                question_id = index["question_ids"][idx]
                similar_ids.append(question_id)
                title = question_map.get(question_id, "Unknown")
                tags = ", ".join(tag_map.get(question_id, []))
                similar_rows.append(
                    {
                        "Question ID": question_id,
                        "Title": title,
                        "Similarity %": round(scores[idx] * 100, 2),
                        "Tags": tags,
                    }
                )
            st.dataframe(similar_rows, use_container_width=True)

            st.subheader("Top Answers From Similar Questions")
            answer_map = fetch_top_answers(similar_ids, limit_per_question=2)
            if not answer_map:
                st.info("No answers found for similar questions.")
            else:
                for row in similar_rows:
                    question_id = row["Question ID"]
                    answers = answer_map.get(question_id, [])
                    if not answers:
                        continue
                    title = row["Title"]
                    with st.expander(f"Q{question_id}: {title}"):
                        for answer in answers:
                            status = "accepted" if answer["accepted"] else "answer"
                            st.markdown(
                                f"**{status}** by {answer['username']} (upvotes: {answer['upvotes']})"
                            )
                            st.write(answer["answer_text"])

        st.subheader("Additional Insights")
        tab1, tab2, tab3 = st.tabs(["Activity Timeline", "Tag Overlap", "Influence Radar"])

        with tab1:
            timeline = fetch_activity_timeline(user_ids)
            if timeline.empty:
                st.info("No activity timeline data available.")
            else:
                timeline["user_label"] = timeline["user_id"].apply(lambda uid: f"user_{uid}")
                pivot = timeline.pivot_table(
                    index="day", columns="user_label", values="count", fill_value=0
                ).sort_index()
                st.line_chart(pivot)

        with tab2:
            render_tag_heatmap(user_ids, features)

        with tab3:
            render_influence_radar(user_ids)


if __name__ == "__main__":
    main()
