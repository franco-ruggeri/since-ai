import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from langchain_featherless_ai import ChatFeatherlessAi
import os
from dotenv import load_dotenv

from env_vars import ENV_FEATHERLESS_API_KEY


class SemanticClusterer:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.cluster_labels = None
        self.cluster_names = None

    def _embed(self, texts, show_progress=True):
        texts_list = texts.tolist() if isinstance(texts, pd.Series) else texts
        texts_list = [str(t).strip() if pd.notna(t) else "" for t in texts_list]
        return self.model.encode(
            texts_list,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

    def _cluster(self, embeddings, max_clusters=50):
        print("Finding optimal number of clusters...")
        silhouette_scores = []
        elbow_scores = []
        best_k = 2
        best_score = -1

        for k in range(2, min(max_clusters + 1, len(embeddings))):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)
            score = silhouette_score(embeddings, labels)
            silhouette_scores.append(score)
            elbow_scores.append(kmeans.inertia_)

            if score > best_score:
                best_score = score
                best_k = k

        plt.figure()
        plt.plot(range(2, max_clusters), silhouette_scores)
        plt.figure()
        plt.plot(range(2, max_clusters), elbow_scores)
        plt.show()

        n_clusters = best_k
        print(f"Optimal K: {n_clusters} (silhouette score: {best_score:.3f})")

        print(f"Running K-Means with {n_clusters} clusters...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        return cluster_labels

    def _label_cluster(self, sample_texts):
        load_dotenv()

        api_key = ENV_FEATHERLESS_API_KEY
        api_url = "https://api.featherless.ai/v1"
        model = "Qwen/Qwen2.5-Coder-32B-Instruct"

        llm = ChatFeatherlessAi(
            api_key=api_key,
            base_url=api_url,
        )

        system_prompt = """
You are an expert at summarizing and labeling clusters of text data.
Given a list of texts, provide a short, descriptive label that best represents the main topic or theme of the cluster.
"""
        user_prompt = (
            "Here are some example texts from a cluster:\n"
            + "\n".join(f"- {t}" for t in sample_texts)
            + "\n\nWhat is a concise label for this cluster?"
        )
        messages = [
            ("system", system_prompt),
            ("human", user_prompt),
        ]

        response = llm.invoke(
            messages,
            model=model,
            temperature=0.3,
            seed=42,
        ).content

        return response.strip()

    def _label(self, texts, cluster_ids):
        texts_array = np.asarray(texts)
        cluster_names = {}

        unique_cluster_ids = set(cluster_ids)
        if -1 in unique_cluster_ids:
            cluster_names[-1] = "Others"
            unique_cluster_ids.remove(-1)

        print("Generating LLM-based cluster labels...")
        for cluster_id in sorted(unique_cluster_ids):
            mask = cluster_ids == cluster_id
            cluster_texts = texts_array[mask]

            # Sample up to 10 representative texts
            # TODO: pick the most central texts instead of random
            sample_size = min(10, len(cluster_texts))
            sample_texts = np.random.choice(cluster_texts, sample_size, replace=False)

            # Get label from LLM
            label = self._label_cluster(sample_texts.tolist())
            cluster_names[cluster_id] = label
            print(f"Cluster {cluster_id}: {label}")

        return [cluster_names[cluster_id] for cluster_id in cluster_ids]

    def fit_predict(self, texts):
        embeddings = self._embed(texts)
        cluster_ids = self._cluster(embeddings)
        cluster_names = self._label(texts, cluster_ids)
        return cluster_names
