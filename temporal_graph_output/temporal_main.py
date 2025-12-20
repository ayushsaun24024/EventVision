import os
import json
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from datetime import datetime, timezone
from dateutil import parser
from typing import List, Tuple

EMBEDDINGS_FILE = "./eventa_embeddings_Qwen3/database_embeddings_Qwen3.npy"
ARTICLE_IDS_FILE = "./eventa_embeddings_Qwen3/database_article_ids_Qwen3.npy"
DATASET_JSON = "./Dataset/database.json"
OUTPUT_DIR = "./temporal_graphs_output"

TIME_WINDOW_DAYS = 1
MAX_EDGES_PER_NODE = 50
MIN_TEMPORAL_WEIGHT = 0.5
VERBOSE = True

def load_embeddings(embeddings_path: str, article_ids_path: str):
    embeddings = np.load(embeddings_path)
    article_ids = np.load(article_ids_path, allow_pickle=True).tolist()
    return embeddings, article_ids

def load_article_dates(json_path: str, article_ids: List[str]):
    with open(json_path, 'r', encoding='utf-8') as f:
        database = json.load(f)
    dates = {}
    invalid_count = 0
    for aid in article_ids:
        if aid in database:
            date_str = database[aid].get('date', '')
            try:
                dt = parser.parse(date_str)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                else:
                    dt = dt.astimezone(timezone.utc)
                dates[aid] = dt
            except Exception:
                dates[aid] = None
                invalid_count += 1
        else:
            dates[aid] = None
            invalid_count += 1
    if VERBOSE:
        total = len(article_ids)
        valid = total - invalid_count
        print(f"Loaded dates: total={total}, valid={valid}, invalid={invalid_count}", flush=True)
    return dates

def compute_temporal_edges_sparse(
    article_ids: List[str],
    dates: dict,
    time_window_days: int = 1,
    max_edges_per_node: int = 50,
    min_weight: float = 0.5
) -> Tuple[List[Tuple[int,int,float]], List[str], List[int]]:
    articles_with_dates = [(aid, dates[aid]) for aid in article_ids if dates.get(aid) is not None]
    articles_with_dates.sort(key=lambda x: x[1])
    sorted_article_ids = [aid for aid, _ in articles_with_dates]
    num_nodes = len(sorted_article_ids)

    aid_to_original_idx = {aid: idx for idx, aid in enumerate(article_ids)}
    original_indices = [aid_to_original_idx[aid] for aid in sorted_article_ids]

    i_list, j_list, w_list = [], [], []

    n = num_nodes
    if n == 0:
        return [], sorted_article_ids, original_indices

    e = 1
    for i in tqdm(range(n), desc="Temporal edges (sparse)"):
        date_i = articles_with_dates[i][1]
        if e < i + 1:
            e = i + 1
        while e < n:
            days_diff = (articles_with_dates[e][1] - date_i).total_seconds() / 86400.0
            if days_diff <= time_window_days:
                e += 1
            else:
                break
        edges_added = 0
        for j in range(i + 1, e):
            if edges_added >= max_edges_per_node:
                break
            days_diff = (articles_with_dates[j][1] - date_i).total_seconds() / 86400.0
            weight = 1.0 / (days_diff + 1.0)
            if weight < min_weight:
                continue
            i_list.append(i)
            j_list.append(j)
            w_list.append(float(weight))
            edges_added += 1

    edge_list = list(zip(i_list, j_list, w_list))
    if VERBOSE:
        print(f"Computed undirected-edge representation (unique edges stored once): {len(edge_list)}", flush=True)
    return edge_list, sorted_article_ids, original_indices

def calculate_graph_statistics(edge_list: List[Tuple[int,int,float]], num_nodes: int):
    if num_nodes == 0:
        stats = {
            'num_nodes': 0,
            'num_edges': 0,
            'avg_degree': 0.0,
            'max_degree': 0,
            'min_degree': 0,
            'density': 0.0,
            'avg_weight': 0.0,
            'max_weight': 0.0,
            'min_weight': 0.0
        }
        return stats, [], []

    deg = np.zeros(num_nodes, dtype=int)
    edge_weights = []

    seen = set()
    unique_edges = []
    for i, j, w in edge_list:
        key = frozenset({i, j})
        if key in seen:
            continue
        seen.add(key)
        unique_edges.append((i, j, w))

    for i, j, weight in unique_edges:
        deg[i] += 1
        deg[j] += 1
        edge_weights.append(weight)

    degree_values = deg.tolist()
    num_edges = len(unique_edges)
    max_possible_edges = (num_nodes * (num_nodes - 1)) / 2
    density = float(num_edges / max_possible_edges) if max_possible_edges > 0 else 0.0

    stats = {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'avg_degree': float(np.mean(degree_values)) if len(degree_values) > 0 else 0.0,
        'max_degree': int(np.max(degree_values)) if len(degree_values) > 0 else 0,
        'min_degree': int(np.min(degree_values)) if len(degree_values) > 0 else 0,
        'density': density,
        'avg_weight': float(np.mean(edge_weights)) if edge_weights else 0.0,
        'max_weight': float(np.max(edge_weights)) if edge_weights else 0.0,
        'min_weight': float(np.min(edge_weights)) if edge_weights else 0.0
    }
    return stats, degree_values, edge_weights

def save_graph_structure(edge_list, node_features, num_nodes, num_edges, output_path):
    graph_data = {
        'edge_list': edge_list,
        'node_features': node_features,
        'num_nodes': num_nodes,
        'num_edges': num_edges
    }
    with open(output_path, 'wb') as f:
        pickle.dump(graph_data, f, protocol=pickle.HIGHEST_PROTOCOL)

def save_node_mapping(sorted_article_ids: List[str], original_indices: List[int], output_path: str):
    mapping = {
        'sorted_index_to_article_id': {str(i): aid for i, aid in enumerate(sorted_article_ids)},
        'sorted_index_to_original_idx': {str(i): int(orig) for i, orig in enumerate(original_indices)}
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2)

def save_statistics_csv(stats: dict, output_path: str):
    pd.DataFrame([stats]).to_csv(output_path, index=False)

def generate_basic_plots(degree_values, edge_weights, stats, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use('default')

    if degree_values:
        plt.figure(figsize=(8,5))
        bins = min(50, max(5, len(degree_values)//2))
        plt.hist(degree_values, bins=bins)
        plt.title('Degree distribution (temporal)')
        plt.xlabel('degree')
        plt.ylabel('count')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'degree_distribution_temporal.png'), dpi=150)
        plt.close()

    if edge_weights:
        plt.figure(figsize=(8,5))
        bins = min(50, max(5, len(edge_weights)//2))
        plt.hist(edge_weights, bins=bins)
        plt.title('Edge weight distribution (temporal)')
        plt.xlabel('weight')
        plt.ylabel('count')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'weight_distribution_temporal.png'), dpi=150)
        plt.close()

def build_temporal_graph(
    embeddings_path: str,
    article_ids_path: str,
    json_path: str,
    output_dir: str,
    time_window_days: int = TIME_WINDOW_DAYS,
    max_edges_per_node: int = MAX_EDGES_PER_NODE,
    min_weight: float = MIN_TEMPORAL_WEIGHT
):
    os.makedirs(output_dir, exist_ok=True)

    embeddings, article_ids = load_embeddings(embeddings_path, article_ids_path)
    dates = load_article_dates(json_path, article_ids)

    edge_list, sorted_article_ids, original_indices = compute_temporal_edges_sparse(
        article_ids,
        dates,
        time_window_days=time_window_days,
        max_edges_per_node=max_edges_per_node,
        min_weight=min_weight
    )

    num_nodes = len(sorted_article_ids)
    stats, degree_values, edge_weights = calculate_graph_statistics(edge_list, num_nodes)

    original_indices = np.array(original_indices, dtype=int) if len(original_indices) > 0 else np.array([], dtype=int)
    node_features = None
    if embeddings is not None and embeddings.size > 0 and original_indices.size > 0:
        if embeddings.shape[0] == len(article_ids):
            try:
                node_features = embeddings[original_indices]
            except Exception:
                node_features = np.zeros((num_nodes, embeddings.shape[1] if embeddings.ndim > 1 else 0))
        elif embeddings.shape[0] >= int(original_indices.max()) + 1:
            try:
                node_features = embeddings[original_indices]
            except Exception:
                node_features = np.zeros((num_nodes, embeddings.shape[1] if embeddings.ndim > 1 else 0))
        else:
            node_features = np.zeros((num_nodes, embeddings.shape[1] if embeddings.ndim > 1 else 0))
            if VERBOSE:
                print("Warning: embeddings rows do not align with article_ids; node_features set to zeros.", flush=True)
    else:
        dim = embeddings.shape[1] if (embeddings is not None and embeddings.ndim > 1) else 0
        node_features = np.zeros((num_nodes, dim))

    save_graph_structure(edge_list, node_features, num_nodes, stats['num_edges'], os.path.join(output_dir, 'graph_temporal.pkl'))
    save_node_mapping(sorted_article_ids, original_indices.tolist(), os.path.join(output_dir, 'node_mapping_temporal.json'))
    save_statistics_csv(stats, os.path.join(output_dir, 'graph_statistics_temporal.csv'))
    generate_basic_plots(degree_values, edge_weights, stats, output_dir)

    print("Temporal graph built successfully", flush=True)
    print(f"Nodes: {stats['num_nodes']}, Edges: {stats['num_edges']}, Avg Degree: {stats['avg_degree']:.2f}", flush=True)
    return stats

if __name__ == "__main__":
    build_temporal_graph(
        EMBEDDINGS_FILE,
        ARTICLE_IDS_FILE,
        DATASET_JSON,
        OUTPUT_DIR,
        time_window_days=TIME_WINDOW_DAYS,
        max_edges_per_node=MAX_EDGES_PER_NODE,
        min_weight=MIN_TEMPORAL_WEIGHT
    )
"4048595"