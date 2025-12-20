import os
import json
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

def load_embeddings(embeddings_path, article_ids_path):
    embeddings = np.load(embeddings_path)
    article_ids = np.load(article_ids_path, allow_pickle=True).tolist()
    return embeddings, article_ids


def normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / (norms + 1e-10)


def compute_similarity_batch(embeddings, batch_size, threshold):
    num_nodes = len(embeddings)
    edge_list = []
    
    for i in tqdm(range(0, num_nodes, batch_size), desc="Computing similarities"):
        end_i = min(i + batch_size, num_nodes)
        batch_emb = embeddings[i:end_i]
        
        similarities = np.dot(batch_emb, embeddings.T)
        
        for local_idx in range(len(batch_emb)):
            global_idx = i + local_idx
            row_sims = similarities[local_idx]
            
            valid_indices = np.where((row_sims > threshold) & (np.arange(num_nodes) > global_idx))[0]
            
            for j in valid_indices:
                edge_list.append((global_idx, int(j), float(row_sims[j])))
    
    return edge_list


def calculate_graph_statistics(edge_list, num_nodes):
    degrees = {}
    edge_weights = []
    
    for i, j, weight in edge_list:
        degrees[i] = degrees.get(i, 0) + 1
        degrees[j] = degrees.get(j, 0) + 1
        edge_weights.append(weight)
    
    degree_values = list(degrees.values())
    avg_degree = np.mean(degree_values) if degree_values else 0
    max_degree = max(degree_values) if degree_values else 0
    min_degree = min(degree_values) if degree_values else 0
    
    num_edges = len(edge_list)
    max_possible_edges = (num_nodes * (num_nodes - 1)) / 2
    density = num_edges / max_possible_edges if max_possible_edges > 0 else 0
    
    avg_similarity = np.mean(edge_weights) if edge_weights else 0
    max_similarity = max(edge_weights) if edge_weights else 0
    min_similarity = min(edge_weights) if edge_weights else 0
    
    stats = {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'avg_degree': avg_degree,
        'max_degree': max_degree,
        'min_degree': min_degree,
        'density': density,
        'avg_similarity': avg_similarity,
        'max_similarity': max_similarity,
        'min_similarity': min_similarity
    }
    
    return stats, degree_values, edge_weights


def save_graph_structure(edge_list, embeddings, num_nodes, num_edges, output_path):
    graph_data = {
        'edge_list': edge_list,
        'node_features': embeddings,
        'num_nodes': num_nodes,
        'num_edges': num_edges
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(graph_data, f)


def save_statistics_csv(stats, output_path):
    df = pd.DataFrame(list(stats.items()), columns=['metric', 'value'])
    df.to_csv(output_path, index=False)


def save_node_mapping(article_ids, output_path):
    mapping = {idx: aid for idx, aid in enumerate(article_ids)}
    with open(output_path, 'w') as f:
        json.dump(mapping, f, indent=2)


def generate_plots(degree_values, edge_weights, stats, output_dir):
    plt.style.use('default')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(degree_values, bins=50, color='#0277bd', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Node Degree', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Degree Distribution - Semantic Graph', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'degree_distribution_semantic.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(edge_weights, bins=50, color='#2e7d32', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Cosine Similarity', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Similarity Distribution - Semantic Graph', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'similarity_distribution_semantic.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    metrics1 = ['num_nodes', 'num_edges']
    values1 = [stats[m] for m in metrics1]
    ax1.bar(metrics1, values1, color=['#0277bd', '#2e7d32'], alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Node and Edge Counts', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    metrics2 = ['avg_degree', 'density']
    values2 = [stats['avg_degree'], stats['density'] * 1000]
    labels2 = ['Avg Degree', 'Density (×1000)']
    ax2.bar(labels2, values2, color=['#d84315', '#f57c00'], alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Value', fontsize=12)
    ax2.set_title('Graph Statistics', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'graph_statistics_semantic.png'), dpi=300, bbox_inches='tight')
    plt.close()


def build_semantic_graph(embeddings_path, article_ids_path, json_path, output_dir, similarity_threshold, batch_size):
    os.makedirs(output_dir, exist_ok=True)
    
    embeddings, article_ids = load_embeddings(embeddings_path, article_ids_path)
    
    embeddings_normalized = normalize_embeddings(embeddings)
    
    edge_list = compute_similarity_batch(embeddings_normalized, batch_size, similarity_threshold)
    
    stats, degree_values, edge_weights = calculate_graph_statistics(edge_list, len(article_ids))
    
    save_graph_structure(
        edge_list, 
        embeddings_normalized, 
        stats['num_nodes'], 
        stats['num_edges'],
        os.path.join(output_dir, 'graph_semantic.pkl')
    )
    
    save_statistics_csv(stats, os.path.join(output_dir, 'graph_statistics_semantic.csv'))
    
    save_node_mapping(article_ids, os.path.join(output_dir, 'node_mapping_semantic.json'))
    
    generate_plots(degree_values, edge_weights, stats, output_dir)
    
    print(f"Semantic graph built successfully")
    print(f"Nodes: {stats['num_nodes']}, Edges: {stats['num_edges']}, Avg Degree: {stats['avg_degree']:.2f}")


if __name__ == "__main__":
    EMBEDDINGS_FILE = "./eventa_embeddings_Qwen3/database_embeddings_Qwen3.npy"
    ARTICLE_IDS_FILE = "./eventa_embeddings_Qwen3/database_article_ids_Qwen3.npy"
    DATASET_JSON = "./Dataset/database.json"
    OUTPUT_DIR = "./semantic_graphs_output"
    SIMILARITY_THRESHOLD = 0.7
    BATCH_SIZE = 1000
    
    build_semantic_graph(
        EMBEDDINGS_FILE,
        ARTICLE_IDS_FILE,
        DATASET_JSON,
        OUTPUT_DIR,
        SIMILARITY_THRESHOLD,
        BATCH_SIZE
    )
"2265062"