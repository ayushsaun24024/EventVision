import os
import json
import spacy
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict


def load_embeddings(embeddings_path, article_ids_path):
    embeddings = np.load(embeddings_path)
    article_ids = np.load(article_ids_path, allow_pickle=True).tolist()
    return embeddings, article_ids


def extract_entities(text, nlp):
    doc = nlp(text)
    entities = set()
    for ent in doc.ents:
        if ent.label_ in ['PERSON', 'ORG', 'GPE', 'LOC', 'EVENT']:
            entities.add(ent.text.lower().strip())
    return entities


def load_article_entities(json_path, article_ids, batch_size=100):
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'lemmatizer'])
    
    with open(json_path, 'r', encoding='utf-8') as f:
        database = json.load(f)
    
    entities_dict = {}
    
    texts = []
    aids = []
    
    for aid in article_ids:
        if aid in database:
            title = database[aid].get('title', '')
            content = database[aid].get('content', '')
            text = f"{title} {content}"
            texts.append(text[:1500])
            aids.append(aid)
        else:
            entities_dict[aid] = set()
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting entities"):
        batch_texts = texts[i:i+batch_size]
        batch_aids = aids[i:i+batch_size]
        
        docs = nlp.pipe(batch_texts, batch_size=batch_size)
        
        for aid, doc in zip(batch_aids, docs):
            entities = set()
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'LOC', 'EVENT']:
                    entities.add(ent.text.lower().strip())
            entities_dict[aid] = entities
    
    return entities_dict


def jaccard_similarity(set1, set2):
    if len(set1) == 0 and len(set2) == 0:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0


def compute_entity_edges(article_ids, entities_dict, threshold, batch_size=1000):
    edge_list = []
    num_articles = len(article_ids)
    
    entity_to_articles = defaultdict(list)
    for idx, aid in enumerate(article_ids):
        for entity in entities_dict.get(aid, set()):
            entity_to_articles[entity].append(idx)
    
    candidate_pairs = set()
    for entity, article_indices in entity_to_articles.items():
        if len(article_indices) > 1:
            article_indices = article_indices[:50]
            for i in range(len(article_indices)):
                for j in range(i + 1, len(article_indices)):
                    idx_i = article_indices[i]
                    idx_j = article_indices[j]
                    if idx_i < idx_j:
                        candidate_pairs.add((idx_i, idx_j))
                    else:
                        candidate_pairs.add((idx_j, idx_i))
    
    candidate_pairs = sorted(list(candidate_pairs))
    
    for i, j in tqdm(candidate_pairs, desc="Computing entity similarities"):
        aid_i = article_ids[i]
        aid_j = article_ids[j]
        
        entities_i = entities_dict.get(aid_i, set())
        entities_j = entities_dict.get(aid_j, set())
        
        similarity = jaccard_similarity(entities_i, entities_j)
        
        if similarity > threshold:
            edge_list.append((i, j, float(similarity)))
    
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
    ax.hist(degree_values, bins=50, color='#f57f17', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Node Degree', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Degree Distribution - Entity Graph', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'degree_distribution_entity.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(edge_weights, bins=50, color='#00838f', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Jaccard Similarity', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Similarity Distribution - Entity Graph', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'similarity_distribution_entity.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    metrics1 = ['num_nodes', 'num_edges']
    values1 = [stats[m] for m in metrics1]
    ax1.bar(metrics1, values1, color=['#f57f17', '#00838f'], alpha=0.7, edgecolor='black')
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
    plt.savefig(os.path.join(output_dir, 'graph_statistics_entity.png'), dpi=300, bbox_inches='tight')
    plt.close()


def build_entity_graph(embeddings_path, article_ids_path, json_path, output_dir, similarity_threshold):
    os.makedirs(output_dir, exist_ok=True)
    
    embeddings, article_ids = load_embeddings(embeddings_path, article_ids_path)
    
    entities_dict = load_article_entities(json_path, article_ids, batch_size=8)
    
    edge_list = compute_entity_edges(article_ids, entities_dict, similarity_threshold, batch_size=1000)
    
    stats, degree_values, edge_weights = calculate_graph_statistics(edge_list, len(article_ids))
    
    save_graph_structure(
        edge_list,
        embeddings,
        stats['num_nodes'],
        stats['num_edges'],
        os.path.join(output_dir, 'graph_entity.pkl')
    )
    
    save_statistics_csv(stats, os.path.join(output_dir, 'graph_statistics_entity.csv'))
    
    save_node_mapping(article_ids, os.path.join(output_dir, 'node_mapping_entity.json'))
    
    generate_plots(degree_values, edge_weights, stats, output_dir)
    
    print(f"Entity graph built successfully")
    print(f"Nodes: {stats['num_nodes']}, Edges: {stats['num_edges']}, Avg Degree: {stats['avg_degree']:.2f}")


if __name__ == "__main__":
    EMBEDDINGS_FILE = "./eventa_embeddings_Qwen3/database_embeddings_Qwen3.npy"
    ARTICLE_IDS_FILE = "./eventa_embeddings_Qwen3/database_article_ids_Qwen3.npy"
    DATASET_JSON = "./Dataset/database.json"
    OUTPUT_DIR = "./entity_graphs_output"
    SIMILARITY_THRESHOLD = 0.4
    
    build_entity_graph(
        EMBEDDINGS_FILE,
        ARTICLE_IDS_FILE,
        DATASET_JSON,
        OUTPUT_DIR,
        SIMILARITY_THRESHOLD
    )
"2296303"