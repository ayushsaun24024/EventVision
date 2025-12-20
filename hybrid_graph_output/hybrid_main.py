import os
import json
import pickle
from datetime import datetime
from typing import List, Tuple, Dict, Set, Optional
from collections import defaultdict
from datetime import timezone
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import spacy
from dateutil import parser

EMBEDDINGS_FILE = "./eventa_embeddings_Qwen3/database_embeddings_Qwen3.npy"
ARTICLE_IDS_FILE = "./eventa_embeddings_Qwen3/database_article_ids_Qwen3.npy"
DATASET_JSON = "./Dataset/database.json"
OUTPUT_DIR = "./hybrid_graphs_output"

SEMANTIC_THRESHOLD = 0.85
SEMANTIC_TOP_K = 10
SEMANTIC_BATCH_SIZE = 128
TIME_WINDOW_DAYS = 1
TEMPORAL_MAX_EDGES_PER_NODE = 50
ENTITY_THRESHOLD = 0.6
ENTITY_MAX_ARTICLES_PER_ENTITY = 50
ENTITY_MAX_PAIRS_PER_ARTICLE = 20
ENTITY_MIN_SHARED = 1
ENTITY_BATCH_SIZE = 50
VERBOSE = True
PLOT_DPI = 150

def load_embeddings(emb_path: str, ids_path: str):
    embeddings = np.load(emb_path)
    article_ids = np.load(ids_path, allow_pickle=True).tolist()
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(-1, 1)
    if embeddings.shape[0] != len(article_ids):
        raise ValueError(f"embeddings rows ({embeddings.shape[0]}) != number of article_ids ({len(article_ids)})")
    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain NaN or Inf")
    return embeddings, article_ids

def normalize_embeddings(emb: np.ndarray):
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return emb / norms

def load_database(json_path: str):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def parse_dates_for_articles(database: Dict, article_ids: List[str]):
    dates = {}
    invalid = 0
    for aid in article_ids:
        rec = database.get(aid, {})
        ds = rec.get('date', '')
        try:
            dt = parser.parse(ds)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            dates[aid] = dt
        except Exception:
            dates[aid] = None
            invalid += 1
    if VERBOSE:
        total = len(article_ids)
        valid = total - invalid
        print(f"dates loaded: total={total}, valid={valid}, invalid={invalid}", flush=True)
    return dates

def load_entities_spacy(database: Dict, article_ids: List[str], batch_size: int = ENTITY_BATCH_SIZE):
    nlp = spacy.load("en_core_web_sm")

    pipe_names = [name for name, _ in nlp.pipeline]

    disable_list = ['parser', 'lemmatizer', 'textcat']
    nlp_disable = [c for c in disable_list if c in pipe_names]

    nlp = spacy.load("en_core_web_sm", disable=nlp_disable)
    entities_dict: Dict[str, Set[str]] = {}
    texts = []
    aids = []
    for aid in article_ids:
        rec = database.get(aid, {})
        title = rec.get('title', '') or ''
        content = rec.get('content', '') or ''
        text = (title + " " + content)[:1500]
        texts.append(text)
        aids.append(aid)
    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting entities"):
        batch_texts = texts[i:i+batch_size]
        batch_aids = aids[i:i+batch_size]
        docs = nlp.pipe(batch_texts, batch_size=batch_size)
        for aid, doc in zip(batch_aids, docs):
            ents = set()
            for ent in doc.ents:
                if ent.label_ in {'PERSON', 'ORG', 'GPE', 'LOC', 'EVENT'}:
                    t = ent.text.lower().strip()
                    t = t.strip('''"'·`.,:;()[]{}<>/\\''')
                    if t:
                        ents.add(t)
            entities_dict[aid] = ents
    missing = [aid for aid in article_ids if aid not in entities_dict]
    for m in missing:
        entities_dict[m] = set()
    if VERBOSE:
        total = len(article_ids)
        nonempty = sum(1 for aid in article_ids if len(entities_dict.get(aid, set())) > 0)
        print(f"entities extracted: total_articles={total}, with_entities={nonempty}", flush=True)
    return entities_dict

def semantic_neighbors_ann(embeddings_norm: np.ndarray, top_k: int, threshold: Optional[float]=None):
    try:
        import faiss
        d = embeddings_norm.shape[1]
        index = faiss.IndexFlatIP(d)
        faiss.normalize_L2(embeddings_norm)
        index.add(embeddings_norm)
        k = top_k + 1
        D, I = index.search(embeddings_norm, k)
        i_list, j_list, w_list = [], [], []
        n = embeddings_norm.shape[0]
        for i in range(n):
            for j_idx, score in zip(I[i], D[i]):
                if j_idx == i:
                    continue
                if threshold is not None and score <= threshold:
                    continue
                a, b = (i, int(j_idx)) if i < int(j_idx) else (int(j_idx), i)
                i_list.append(a)
                j_list.append(b)
                w_list.append(float(score))
        return i_list, j_list, w_list
    except Exception:
        return None

def semantic_neighbors_fallback(embeddings_norm: np.ndarray, batch_size: int, top_k: int, threshold: Optional[float]=None):
    n = embeddings_norm.shape[0]
    i_list, j_list, w_list = [], [], []
    sub = max(1, batch_size)
    for start in tqdm(range(0, n, sub), desc="Computing semantic similarities"):
        end = min(start + sub, n)
        batch = embeddings_norm[start:end]
        sims = np.dot(batch, embeddings_norm.T)
        for local in range(sims.shape[0]):
            i_global = start + local
            row = sims[local]
            row[i_global] = -np.inf
            if threshold is not None:
                cand = np.where(row > threshold)[0]
            else:
                cand = np.arange(n)
            if cand.size == 0:
                continue
            if cand.size > top_k:
                top_idxs = np.argpartition(-row[cand], top_k-1)[:top_k]
                chosen = cand[top_idxs]
            else:
                chosen = cand
            for j in chosen:
                a, b = (i_global, int(j)) if i_global < int(j) else (int(j), i_global)
                i_list.append(a)
                j_list.append(b)
                w_list.append(float(row[int(j)]))
    return i_list, j_list, w_list

def compute_semantic_edges(embeddings: np.ndarray, top_k: int = SEMANTIC_TOP_K, threshold: Optional[float] = SEMANTIC_THRESHOLD, batch_size: int = SEMANTIC_BATCH_SIZE):
    embeddings_norm = normalize_embeddings(embeddings.copy())
    ann_result = semantic_neighbors_ann(embeddings_norm, top_k, threshold)
    if ann_result is not None:
        i_list, j_list, w_list = ann_result
    else:
        i_list, j_list, w_list = semantic_neighbors_fallback(embeddings_norm, batch_size, top_k, threshold)
    if VERBOSE:
        print(f"semantic candidate edges (pre-unique): {len(i_list)}", flush=True)
    return i_list, j_list, w_list

def compute_temporal_edges(article_ids: List[str], dates: Dict[str, Optional[datetime]], time_window_days: int = TIME_WINDOW_DAYS, max_edges_per_node: int = TEMPORAL_MAX_EDGES_PER_NODE):
    indexed = [(idx, aid, dates.get(aid)) for idx, aid in enumerate(article_ids) if dates.get(aid) is not None]
    indexed.sort(key=lambda x: x[2])
    n = len(indexed)
    i_list, j_list, w_list = [], [], []
    e = 1
    for i in tqdm(range(n), desc="Temporal edges"):
        idx_i, aid_i, date_i = indexed[i]
        if e < i + 1:
            e = i + 1
        while e < n:
            if (indexed[e][2] - date_i).total_seconds() / 86400.0 <= time_window_days:
                e += 1
            else:
                break
        added = 0
        for j in range(i + 1, e):
            if added >= max_edges_per_node:
                break
            idx_j = indexed[j][0]
            days_diff = (indexed[j][2] - date_i).total_seconds() / 86400.0
            weight = 1.0 / (days_diff + 1.0)
            a, b = (idx_i, idx_j) if idx_i < idx_j else (idx_j, idx_i)
            i_list.append(a)
            j_list.append(b)
            w_list.append(float(weight))
            added += 1
    if VERBOSE:
        print(f"temporal candidate edges (pre-unique): {len(i_list)}", flush=True)
    return i_list, j_list, w_list

def compute_entity_edges(article_ids: List[str], entities: Dict[str, Set[str]], threshold: float = ENTITY_THRESHOLD, max_articles_per_entity: int = ENTITY_MAX_ARTICLES_PER_ENTITY, max_pairs_per_article: int = ENTITY_MAX_PAIRS_PER_ARTICLE, min_shared: int = ENTITY_MIN_SHARED):
    entity_to_articles = defaultdict(list)
    for idx, aid in enumerate(article_ids):
        for ent in entities.get(aid, set()):
            entity_to_articles[ent].append(idx)
    i_list, j_list, w_list = [], [], []
    pair_counts = defaultdict(int)
    items = list(entity_to_articles.items())
    for ent, art_idxs in tqdm(items, desc="Entity -> article lists"):
        if len(art_idxs) < 2:
            continue
        art_idxs = art_idxs[:max_articles_per_entity]
        L = len(art_idxs)
        for a in range(L):
            i = art_idxs[a]
            if pair_counts[i] >= max_pairs_per_article:
                continue
            for b in range(a + 1, L):
                j = art_idxs[b]
                if pair_counts[j] >= max_pairs_per_article:
                    continue
                ents_i = entities.get(article_ids[i], set())
                ents_j = entities.get(article_ids[j], set())
                inter = len(ents_i.intersection(ents_j))
                union = len(ents_i.union(ents_j))
                sim = inter / union if union > 0 else 0.0
                if inter >= min_shared and sim > threshold:
                    a_idx, b_idx = (i, j) if i < j else (j, i)
                    i_list.append(a_idx)
                    j_list.append(b_idx)
                    w_list.append(float(sim))
                    pair_counts[i] += 1
                    pair_counts[j] += 1
                    if pair_counts[i] >= max_pairs_per_article:
                        break
    if VERBOSE:
        print(f"entity candidate edges (pre-unique): {len(i_list)}", flush=True)
    return i_list, j_list, w_list

def merge_edges(semantic_triplet, temporal_triplet, entity_triplet):
    edge_map = {}
    def add_edge(i, j, typ, w):
        key = frozenset({int(i), int(j)})
        if len(key) != 2:
            return
        if key not in edge_map:
            edge_map[key] = {'semantic': 0.0, 'temporal': 0.0, 'entity': 0.0, 'present': 0}
        if typ == 'semantic' and w > 0 and edge_map[key]['semantic'] == 0.0:
            edge_map[key]['semantic'] = float(w)
            edge_map[key]['present'] += 1
        if typ == 'temporal' and w > 0 and edge_map[key]['temporal'] == 0.0:
            edge_map[key]['temporal'] = float(w)
            edge_map[key]['present'] += 1
        if typ == 'entity' and w > 0 and edge_map[key]['entity'] == 0.0:
            edge_map[key]['entity'] = float(w)
            edge_map[key]['present'] += 1
    if semantic_triplet is not None:
        si, sj, sw = semantic_triplet
        for a, b, w in zip(si, sj, sw):
            add_edge(a, b, 'semantic', w)
    if temporal_triplet is not None:
        ti, tj, tw = temporal_triplet
        for a, b, w in zip(ti, tj, tw):
            add_edge(a, b, 'temporal', w)
    if entity_triplet is not None:
        ei, ej, ew = entity_triplet
        for a, b, w in zip(ei, ej, ew):
            add_edge(a, b, 'entity', w)
    edges = []
    counts = {'semantic': 0, 'temporal': 0, 'entity': 0}
    for key, vals in edge_map.items():
        members = sorted(list(key))
        i, j = members[0], members[1]
        present = vals['present'] if vals['present'] > 0 else 1
        s = vals['semantic']
        t = vals['temporal']
        e = vals['entity']
        total_present = sum(1 for x in [s, t, e] if x > 0)
        avg_weight = (s + t + e) / total_present if total_present > 0 else 0.0
        edges.append((i, j, float(avg_weight)))
        counts['semantic'] += 1 if s > 0 else 0
        counts['temporal'] += 1 if t > 0 else 0
        counts['entity'] += 1 if e > 0 else 0
    if VERBOSE:
        print(f"merged unique edges: {len(edges)}", flush=True)
    return edges, counts

def calculate_graph_statistics(edge_list: List[Tuple[int,int,float]], num_nodes: int):
    if num_nodes == 0:
        return {
            'num_nodes': 0, 'num_edges': 0, 'avg_degree': 0.0, 'max_degree': 0, 'min_degree': 0,
            'density': 0.0, 'avg_weight': 0.0, 'max_weight': 0.0, 'min_weight': 0.0
        }, [], []
    deg = np.zeros(num_nodes, dtype=int)
    weights = []
    seen = set()
    for i, j, w in edge_list:
        key = frozenset({int(i), int(j)})
        if key in seen:
            continue
        seen.add(key)
        deg[int(i)] += 1
        deg[int(j)] += 1
        weights.append(float(w))
    degree_values = deg.tolist()
    num_edges = len(seen)
    max_possible = (num_nodes * (num_nodes - 1)) / 2
    density = float(num_edges / max_possible) if max_possible > 0 else 0.0
    stats = {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'avg_degree': float(np.mean(degree_values)) if degree_values else 0.0,
        'max_degree': int(np.max(degree_values)) if degree_values else 0,
        'min_degree': int(np.min(degree_values)) if degree_values else 0,
        'density': density,
        'avg_weight': float(np.mean(weights)) if weights else 0.0,
        'max_weight': float(np.max(weights)) if weights else 0.0,
        'min_weight': float(np.min(weights)) if weights else 0.0
    }
    return stats, degree_values, weights

def save_graph_pickle(edge_list, node_features, num_nodes, num_edges, path):
    data = {'edge_list': edge_list, 'node_features': node_features, 'num_nodes': num_nodes, 'num_edges': num_edges}
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def save_node_mapping(article_ids: List[str], output_path: str):
    mapping = {str(idx): aid for idx, aid in enumerate(article_ids)}
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2)

def save_stats_csv(stats: Dict, output_path: str):
    pd.DataFrame([stats]).to_csv(output_path, index=False)

def save_edge_type_analysis(counts: Dict, output_path: str):
    total = sum(counts.values())
    rows = []
    for k, v in counts.items():
        pct = (v / total * 100) if total > 0 else 0.0
        rows.append({'edge_type': k, 'count': int(v), 'percentage': float(pct)})
    pd.DataFrame(rows).to_csv(output_path, index=False)

def generate_plots(degree_values, edge_weights, stats, edge_type_counts, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use('default')
    if degree_values:
        plt.figure(figsize=(10,6))
        plt.hist(degree_values, bins=min(50, max(5, len(degree_values)//2)))
        plt.title('Degree distribution (hybrid)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'degree_distribution_hybrid.png'), dpi=PLOT_DPI)
        plt.close()
    if edge_weights:
        plt.figure(figsize=(10,6))
        plt.hist(edge_weights, bins=min(50, max(5, len(edge_weights)//2)))
        plt.title('Edge weight distribution (hybrid)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'weight_distribution_hybrid.png'), dpi=PLOT_DPI)
        plt.close()
    plt.figure(figsize=(10,5))
    plt.bar(['num_nodes','num_edges'], [stats.get('num_nodes',0), stats.get('num_edges',0)])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'graph_counts_hybrid.png'), dpi=PLOT_DPI)
    plt.close()
    plt.figure(figsize=(8,5))
    types = list(edge_type_counts.keys())
    vals = [edge_type_counts[t] for t in types]
    plt.bar(types, vals)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'edge_type_distribution_hybrid.png'), dpi=PLOT_DPI)
    plt.close()

def build_hybrid_graph(
    embeddings_path: str,
    article_ids_path: str,
    json_path: str,
    output_dir: str,
    semantic_threshold: float = SEMANTIC_THRESHOLD,
    semantic_top_k: int = SEMANTIC_TOP_K,
    semantic_batch_size: int = SEMANTIC_BATCH_SIZE,
    time_window_days: int = TIME_WINDOW_DAYS,
    temporal_max_edges_per_node: int = TEMPORAL_MAX_EDGES_PER_NODE,
    entity_threshold: float = ENTITY_THRESHOLD,
    entity_max_articles_per_entity: int = ENTITY_MAX_ARTICLES_PER_ENTITY,
    entity_max_pairs_per_article: int = ENTITY_MAX_PAIRS_PER_ARTICLE,
    entity_min_shared: int = ENTITY_MIN_SHARED
):
    os.makedirs(output_dir, exist_ok=True)
    embeddings, article_ids = load_embeddings(embeddings_path, article_ids_path)
    database = load_database(json_path)
    dates = parse_dates_for_articles(database, article_ids)
    entities = load_entities_spacy(database, article_ids, batch_size=ENTITY_BATCH_SIZE)
    sem_i, sem_j, sem_w = compute_semantic_edges(embeddings, top_k=semantic_top_k, threshold=semantic_threshold, batch_size=semantic_batch_size)
    temp_i, temp_j, temp_w = compute_temporal_edges(article_ids, dates, time_window_days=time_window_days, max_edges_per_node=temporal_max_edges_per_node)
    ent_i, ent_j, ent_w = compute_entity_edges(article_ids, entities, threshold=entity_threshold, max_articles_per_entity=entity_max_articles_per_entity, max_pairs_per_article=entity_max_pairs_per_article, min_shared=entity_min_shared)
    edge_list, edge_type_counts = merge_edges((sem_i, sem_j, sem_w), (temp_i, temp_j, temp_w), (ent_i, ent_j, ent_w))
    num_nodes = len(article_ids)
    stats, degree_values, edge_weights = calculate_graph_statistics(edge_list, num_nodes)
    try:
        pd.DataFrame(edge_list, columns=['i','j','weight']).to_parquet(os.path.join(output_dir, 'merged_edges.parquet'), index=False)
    except Exception as e:
        if VERBOSE:
            print(f"parquet save failed: {e}", flush=True)
    node_feature_indices = np.arange(num_nodes, dtype=int)
    node_features = embeddings[node_feature_indices] if embeddings is not None and embeddings.size > 0 else np.zeros((num_nodes, embeddings.shape[1] if embeddings is not None and embeddings.ndim > 1 else 0))
    save_graph_pickle(edge_list, node_features, num_nodes, stats['num_edges'], os.path.join(output_dir, 'graph_hybrid.pkl'))
    save_node_mapping(article_ids, os.path.join(output_dir, 'node_mapping_hybrid.json'))
    save_stats_csv(stats, os.path.join(output_dir, 'graph_statistics_hybrid.csv'))
    save_edge_type_analysis(edge_type_counts, os.path.join(output_dir, 'edge_type_analysis_hybrid.csv'))
    generate_plots(degree_values, edge_weights, stats, edge_type_counts, output_dir)
    if VERBOSE:
        print("Hybrid graph built successfully", flush=True)
        print(f"Nodes: {stats['num_nodes']}, Edges: {stats['num_edges']}, Avg Degree: {stats['avg_degree']:.2f}", flush=True)
        print(f"Semantic edges: {edge_type_counts.get('semantic',0)}, Temporal edges: {edge_type_counts.get('temporal',0)}, Entity edges: {edge_type_counts.get('entity',0)}", flush=True)
    return {
        'stats': stats,
        'edge_type_counts': edge_type_counts,
        'edge_list': edge_list
    }

if __name__ == "__main__":
    build_hybrid_graph(
        EMBEDDINGS_FILE,
        ARTICLE_IDS_FILE,
        DATASET_JSON,
        OUTPUT_DIR,
        semantic_threshold=SEMANTIC_THRESHOLD,
        semantic_top_k=SEMANTIC_TOP_K,
        semantic_batch_size=SEMANTIC_BATCH_SIZE,
        time_window_days=TIME_WINDOW_DAYS,
        temporal_max_edges_per_node=TEMPORAL_MAX_EDGES_PER_NODE,
        entity_threshold=ENTITY_THRESHOLD,
        entity_max_articles_per_entity=ENTITY_MAX_ARTICLES_PER_ENTITY,
        entity_max_pairs_per_article=ENTITY_MAX_PAIRS_PER_ARTICLE,
        entity_min_shared=ENTITY_MIN_SHARED
    )
"4051509"