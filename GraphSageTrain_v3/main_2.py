import os
import math
import torch
import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, GATv2Conv
from sklearn.model_selection import train_test_split
import pickle


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_CSV = "/storage32Tb/jay/mixtralModel/ADL Project/Dataset/train_set.csv"
ARTICLE_EMB_FILE = "/storage32Tb/jay/mixtralModel/ADL Project/eventa_embeddings_Qwen3/database_embeddings_Qwen3.npy"
ARTICLE_IDS_FILE = "/storage32Tb/jay/mixtralModel/ADL Project/eventa_embeddings_Qwen3/database_article_ids_Qwen3.npy"
CAPTION_EMB_FILE = "/storage32Tb/jay/mixtralModel/ADL Project/Dataset/caption_embeddings_qwen.npy"
HIDDEN_DIM = 1024
EMBED_DIM = 512
TEMPERATURE = 0.07
LR = 1e-5
BATCH_SIZE = 32
EPOCHS = 50
PATIENCE = 10
K_NEIGHBORS = 128
NUM_HARD_NEGATIVES = 7
VAL_QUERIES = 256
OUTPUT_DIR = "/storage32Tb/jay/mixtralModel/ADL Project/graphSageTrain/Results"
SUBGRAPH_BATCH = 256
KNN_CACHE_FILE = "/storage32Tb/jay/mixtralModel/ADL Project/graphSageTrain/knn_graph_cache.pkl"
METRICS_CSV = os.path.join(OUTPUT_DIR, "training_metrics.csv")


os.makedirs(OUTPUT_DIR, exist_ok=True)


def l2_norm_np(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def build_faiss(x):
    a = np.ascontiguousarray(x.astype("float32"))
    faiss.normalize_L2(a)
    idx = faiss.IndexFlatIP(a.shape[1])
    idx.add(a)
    return idx


def build_article_knn_graph(art_emb, k=32, cache_file=None):
    if cache_file and os.path.exists(cache_file):
        print(f"Loading cached KNN graph from {cache_file}")
        with open(cache_file, 'rb') as f:
            cached = pickle.load(f)
        return cached['neighbors'], cached['edge_index']
    
    faiss_idx = build_faiss(art_emb)
    n_articles = art_emb.shape[0]
    edges = [[], []]
    neighbors = {}
    for i in tqdm(range(n_articles)):
        q = art_emb[i:i+1]
        q = np.ascontiguousarray(q, dtype='float32')
        D, I = faiss_idx.search(q, k+1)
        lst = []
        for nb in I[0]:
            if nb != i:
                lst.append(int(nb))
                edges[0].append(i)
                edges[1].append(int(nb))
        neighbors[i] = lst
    edge_index = torch.tensor(edges, dtype=torch.long)
    
    if cache_file:
        print(f"Saving KNN graph to {cache_file}")
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump({'neighbors': neighbors, 'edge_index': edge_index}, f)
    
    return neighbors, edge_index


class ImprovedGraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.15, use_attention=True):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        
        self.conv1 = SAGEConv(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        self.use_attention = use_attention
        if use_attention:
            self.conv2 = GATv2Conv(hidden_dim, hidden_dim, heads=1, 
                                   dropout=dropout, concat=False, add_self_loops=True)
        else:
            self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index):
        h0 = F.gelu(self.input_norm(self.input_proj(x)))
        
        h1 = self.conv1(h0, edge_index)
        h1 = self.norm1(h1)
        h1 = F.gelu(h1)
        h1 = self.dropout(h1)
        h1 = h1 + h0
        
        h2 = self.conv2(h1, edge_index)
        h2 = self.norm2(h2)
        h2 = F.gelu(h2)
        h2 = self.dropout(h2)
        h2 = h2 + h1
        
        out = self.output_proj(h2)
        
        return F.normalize(out, dim=-1)


class EnhancedDualEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim, dropout=0.15, use_attention=True):
        super().__init__()
        
        self.article_encoder = ImprovedGraphEncoder(
            input_dim, hidden_dim, embed_dim, dropout, use_attention
        )
        
        self.caption_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim)
        )
        
    def encode_article(self, x, edge_index):
        return self.article_encoder(x, edge_index)
    
    def encode_caption(self, x):
        return F.normalize(self.caption_encoder(x), dim=-1)
    
    def forward(self, article_x, edge_index, caption_x):
        article_embed = self.encode_article(article_x, edge_index)
        caption_embed = self.encode_caption(caption_x)
        return article_embed, caption_embed


def hard_negative_contrastive_loss(model, art_emb, cap_raw, pos_pairs, art_neighbors, num_hard_neg, device):
    batch_size = len(pos_pairs)
    pos_art_idx = torch.tensor([p[0] for p in pos_pairs], device=device)
    pos_cap_idx = torch.tensor([p[1] for p in pos_pairs], device=device)
    cap_emb = model.encode_caption(cap_raw)
    pos_art_emb = art_emb[pos_art_idx]
    pos_cap_emb = cap_emb[pos_cap_idx]
    neg_idx = []
    n = art_emb.size(0)
    for a in pos_art_idx.cpu().numpy():
        if a in art_neighbors and len(art_neighbors[a]) > 0:
            s = min(num_hard_neg, len(art_neighbors[a]))
            c = np.random.choice(art_neighbors[a], size=s, replace=False).tolist()
            neg_idx.extend(c)
        else:
            neg_idx.extend(np.random.choice(n, size=num_hard_neg, replace=True).tolist())
    neg_emb = art_emb[neg_idx].view(batch_size, num_hard_neg, -1)
    pos_sim = (pos_art_emb * pos_cap_emb).sum(-1)
    neg_sim = torch.bmm(pos_cap_emb.unsqueeze(1), neg_emb.transpose(1, 2)).squeeze(1)
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], 1) / TEMPERATURE
    labels = torch.zeros(batch_size, dtype=torch.long, device=device)
    return F.cross_entropy(logits, labels)


def recall_at_k(pred, gt, k):
    return sum(1 for i, g in enumerate(gt) if g in pred[i,:k]) / len(gt) if len(gt)>0 else 0


def validate(model, art_emb_numpy, cap_emb, art_ids, df, val_idx, topk=20):
    model.eval()
    with torch.no_grad():
        art_z = art_emb_numpy
        cap_z = model.encode_caption(torch.tensor(cap_emb).float().to(DEVICE)).cpu().numpy()
    idx = build_faiss(art_z)
    val_idx = np.array(val_idx)
    if len(val_idx)>VAL_QUERIES:
        val_idx = np.random.choice(val_idx, VAL_QUERIES, False)
    q = np.ascontiguousarray(cap_z[val_idx], dtype="float32")
    _, pred = idx.search(q, topk)
    gts_raw = df["retrieved_article_id"].astype(str).values[val_idx]
    mid = {str(a):i for i,a in enumerate(art_ids)}
    gt_idx = [mid[g] if g in mid else -1 for g in gts_raw]
    mask = [i for i,v in enumerate(gt_idx) if v!=-1]
    if not mask:
        return {"r1":0,"r5":0,"r10":0,"r20":0}
    pred = pred[mask]
    gt = [gt_idx[i] for i in mask]
    return {
        "r1": recall_at_k(pred, gt, 1),
        "r5": recall_at_k(pred, gt, 5),
        "r10": recall_at_k(pred, gt, 10),
        "r20": recall_at_k(pred, gt, 20)
    }


def plot_curve(arr,name):
    plt.figure(figsize=(10,6))
    plt.plot(arr,linewidth=2)
    plt.grid(True,linestyle="--",alpha=0.6)
    plt.title(name)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,f"{name}.png"))
    plt.close()


def compute_article_embeddings(model, x_full, edge_index_full, device, batch_size=SUBGRAPH_BATCH):
    N = x_full.size(0)
    emb = torch.zeros((N, EMBED_DIM))
    model.eval()
    
    edge_dict = {}
    for i in range(edge_index_full.size(1)):
        src = int(edge_index_full[0, i])
        dst = int(edge_index_full[1, i])
        if src not in edge_dict:
            edge_dict[src] = []
        edge_dict[src].append(dst)
    
    for i in tqdm(range(0, N, batch_size), desc="Computing embeddings"):
        end_idx = min(i + batch_size, N)
        batch_nodes = list(range(i, end_idx))
        
        one_hop = set(batch_nodes)
        for node in batch_nodes:
            if node in edge_dict:
                one_hop.update(edge_dict[node][:32])
        
        sub_nodes = sorted(list(one_hop))
        node_to_idx = {n: idx for idx, n in enumerate(sub_nodes)}
        
        sub_x = x_full[sub_nodes].to(device)
        
        sub_edges = [[], []]
        for node in sub_nodes:
            if node in edge_dict:
                for neighbor in edge_dict[node]:
                    if neighbor in node_to_idx:
                        sub_edges[0].append(node_to_idx[node])
                        sub_edges[1].append(node_to_idx[neighbor])
        
        if len(sub_edges[0]) > 0:
            sub_edge_index = torch.tensor(sub_edges, dtype=torch.long).to(device)
        else:
            sub_edge_index = torch.empty((2, 0), dtype=torch.long).to(device)
        
        with torch.no_grad():
            out = model.encode_article(sub_x, sub_edge_index)
            
            for batch_node in batch_nodes:
                local_idx = node_to_idx[batch_node]
                emb[batch_node] = out[local_idx].cpu()
        
        del sub_x, sub_edge_index, out
        torch.cuda.empty_cache()
    
    return emb.numpy()


def save_metrics_to_csv(epoch, loss, metrics, csv_path):
    row = {
        'epoch': epoch,
        'loss': loss,
        'recall@1': metrics['r1'],
        'recall@5': metrics['r5'],
        'recall@10': metrics['r10'],
        'recall@20': metrics['r20']
    }
    df_new = pd.DataFrame([row])
    if os.path.exists(csv_path):
        df_new.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df_new.to_csv(csv_path, mode='w', header=True, index=False)


def main():
    art_all = np.load(ARTICLE_EMB_FILE).astype("float32")
    art_ids_all = np.load(ARTICLE_IDS_FILE, allow_pickle=True).tolist()
    df = pd.read_csv(TRAIN_CSV).dropna(subset=["caption","retrieved_article_id"]).reset_index()
    cap_all = np.load(CAPTION_EMB_FILE).astype("float32")
    if cap_all.shape[0]!=len(df):
        raise RuntimeError()
    art_ids = [str(a) for a in art_ids_all]
    art = l2_norm_np(art_all)
    cap_all = l2_norm_np(cap_all)
    if cap_all.shape[1]!=art.shape[1]:
        m=min(cap_all.shape[1],art.shape[1])
        cap_all=cap_all[:,:m]
        art=art[:,:m]
    input_dim = art.shape[1]
    
    art_neighbors, edge_index = build_article_knn_graph(art, k=K_NEIGHBORS, cache_file=KNN_CACHE_FILE)
    
    data = Data(x=torch.tensor(art).float(), edge_index=edge_index)
    aid = {a:i for i,a in enumerate(art_ids)}
    idx_all = np.arange(len(df))
    train_idx,val_idx = train_test_split(idx_all,test_size=0.2,random_state=42)
    train_pairs=[]
    for cap_idx in train_idx:
        a = str(df.loc[cap_idx,"retrieved_article_id"])
        if a in aid:
            train_pairs.append((aid[a],cap_idx))
    model = EnhancedDualEncoder(input_dim, HIDDEN_DIM, EMBED_DIM).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS,eta_min=1e-6)
    best_r5=-1
    patience=0
    losses=[]; r1s=[]; r5s=[]; r10s=[]; r20s=[]; grads=[]
    cap_tensor=torch.tensor(cap_all).float().to(DEVICE)
    for epoch in range(1,EPOCHS+1):
        model.train()
        art_z = compute_article_embeddings(model, data.x, data.edge_index, DEVICE)
        art_emb_tensor = torch.tensor(art_z).float().to(DEVICE)
        np.random.shuffle(train_pairs)
        ep_loss=0
        nb=0
        for i in range(0,len(train_pairs),BATCH_SIZE):
            batch = train_pairs[i:i+BATCH_SIZE]
            loss = hard_negative_contrastive_loss(model,art_emb_tensor,cap_tensor,batch,art_neighbors,NUM_HARD_NEGATIVES,DEVICE)
            opt.zero_grad()
            loss.backward()
            gn = math.sqrt(sum((p.grad.data.norm(2).item()**2 for p in model.parameters() if p.grad is not None)))
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step()
            ep_loss+=loss.item()
            nb+=1
            grads.append(gn)
        losses.append(ep_loss/nb)
        scheduler.step()
        vm = validate(model,art_z,cap_all,art_ids,df,val_idx,20)
        r1s.append(vm["r1"]); r5s.append(vm["r5"]); r10s.append(vm["r10"]); r20s.append(vm["r20"])
        save_metrics_to_csv(epoch, losses[-1], vm, METRICS_CSV)
        print(f"Epoch {epoch}: Loss={losses[-1]:.4f}, R@1={vm['r1']:.4f}, R@5={vm['r5']:.4f}, R@10={vm['r10']:.4f}, R@20={vm['r20']:.4f}")
        if vm["r5"]>best_r5:
            best_r5=vm["r5"]; patience=0
            torch.save({'model_state_dict':model.state_dict(),'article_edge_index':edge_index,'article_ids':art_ids}, os.path.join(OUTPUT_DIR,"best_model.pt"))
        else:
            patience+=1
        if patience>=PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break
    torch.save({'model_state_dict':model.state_dict(),'article_edge_index':edge_index,'article_ids':art_ids}, os.path.join(OUTPUT_DIR,"final_model.pt"))
    plot_curve(losses,"loss")
    plot_curve(r1s,"val_r1")
    plot_curve(r5s,"val_r5")
    plot_curve(r10s,"val_r10")
    plot_curve(r20s,"val_r20")
    plt.figure(figsize=(10,6))
    plt.plot(grads)
    plt.grid(True,linestyle="--",alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,"grad_norm.png"))
    plt.close()


if __name__=="__main__":
    main()
