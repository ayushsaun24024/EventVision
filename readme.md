# 🎯 EventVision: Context-Aware Image Retrieval from Long Captions

> *Because finding the right image shouldn't be like finding a needle in a haystack*

Ever tried searching for an image of "massive protests following the 2023 election in Argentina"? Traditional models like CLIP fall flat with their 77-token limit when you need to capture the full story. **EventVision** tackles this head-on by understanding complex, event-centric descriptions and retrieving images that actually match the narrative .

## 🚀 What's This About?

This project builds a **dual-stage retrieval pipeline** that bridges the gap between long, information-rich captions and relevant images from a massive database (200K+ articles, each with multiple images). Think of it as a smart librarian who actually reads the full context before handing you exactly what you need .

### The Challenge
- Traditional vision-language models choke on long captions 
- Event-centric queries need understanding of causality, temporal dynamics, and participant roles 
- We need to search through 200K+ articles efficiently while maintaining accuracy 

### Our Solution
A hybrid retrieval system that combines:
1. **Fast FAISS-based semantic search** to narrow down candidates 
2. **Smart reranking** using cross-encoders for articles and CLIP for images 
3. **Two-hop retrieval**: First find relevant articles, then pinpoint the perfect images 

## 🏗️ Architecture

### Stage 1: Article Retrieval
- **Embedding Model**: `Qwen/Qwen3-Embedding-0.6B` or `BAAI/bge-m3` 
- **Initial Retrieval**: FAISS index fetches top-100 article candidates 
- **Reranking**: `Qwen3-Reranker-0.6B` or `bge-reranker-v2-m3` cross-encoder refines to top-20 

### Stage 2: Image Retrieval
- **Candidate Pool**: Gather images from top-20 articles 
- **Visual Reranking**: CLIP (`vit-large-patch14` or `vit-base-patch32`) scores query-image similarity 
- **Final Output**: Top-10 most relevant images 

## 📊 Performance

Tested on a validation split with 2,204 queries :

### Qwen3 Pipeline Results 
| Metric | Article Retrieval | Image Retrieval |
|--------|------------------|-----------------|
| **mAP** | - | - |
| **MRR** | - | - |
| **Recall@1** | - | 0.26 |
| **Recall@5** | - | 0.50 |
| **Recall@10** | - | 0.57 |

### BGE-m3 Pipeline Results 
| Metric | Article Retrieval | Image Retrieval |
|--------|------------------|-----------------|
| **mAP** | 0.5505 | 0.3581 |
| **MRR** | 0.5505 | 0.3581 |
| **Recall@1** | 0.4578 | 0.2595 |
| **Recall@5** | 0.6656 | 0.4964 |
| **Recall@10** | 0.7391 | 0.5672 |

*Pro tip: The article retrieval scores look way better, but that's because images are the final boss level* 🎮

## 🛠️ Tech Stack

- **PyTorch** + **Transformers** for model operations 
- **FAISS** for blazing-fast similarity search 
- **Sentence-Transformers** for encoding 
- **CLIP** for multimodal understanding 
- **NumPy** + **Pandas** for data wrangling 

## 🎯 Key Features

✅ **Handles long captions**: No more 77-token CLIP limits  
✅ **Two-stage reranking**: Balances speed and accuracy  
✅ **Precomputed embeddings**: Lightning-fast inference with FAISS   
✅ **Batch processing**: GPU-optimized for efficiency   
✅ **Flexible architecture**: Swap models easily (tested Qwen3 and BGE-m3)   

## 📝 Dataset

- **Database**: 202,803 articles with associated images 
- **Train**: 19,836 query-article-image triplets 
- **Validation**: 2,204 samples (90/10 split) 
- **Test**: 3,000 queries for final evaluation 

Each article contains title, content, and a list of image IDs .

## 🔬 Evaluation Metrics

We track multiple retrieval metrics :
- **mAP**: Mean Average Precision 
- **MRR**: Mean Reciprocal Rank 
- **Recall@K**: Hit rate at top K positions (K = 1, 5, 10, 20, 50) 

## 💡 Cool Implementation Details

1. **Smart Tokenization**: Query-document pairs are formatted with special tokens for the Qwen3 reranker 
2. **Memory Management**: Aggressive GPU cache clearing after each batch 
3. **Normalization**: L2-normalized embeddings for cosine similarity via inner product 
4. **Robust Image Loading**: Handles missing/corrupted images gracefully 
5. **Submission Format**: Auto-fills with "#" for missing predictions 

## 🎨 Visualizations

The notebook generates performance charts comparing article vs. image retrieval across different recall thresholds. Spoiler: Getting the right article is easier than nailing the exact image 📸

## 🤝 Team

**Guide**: Dr. A. V. Subramanyam 

**Contributors**:
- Ayush Saun (MT24024) - Advanced retrieval framework 
- Manogna Pasumarthi (MT24127) - Model adaptation mechanisms 

## 📚 References

This project is inspired by recent work in event-aware multimodal retrieval:
- EVENT-Retriever: Event-Aware Multimodal Image Retrieval 
- Event-Enriched Image Analysis challenges 
- Hybrid dense-sparse reranking frameworks 

## 🚦 Running the Code

The entire pipeline lives in a single Jupyter notebook (`.ipynb`). Just make sure you have:
- A dataset folder with `database.json`, CSV files, and compressed images
- Precomputed embeddings (or run the embedding generation cells)
- A GPU (trust us, you'll need it) 🔥

Hit run and watch the magic happen! ✨

---

*Built with ❤️ and way too much GPU time*
