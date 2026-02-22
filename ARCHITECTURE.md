Data Flow

Load 20 Newsgroups (18,846 docs) → build df (text, label)
Split into X_train / X_test (80/20)
Part 1: Vectorize with BoW / TF-IDF → train classifiers → evaluate
Part 2: Encode with SentenceTransformer → cache to embeddings.npy → train classifiers → evaluate
Part 3: Load embeddings → Elbow method → KMeans (K=5) → GPT labels top clusters → sub-cluster 2 biggest → print topic tree

Module Responsibilities

Data Loading — fetches dataset, builds DataFrame, splits train/test
Part 1 (Sparse Features) — vectorizes text with BoW/TF-IDF, trains MNB/LR/SVM/RF via sklearn Pipeline, reports Accuracy + Macro-F1, plots confusion matrix
Part 2 (Dense Embeddings) — encodes docs with all-MiniLM-L6-v2, caches embeddings, trains same classifiers on dense vectors
Part 3A (Clustering) — runs elbow method over K=2–9, fits final KMeans (K=5), assigns cluster labels
Part 3B (Sub-clustering) — identifies 2 largest clusters, re-clusters each into 3 sub-clusters
Part 3C (LLM Labeling) — sends representative docs to GPT, generates short topic labels for all clusters and sub-clusters, prints 2-level topic tree

Key Variables

df — full dataset (text + label)
X, y — features and targets
X_train, X_test — 80/20 split
embeddings — SentenceTransformer vectors (shape: N×384)
emb — embeddings aligned to working set
cluster_labels — KMeans assignment per document
top_labels — GPT label per top-level cluster
sub_labels — GPT label per sub-cluster
