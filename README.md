NLP Pipeline — 20 Newsgroups
Overview
Multi-class text classification and topic clustering on the 20 Newsgroups dataset using sparse features, dense embeddings, and LLM-generated topic labels.

Setup
conda create -n nlp_pipeline python=3.10
conda activate nlp_pipeline
pip install scikit-learn pandas numpy matplotlib seaborn sentence-transformers openai

How to Run
Run all cells sequentially in Jupyter notebook.
Part 1 — BoW / TF-IDF + Classifiers
Trains MNB, Logistic Regression, Linear SVM, Random Forest with BoW and TF-IDF. Best: TF-IDF + Linear SVM (F1=0.75)
Part 2 — SentenceTransformer Embeddings
Encodes docs with all-MiniLM-L6-v2, trains same classifiers. Embeddings cached to embeddings.npy.
Part 3 — Clustering + Topic Tree
KMeans (K=5) on embeddings → GPT labels each cluster → 2-level topic tree.

⚠️ Requires OpenAI API key — replace API_HERE in the code

Files
NLP Pipeline.ipynb
embeddings.npy (auto-generated)
README.md
