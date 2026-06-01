# Hybrid Movie Recommender System

**MSc in Artificial Intelligence — Εφαρμογές Τεχνητής Νοημοσύνης**
**Θέμα 2: Υλοποίηση Υβριδικού Συστήματος Συστάσεων**

---

## Overview

Σύστημα συστάσεων ταινιών που συνδυάζει **Content-Based Filtering** και **Collaborative Filtering (SVD)** σε μια υβριδική προσέγγιση weighted average. Το σύστημα εκπαιδεύεται και αξιολογείται στο **MovieLens 1M** dataset και παρέχει web interface μέσω Streamlit.

---

## Dataset

**MovieLens 1M** — [grouplens.org/datasets/movielens/1m](https://grouplens.org/datasets/movielens/1m/)

| | |
|---|---|
| Χρήστες | 6,040 |
| Ταινίες | 3,706 |
| Ratings | 1,000,209 |
| Sparsity | ~95.5% |
| Rating scale | 1–5 |

---

## Μεθοδολογία

### Content-Based Filtering
- Κάθε ταινία αναπαρίσταται ως binary vector 18 genres (MultiLabelBinarizer)
- Πρόβλεψη μέσω **cosine similarity**: weighted average των ratings του χρήστη, σταθμισμένο με την ομοιότητα ταινιών

### Collaborative Filtering
- **SVD (Matrix Factorization)** — 100 latent factors, 20 epochs, lr=0.005, reg=0.02
- Υλοποίηση μέσω `scikit-surprise`

### Hybrid Model
```
score(u, i) = α × CB_score + (1 − α) × CF_score
```
- Το βέλτιστο **α = 0.10** βρέθηκε μέσω grid search (ελαχιστοποίηση RMSE)

---

## Αποτελέσματα

| Μοντέλο | RMSE | MAE | Precision@10 | Recall@10 | F1@10 |
|---|---|---|---|---|---|
| Content-Based | 1.0162 | 0.8119 | 0.6072 | 0.5956 | 0.6014 |
| Collaborative (SVD) | 0.8784 | 0.6902 | 0.6866 | 0.6387 | 0.6618 |
| **Hybrid (α=0.10)** | **0.8768** | **0.6919** | **0.6870** | **0.6390** | **0.6621** |

---

## Δομή Project

```
MSc-in-AI-Recommender-System/
├── data/
│   └── download_data.py          # Κατεβάζει το MovieLens 1M αυτόματα
├── notebooks/
│   ├── 01_eda.ipynb               # Exploratory Data Analysis
│   ├── 02_content_based.ipynb    # Content-Based Baseline
│   ├── 03_collaborative.ipynb    # SVD Collaborative Filtering
│   └── 04_hybrid_evaluation.ipynb # Hybrid + Τελική Σύγκριση
├── app/
│   └── streamlit_app.py          # Web application
├── results/                       # Plots & JSON αποτελεσμάτων
└── requirements.txt
```

---

## Εγκατάσταση & Εκτέλεση

### 1. Εγκατάσταση dependencies

```bash
pip install -r requirements.txt
```

### 2. Κατέβασμα Dataset

```bash
python data/download_data.py
```

### 3. Εκτέλεση Notebooks (με σειρά)

```bash
jupyter notebook notebooks/01_eda.ipynb
jupyter notebook notebooks/02_content_based.ipynb
jupyter notebook notebooks/03_collaborative.ipynb
jupyter notebook notebooks/04_hybrid_evaluation.ipynb
```

> Κάθε notebook αποθηκεύει ενδιάμεσα αρχεία στο `data/` και αποτελέσματα στο `results/`.  
> Πρέπει να εκτελεστούν **με σειρά** γιατί το καθένα εξαρτάται από το προηγούμενο.

### 4. Εκκίνηση Web App

```bash
cd app
streamlit run streamlit_app.py
```

Ανοίξτε τον browser στο **http://localhost:8501**

---

## Web App — Σελίδες

| Σελίδα | Περιγραφή |
|---|---|
| **Αρχική** | Στατιστικά dataset, επεξήγηση μεθοδολογίας |
| **Συστάσεις** | Επιλογή user ID, μοντέλου (CB / CF / Hybrid) και αριθμού συστάσεων |
| **Αξιολόγηση Μοντέλων** | Σύγκριση RMSE, MAE, Precision/Recall/F1 και για τα 3 μοντέλα |

---

## Dependencies

```
pandas, numpy, scikit-learn, scikit-surprise, streamlit, matplotlib, seaborn
```
