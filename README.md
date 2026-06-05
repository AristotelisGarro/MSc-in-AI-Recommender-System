# Hybrid Movie Recommender System

**MSc in Artificial Intelligence — Εφαρμογές Τεχνητής Νοημοσύνης**  
**Θέμα 2: Υλοποίηση Υβριδικού Συστήματος Συστάσεων**

---

## Overview

Σύστημα συστάσεων ταινιών που υλοποιεί, συγκρίνει και συνδυάζει **Content-Based Filtering** και **Collaborative Filtering** σε υβριδική προσέγγιση. Εκπαιδεύεται και αξιολογείται στο **MovieLens 1M** dataset και παρέχει διαδραστικό web interface μέσω Streamlit.

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
| Min ratings/user | 20 |
| Train/Test split | 80% / 20% (random, seed=42) |

---

## Μεθοδολογία

### Content-Based Filtering (CB)

Δύο μέθοδοι, βασισμένες στα 18 genres κάθε ταινίας (binary vectors μέσω `MultiLabelBinarizer`):

| Μέθοδος | Περιγραφή |
|---|---|
| **Item-based CB** | `pred(u,i) = Σ sim(i,j)·r(u,j) / Σ\|sim(i,j)\|` — similarity-weighted average ratings |
| **User Profile CB** | Weighted mean genre vector → cosine similarity → γραμμική κλιμάκωση [0,1]→[1,5] |

Το Item-based CB επιλέχθηκε ως CB component του hybrid λόγω σαφούς υπεροχής (RMSE 1.0162 vs 1.4424).

### Collaborative Filtering (CF)

**Memory-based — KNN** (k=40, min_support=5):

| Ομοιότητα | RMSE |
|---|---|
| KNN User Cosine | 0.9742 |
| KNN User Pearson | 0.9612 |
| KNN User Jaccard | 0.9705 |
| KNN Item Cosine | 0.9996 |

**Model-based — SVD** (επιλεγμένο ως CF component):
```
r̂(u,i) = μ + b_u + b_i + q_i^T · p_u
```
Παράμετροι: `n_factors=100, n_epochs=20, lr=0.005, reg=0.02`, 3-fold CV

### Hybrid Models

| Μέθοδος | Τύπος |
|---|---|
| **Weighted Hybrid** | `score = α·CB + (1−α)·CF` — βέλτιστο **α=0.10** μέσω grid search |
| **Switching Hybrid** | `CB αν \|ratings(u)\| < threshold, αλλιώς CF` — βέλτιστο threshold=5 |

---

## Αποτελέσματα

| Μοντέλο | RMSE | MAE | P@5 | R@5 | F1@5 | P@10 | R@10 | F1@10 |
|---|---|---|---|---|---|---|---|---|
| Content-Based | 1.0162 | 0.8119 | 0.6793 | 0.3933 | 0.4982 | 0.6072 | 0.5956 | 0.6014 |
| Collaborative (SVD) | 0.8784 | 0.6902 | 0.7917 | 0.4404 | 0.5660 | 0.6866 | 0.6387 | 0.6618 |
| **Weighted Hybrid (α=0.10)** | **0.8768** | **0.6919** | **0.7929** | **0.4407** | **0.5665** | **0.6870** | **0.6390** | **0.6621** |
| Switching Hybrid (thr=5) | 0.8784 | 0.6902 | 0.7917 | 0.4404 | 0.5660 | 0.6866 | 0.6387 | 0.6618 |

### Βασικά Ευρήματα

- **Weighted Hybrid** επιτυγχάνει το καλύτερο RMSE και F1@10 — το CB component προσθέτει μικρή αλλά σταθερή βελτίωση πάνω στο pure SVD.
- **Χαμηλό α=0.10** υποδηλώνει ότι με πλούσιο interaction history (min 20 ratings/user) το CF κυριαρχεί. Το CB θα είχε μεγαλύτερη αξία σε cold-start σενάρια.
- **Switching Hybrid = pure CF** στο MovieLens 1M: κανένας χρήστης δεν έχει <5 ratings (minimum 20 εγγυημένο από το dataset), οπότε η CB διαδρομή δεν ενεργοποιείται ποτέ.
- **SVD >> KNN**: ~10% καλύτερο RMSE έναντι KNN Pearson. Λόγος: sparsity 95.5% — το SVD αξιοποιεί ολόκληρο τον πίνακα μέσω latent factors, οι KNN αδυνατούν να βρουν αξιόπιστους γείτονες.
- **Pearson > Cosine > Jaccard** για KNN: το Pearson mean-centering αντισταθμίζει το rating bias (μέσο rating 3.58). Το Jaccard αγνοεί εντελώς τις τιμές — κατάλληλο μόνο για implicit feedback.

---

## Δομή Project

```
MSc-in-AI-Recommender-System/
├── data/
│   ├── download_data.py          # Κατεβάζει το MovieLens 1M αυτόματα
│   ├── train.csv / test.csv      # 80/20 split (παράγεται από notebook 01)
│   ├── movies.csv / users.csv    # Processed dataset files
│   ├── svd_model.pkl             # Εκπαιδευμένο SVD model
│   ├── test_cb_preds.csv         # CB predictions στο test set
│   └── test_cf_preds.csv         # CF predictions στο test set
├── notebooks/
│   ├── 01_eda.ipynb               # EDA: κατανομές, demographics, train/test split
│   ├── 02_content_based.ipynb    # CB: Item-based & User Profile, σύγκριση
│   ├── 03_collaborative.ipynb    # CF: KNN (Cosine/Pearson/Jaccard) & SVD
│   └── 04_hybrid_evaluation.ipynb # Hybrid: Weighted & Switching, grid search α
├── app/
│   └── streamlit_app.py          # Web application (5 σελίδες)
├── results/
│   ├── cb_results.json           # CB metrics
│   ├── cf_results.json           # CF metrics
│   ├── hybrid_results.json       # Hybrid metrics + best_alpha
│   ├── final_comparison.csv      # Σύγκριση όλων των μοντέλων
│   └── *.png                     # Visualizations (EDA, CB, CF, Hybrid)
├── report/
│   ├── generate_report.py        # Παράγει self-contained HTML report
│   └── report.html               # Τελική αναφορά (open in Chrome → PDF)
└── requirements.txt
```

> **Εξάρτηση notebooks:** Κάθε notebook εξαρτάται από τα αρχεία που παράγει το προηγούμενο. Πρέπει να εκτελεστούν με σειρά 01→02→03→04.

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

### 4. Εκκίνηση Web App

```bash
cd app
streamlit run streamlit_app.py
```

Ανοίξτε τον browser στο **http://localhost:8501**

### 5. Παραγωγή Report

```bash
python report/generate_report.py
# → report/report.html (self-contained με embedded images)
# Για PDF: Chrome → Ctrl+P → Save as PDF
```

---

## Web App — Σελίδες

| Σελίδα | Περιεχόμενο |
|---|---|
| **Αρχική** | Στατιστικά dataset, μεθοδολογία, quick comparison (4 μοντέλα) |
| **Προφίλ Χρήστη** | Demographics, κατανομή ratings, αγαπημένα genres, ιστορικό με φίλτρα, δημοφιλείς αθέατες |
| **Συστάσεις** | Top-N (5–20) με επιλογή μοντέλου CB/CF/Hybrid, φιλτράρισμα ανά genre |
| **Σύγκριση Μοντέλων** | Side-by-side CB vs CF vs Hybrid για τον ίδιο χρήστη + overlap analysis |
| **Αξιολόγηση** | Πλήρης πίνακας μετρικών (4 μοντέλα), RMSE/MAE/F1 charts, ανάλυση |

---

## Dependencies

```
pandas
numpy
scikit-learn
scikit-surprise
streamlit
matplotlib
jupyter
```
