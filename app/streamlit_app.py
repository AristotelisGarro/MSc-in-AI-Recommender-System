import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide"
)

# ── Load data & models ────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    train  = pd.read_csv('../data/train.csv')
    test   = pd.read_csv('../data/test.csv')
    movies = pd.read_csv('../data/movies.csv')
    users  = pd.read_csv('../data/users.csv')
    return train, test, movies, users

@st.cache_resource
def load_models(movies):
    # SVD model
    with open('../data/svd_model.pkl', 'rb') as f:
        svd = pickle.load(f)

    # Item similarity matrix for CB
    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(movies['genres'].str.split('|'))
    genre_df = pd.DataFrame(genre_matrix, index=movies['item_id'], columns=mlb.classes_)
    sim_matrix = cosine_similarity(genre_df)
    item_sim_df = pd.DataFrame(sim_matrix, index=genre_df.index, columns=genre_df.index)

    return svd, item_sim_df, mlb

@st.cache_data
def load_results():
    with open('../results/cb_results.json')     as f: cb  = json.load(f)
    with open('../results/cf_results.json')     as f: cf  = json.load(f)
    with open('../results/hybrid_results.json') as f: hyb = json.load(f)
    return cb, cf, hyb

train, test, movies, users = load_data()
svd, item_sim_df, mlb      = load_models(movies)
cb_res, cf_res, hyb_res    = load_results()

GLOBAL_MEAN = train['rating'].mean()
BEST_ALPHA  = hyb_res['best_alpha']

# ── Helper functions ──────────────────────────────────────────────────────────
def cb_predict_score(user_id, item_id, user_ratings_dict):
    if user_id not in user_ratings_dict or item_id not in item_sim_df.index:
        return GLOBAL_MEAN
    rated     = user_ratings_dict[user_id]
    rated_ids = [i for i in rated if i in item_sim_df.index]
    if not rated_ids:
        return GLOBAL_MEAN
    sims   = item_sim_df.loc[item_id, rated_ids].values
    rtings = np.array([rated[i] for i in rated_ids])
    denom  = np.abs(sims).sum()
    return float(np.dot(sims, rtings) / denom) if denom > 0 else GLOBAL_MEAN

def get_recommendations(user_id, n=10, mode='hybrid'):
    user_ratings_dict = train.groupby('user_id').apply(
        lambda df: dict(zip(df['item_id'], df['rating']))
    ).to_dict()

    seen   = set(train[train['user_id'] == user_id]['item_id'].tolist())
    unseen = [i for i in movies['item_id'] if i not in seen]

    rows = []
    for iid in unseen:
        cb_score = cb_predict_score(user_id, iid, user_ratings_dict)
        cf_score = svd.predict(user_id, iid).est

        if   mode == 'cb':     score = cb_score
        elif mode == 'cf':     score = cf_score
        else:                  score = BEST_ALPHA * cb_score + (1 - BEST_ALPHA) * cf_score

        rows.append({'item_id': iid, 'predicted_rating': round(np.clip(score, 1, 5), 2)})

    df = pd.DataFrame(rows).nlargest(n, 'predicted_rating')
    return df.merge(movies[['item_id','title','genres']], on='item_id')

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🎬 Movie Recommender")
st.sidebar.markdown("**MSc AI — Θέμα 2**")
st.sidebar.markdown("Υβριδικό Σύστημα Συστάσεων")
st.sidebar.divider()

page = st.sidebar.radio(
    "Πλοήγηση",
    ["Αρχική", "Συστάσεις", "Αξιολόγηση Μοντέλων"]
)

# ── Page: Αρχική ──────────────────────────────────────────────────────────────
if page == "Αρχική":
    st.title("🎬 Υβριδικό Σύστημα Συστάσεων Ταινιών")
    st.markdown(
        "Σύστημα συστάσεων βασισμένο σε **MovieLens 1M** dataset, "
        "που συνδυάζει **Content-Based** και **Collaborative Filtering (SVD)**."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Χρήστες",  f"{users['user_id'].nunique():,}")
    col2.metric("Ταινίες",  f"{movies['item_id'].nunique():,}")
    col3.metric("Ratings",  f"{(len(train)+len(test)):,}")
    col4.metric("Βέλτιστο α", f"{BEST_ALPHA:.2f}")

    st.divider()
    st.subheader("Μεθοδολογία Υβριδικής Προσέγγισης")
    st.markdown("""
    | Στάδιο | Μέθοδος |
    |---|---|
    | **Content-Based** | Cosine similarity βάσει genres (MultiLabelBinarizer) |
    | **Collaborative** | SVD Matrix Factorization (100 factors, 20 epochs) |
    | **Hybrid** | Weighted average: `α·CB + (1-α)·CF`, α βελτιστοποιήθηκε με grid search |
    """)

# ── Page: Συστάσεις ───────────────────────────────────────────────────────────
elif page == "Συστάσεις":
    st.title("🎯 Εξατομικευμένες Συστάσεις")

    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        user_id = st.selectbox(
            "Επίλεξε User ID",
            sorted(train['user_id'].unique()),
            index=0
        )
    with col2:
        mode = st.radio(
            "Μοντέλο",
            ["hybrid", "cf", "cb"],
            format_func=lambda x: {"hybrid": "Hybrid", "cf": "Collaborative (SVD)", "cb": "Content-Based"}[x],
            horizontal=True
        )
    with col3:
        top_n = st.slider("Αριθμός συστάσεων", 5, 20, 10)

    if st.button("Παρουσίαση Συστάσεων", type="primary"):
        with st.spinner("Υπολογισμός συστάσεων..."):
            recs = get_recommendations(user_id, n=top_n, mode=mode)

        st.subheader(f"Top-{top_n} συστάσεις για User {user_id}")

        # Bar chart
        fig, ax = plt.subplots(figsize=(10, 4))
        short_titles = [t[:30] + "..." if len(t) > 30 else t for t in recs['title']]
        ax.barh(short_titles[::-1], recs['predicted_rating'][::-1], color='steelblue', edgecolor='black')
        ax.set_xlabel('Predicted Rating')
        ax.set_xlim(0, 5.5)
        ax.set_title(f'Top-{top_n} Recommendations — {mode.upper()}')
        plt.tight_layout()
        st.pyplot(fig)

        # Table
        st.dataframe(
            recs[['title', 'genres', 'predicted_rating']].reset_index(drop=True),
            use_container_width=True
        )

        # Ιστορικό χρήστη
        with st.expander("📋 Ιστορικό αξιολογήσεων χρήστη"):
            history = train[train['user_id'] == user_id].merge(
                movies[['item_id','title','genres']], on='item_id'
            )[['title','genres','rating']].sort_values('rating', ascending=False)
            st.dataframe(history.head(20), use_container_width=True)

# ── Page: Αξιολόγηση ─────────────────────────────────────────────────────────
elif page == "Αξιολόγηση Μοντέλων":
    st.title("📊 Σύγκριση Μοντέλων")

    summary = pd.DataFrame([
        {
            "Μοντέλο":        "Content-Based",
            "RMSE":           round(cb_res['rmse'], 4),
            "MAE":            round(cb_res['mae'],  4),
            "Precision@10":   round(cb_res['precision_recall_f1']['10']['Precision'], 4),
            "Recall@10":      round(cb_res['precision_recall_f1']['10']['Recall'],    4),
            "F1@10":          round(cb_res['precision_recall_f1']['10']['F1'],        4),
        },
        {
            "Μοντέλο":        "Collaborative (SVD)",
            "RMSE":           round(cf_res['rmse'], 4),
            "MAE":            round(cf_res['mae'],  4),
            "Precision@10":   round(cf_res['precision_recall_f1']['10']['Precision'], 4),
            "Recall@10":      round(cf_res['precision_recall_f1']['10']['Recall'],    4),
            "F1@10":          round(cf_res['precision_recall_f1']['10']['F1'],        4),
        },
        {
            "Μοντέλο":        f"Hybrid (α={BEST_ALPHA:.2f})",
            "RMSE":           round(hyb_res['rmse'], 4),
            "MAE":            round(hyb_res['mae'],  4),
            "Precision@10":   round(hyb_res['precision_recall_f1']['10']['Precision'], 4),
            "Recall@10":      round(hyb_res['precision_recall_f1']['10']['Recall'],    4),
            "F1@10":          round(hyb_res['precision_recall_f1']['10']['F1'],        4),
        },
    ])

    st.dataframe(summary.set_index("Μοντέλο"), use_container_width=True)

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    models = summary["Μοντέλο"].tolist()
    colors = ['#4C8CBF', '#E07B54', '#5CB85C']

    axes[0].bar(models, summary["RMSE"], color=colors, edgecolor='black', alpha=0.85)
    axes[0].set_title("RMSE (χαμηλότερο = καλύτερο)")
    axes[0].set_ylabel("RMSE")
    axes[0].tick_params(axis='x', rotation=10)
    for i, v in enumerate(summary["RMSE"]):
        axes[0].text(i, v + 0.003, f'{v:.4f}', ha='center', fontsize=9)

    axes[1].bar(models, summary["F1@10"], color=colors, edgecolor='black', alpha=0.85)
    axes[1].set_title("F1@10 (υψηλότερο = καλύτερο)")
    axes[1].set_ylabel("F1@10")
    axes[1].tick_params(axis='x', rotation=10)
    for i, v in enumerate(summary["F1@10"]):
        axes[1].text(i, v + 0.003, f'{v:.4f}', ha='center', fontsize=9)

    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Ανάλυση Αποτελεσμάτων")
    st.markdown(f"""
    - **Content-Based**: RMSE {cb_res['rmse']:.4f} — βασίζεται μόνο στα genres, αγνοεί τα patterns άλλων χρηστών
    - **Collaborative (SVD)**: RMSE {cf_res['rmse']:.4f} — εκμεταλλεύεται το interaction history, σημαντικά καλύτερο
    - **Hybrid (α={BEST_ALPHA:.2f})**: RMSE {hyb_res['rmse']:.4f} — το συνδυαστικό μοντέλο επιτυγχάνει το καλύτερο αποτέλεσμο
    - Το βέλτιστο α={BEST_ALPHA:.2f} δείχνει ότι το **CF κυριαρχεί** αλλά το CB συνεισφέρει θετικά
    """)
