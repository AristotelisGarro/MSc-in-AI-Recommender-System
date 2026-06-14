import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page config ───────────────────────────────────────────────────────────────
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
    # Popularity (number of ratings per movie in train)
    popularity = train.groupby('item_id').agg(
        n_ratings=('rating', 'count'),
        avg_rating=('rating', 'mean')
    ).reset_index()
    return train, test, movies, users, popularity

@st.cache_resource
def load_models(movies):
    with open('../data/svd_model.pkl', 'rb') as f:
        svd = pickle.load(f)
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

@st.cache_data
def load_comparison():
    return pd.read_csv('../results/final_comparison.csv')

@st.cache_data
def build_user_ratings_dict(train_df):
    return train_df.groupby('user_id').apply(
        lambda df: dict(zip(df['item_id'], df['rating'])),
        include_groups=False
    ).to_dict()

train, test, movies, users, popularity = load_data()
svd, item_sim_df, mlb                  = load_models(movies)
cb_res, cf_res, hyb_res                = load_results()
comparison_df                          = load_comparison()
user_ratings_dict                      = build_user_ratings_dict(train)

GLOBAL_MEAN = train['rating'].mean()
BEST_ALPHA  = hyb_res['best_alpha']
ALL_GENRES  = sorted({g for genres in movies['genres'] for g in genres.split('|')})

# ── Helper functions ──────────────────────────────────────────────────────────
def cb_predict_score(user_id, item_id):
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
    seen   = set(train[train['user_id'] == user_id]['item_id'].tolist())
    unseen = [i for i in movies['item_id'] if i not in seen]
    rows = []
    for iid in unseen:
        cb_s = cb_predict_score(user_id, iid)
        cf_s = svd.predict(user_id, iid).est
        if   mode == 'cb':     score = cb_s
        elif mode == 'cf':     score = cf_s
        else:                  score = BEST_ALPHA * cb_s + (1 - BEST_ALPHA) * cf_s
        rows.append({'item_id': iid, 'cb_score': round(cb_s, 2),
                     'cf_score': round(cf_s, 2), 'predicted_rating': round(np.clip(score, 1, 5), 2)})
    df = pd.DataFrame(rows).nlargest(n, 'predicted_rating')
    return df.merge(movies[['item_id','title','genres']], on='item_id')

def rating_color(r):
    if r >= 4:   return 'background-color: #d4edda; color: #155724'
    elif r >= 3: return 'background-color: #fff3cd; color: #856404'
    else:        return 'background-color: #f8d7da; color: #721c24'

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🎬 Movie Recommender")
st.sidebar.markdown("**MSc AI — Θέμα 2**")
st.sidebar.markdown("**Γκάρο Αριστοτέλης — mtn2503**")
st.sidebar.markdown("Υβριδικό Σύστημα Συστάσεων")
st.sidebar.divider()

page = st.sidebar.radio(
    "Πλοήγηση",
    ["Αρχική", "Προφίλ Χρήστη", "Συστάσεις", "Σύγκριση Μοντέλων", "Αξιολόγηση"]
)

# ── Page: Αρχική ──────────────────────────────────────────────────────────────
if page == "Αρχική":
    st.title("🎬 Υβριδικό Σύστημα Συστάσεων Ταινιών")
    st.markdown(
        "Σύστημα συστάσεων βασισμένο σε **MovieLens 1M**, "
        "που συνδυάζει **Content-Based** και **Collaborative Filtering (SVD)**."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Χρήστες",    f"{users['user_id'].nunique():,}")
    c2.metric("Ταινίες",    f"{movies['item_id'].nunique():,}")
    c3.metric("Ratings",    f"{(len(train)+len(test)):,}")
    c4.metric("Βέλτιστο α", f"{BEST_ALPHA:.2f}")

    st.divider()
    st.subheader("Μεθοδολογία")
    st.markdown("""
    | Στάδιο | Μέθοδος |
    |---|---|
    | **Content-Based** | Cosine similarity βάσει genres (MultiLabelBinarizer) |
    | **Collaborative** | SVD Matrix Factorization (100 factors, 20 epochs) |
    | **Hybrid** | `score = α·CB + (1-α)·CF`, α βελτιστοποιήθηκε με grid search |
    """)

    st.divider()
    st.subheader("Αποτελέσματα")
    res_df = (
        comparison_df[['Model', 'RMSE', 'F1@10']]
        .rename(columns={'Model': 'Μοντέλο'})
        .set_index('Μοντέλο')
        .round(4)
    )
    st.dataframe(res_df, use_container_width=True)

# ── Page: Προφίλ Χρήστη ───────────────────────────────────────────────────────
elif page == "Προφίλ Χρήστη":
    st.title("👤 Προφίλ Χρήστη")

    user_id = st.selectbox("Επίλεξε User ID", sorted(train['user_id'].unique()), index=0)

    user_info = users[users['user_id'] == user_id].iloc[0]
    age_map   = {1:'<18', 18:'18-24', 25:'25-34', 35:'35-44', 45:'45-49', 50:'50-55', 56:'56+'}

    c1, c2, c3 = st.columns(3)
    c1.metric("Φύλο",   user_info['gender'])
    c2.metric("Ηλικία", age_map.get(user_info['age'], str(user_info['age'])))
    history_all = train[train['user_id'] == user_id]
    c3.metric("Αξιολογήσεις", len(history_all))

    st.divider()

    col_left, col_right = st.columns([1, 1])

    # Rating distribution για τον χρήστη
    with col_left:
        st.subheader("Κατανομή Ratings")
        rating_counts = history_all['rating'].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(5, 3))
        colors_bar = ['#d9534f','#f0ad4e','#f0ad4e','#5cb85c','#5cb85c']
        ax.bar(rating_counts.index, rating_counts.values, color=colors_bar[:len(rating_counts)], edgecolor='black')
        ax.set_xlabel('Rating')
        ax.set_ylabel('Αριθμός')
        ax.set_title(f'User {user_id} — Rating Distribution')
        plt.tight_layout()
        st.pyplot(fig)

    # Genre preferences
    with col_right:
        st.subheader("Αγαπημένα Genres")
        user_movies = history_all.merge(movies[['item_id','genres']], on='item_id')
        # Weighted by rating: genres από ταινίες με rating >= 4 vs όλες
        liked = user_movies[user_movies['rating'] >= 4]
        all_g  = [g for gs in user_movies['genres'] for g in gs.split('|')]
        liked_g = [g for gs in liked['genres'] for g in gs.split('|')]
        genre_all_s   = pd.Series(all_g).value_counts()
        genre_liked_s = pd.Series(liked_g).value_counts()
        genre_pref = pd.DataFrame({'Όλες': genre_all_s, 'Liked (≥4)': genre_liked_s}).fillna(0).astype(int)
        genre_pref = genre_pref.sort_values('Liked (≥4)', ascending=False).head(10)
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        x = np.arange(len(genre_pref))
        ax2.bar(x - 0.2, genre_pref['Όλες'],      0.4, label='Όλες',      color='#4C8CBF', edgecolor='black')
        ax2.bar(x + 0.2, genre_pref['Liked (≥4)'], 0.4, label='Liked (≥4)', color='#5CB85C', edgecolor='black')
        ax2.set_xticks(x)
        ax2.set_xticklabels(genre_pref.index, rotation=40, ha='right', fontsize=8)
        ax2.set_title('Top Genres')
        ax2.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig2)

    st.divider()
    st.subheader("📋 Πλήρες Ιστορικό Αξιολογήσεων")

    # Filters
    fc1, fc2 = st.columns([1, 2])
    with fc1:
        rating_filter = st.multiselect(
            "Φιλτράρισμα ανά Rating",
            options=[1, 2, 3, 4, 5],
            default=[1, 2, 3, 4, 5]
        )
    with fc2:
        genre_filter = st.multiselect(
            "Φιλτράρισμα ανά Genre",
            options=ALL_GENRES,
            default=[]
        )

    history_full = (
        history_all
        .merge(movies[['item_id','title','genres']], on='item_id')
        [['title', 'genres', 'rating']]
        .sort_values('rating', ascending=False)
    )

    if rating_filter:
        history_full = history_full[history_full['rating'].isin(rating_filter)]
    if genre_filter:
        mask = history_full['genres'].apply(
            lambda gs: any(g in gs.split('|') for g in genre_filter)
        )
        history_full = history_full[mask]

    # Color-code ratings
    def color_rating_row(row):
        if row['rating'] >= 4:   color = '#d4edda'
        elif row['rating'] >= 3: color = '#fff3cd'
        else:                    color = '#f8d7da'
        return [f'background-color: {color}'] * len(row)

    st.markdown(
        f"Εμφανίζονται **{len(history_full)}** αξιολογήσεις "
        "— 🟢 rating ≥ 4 &nbsp; 🟡 rating = 3 &nbsp; 🔴 rating ≤ 2"
    )
    st.dataframe(
        history_full.reset_index(drop=True).style.apply(color_rating_row, axis=1),
        use_container_width=True,
        height=400
    )

    st.divider()
    st.subheader("🔥 Δημοφιλείς Ταινίες που δεν έχεις δει")
    seen = set(history_all['item_id'].tolist())
    unseen_popular = (
        popularity[~popularity['item_id'].isin(seen)]
        .nlargest(15, 'n_ratings')
        .merge(movies[['item_id','title','genres']], on='item_id')
        [['title','genres','n_ratings','avg_rating']]
    )
    unseen_popular['avg_rating'] = unseen_popular['avg_rating'].round(2)
    st.dataframe(unseen_popular.reset_index(drop=True), use_container_width=True)

# ── Page: Συστάσεις ───────────────────────────────────────────────────────────
elif page == "Συστάσεις":
    st.title("🎯 Εξατομικευμένες Συστάσεις")

    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        user_id = st.selectbox("Επίλεξε User ID", sorted(train['user_id'].unique()), index=0)
    with c2:
        mode = st.radio(
            "Μοντέλο",
            ["hybrid", "cf", "cb"],
            format_func=lambda x: {"hybrid": "Hybrid", "cf": "Collaborative (SVD)", "cb": "Content-Based"}[x],
            horizontal=True
        )
    with c3:
        top_n = st.slider("Αριθμός συστάσεων", 5, 20, 10)

    genre_pref_filter = st.multiselect("Φιλτράρισμα αποτελεσμάτων ανά Genre (προαιρετικό)", ALL_GENRES, default=[])

    if st.button("Παρουσίαση Συστάσεων", type="primary"):
        with st.spinner("Υπολογισμός συστάσεων..."):
            recs = get_recommendations(user_id, n=50, mode=mode)

        if genre_pref_filter:
            mask = recs['genres'].apply(lambda gs: any(g in gs.split('|') for g in genre_pref_filter))
            recs = recs[mask]

        recs = recs.head(top_n)

        st.subheader(f"Top-{len(recs)} συστάσεις για User {user_id} ({mode.upper()})")

        fig, ax = plt.subplots(figsize=(10, max(4, len(recs) * 0.4)))
        short_titles = [t[:35] + "…" if len(t) > 35 else t for t in recs['title']]
        bars = ax.barh(short_titles[::-1], recs['predicted_rating'][::-1], color='steelblue', edgecolor='black')
        ax.set_xlabel('Predicted Rating')
        ax.set_xlim(0, 5.5)
        ax.set_title(f'Recommendations — {mode.upper()}  (User {user_id})')
        for bar, val in zip(bars, recs['predicted_rating'][::-1]):
            ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                    f'{val:.2f}', va='center', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)

        st.dataframe(
            recs[['title','genres','predicted_rating','cb_score','cf_score']].reset_index(drop=True),
            use_container_width=True
        )

# ── Page: Σύγκριση Μοντέλων ───────────────────────────────────────────────────
elif page == "Σύγκριση Μοντέλων":
    st.title("🔀 Σύγκριση CB vs CF vs Hybrid ανά Χρήστη")    

    user_id = st.selectbox("Επίλεξε User ID", sorted(train['user_id'].unique()), index=0)
    top_n   = st.slider("Top-N συστάσεις", 5, 15, 10)

    if st.button("Σύγκριση Μοντέλων", type="primary"):
        with st.spinner("Υπολογισμός και για τα 3 μοντέλα..."):
            recs_cb  = get_recommendations(user_id, n=top_n, mode='cb')
            recs_cf  = get_recommendations(user_id, n=top_n, mode='cf')
            recs_hyb = get_recommendations(user_id, n=top_n, mode='hybrid')

        col1, col2, col3 = st.columns(3)

        def show_recs(col, recs, label, color):
            with col:
                st.markdown(f"### {label}")
                fig, ax = plt.subplots(figsize=(5, 4))
                short = [t[:25] + "…" if len(t) > 25 else t for t in recs['title']]
                ax.barh(short[::-1], recs['predicted_rating'][::-1], color=color, edgecolor='black')
                ax.set_xlim(0, 5.5)
                ax.set_xlabel('Predicted Rating')
                plt.tight_layout()
                st.pyplot(fig)
                st.dataframe(recs[['title','predicted_rating']].reset_index(drop=True), use_container_width=True)

        show_recs(col1, recs_cb,  "Content-Based",       '#4C8CBF')
        show_recs(col2, recs_cf,  "Collaborative (SVD)", '#E07B54')
        show_recs(col3, recs_hyb, f"Hybrid (α={BEST_ALPHA})", '#5CB85C')

        st.divider()
        # Overlap analysis
        set_cb  = set(recs_cb['title'])
        set_cf  = set(recs_cf['title'])
        set_hyb = set(recs_hyb['title'])
        st.subheader("Ανάλυση Επικάλυψης")
        oc1, oc2, oc3, oc4 = st.columns(4)
        oc1.metric("CB ∩ CF",       len(set_cb  & set_cf))
        oc2.metric("CB ∩ Hybrid",   len(set_cb  & set_hyb))
        oc3.metric("CF ∩ Hybrid",   len(set_cf  & set_hyb))
        oc4.metric("Κοινές και στα 3", len(set_cb & set_cf & set_hyb))

        only_hybrid = set_hyb - set_cb - set_cf
        if only_hybrid:
            st.markdown("**Αποκλειστικά στο Hybrid** (δεν τα προτείνουν ούτε CB ούτε CF):")
            hyb_only_df = recs_hyb[recs_hyb['title'].isin(only_hybrid)][['title','genres','predicted_rating']]
            st.dataframe(hyb_only_df.reset_index(drop=True), use_container_width=True)

# ── Page: Αξιολόγηση ─────────────────────────────────────────────────────────
elif page == "Αξιολόγηση":
    st.title("📊 Αξιολόγηση Μοντέλων")

    summary = (
        comparison_df
        .rename(columns={'Model': 'Μοντέλο'})
        .round(4)
    )

    st.dataframe(summary.set_index("Μοντέλο"), use_container_width=True)

    st.divider()
    colors = ['#4C8CBF', '#E07B54', '#5CB85C', '#9B59B6']
    models = summary["Μοντέλο"].tolist()
    short_labels = ['CB', 'CF', 'W-Hybrid', 'S-Hybrid']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    x_pos = np.arange(len(models))
    axes[0].bar(x_pos, summary["RMSE"], color=colors, edgecolor='black')
    axes[0].set_title("RMSE (↓ καλύτερο)")
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(short_labels, fontsize=9, rotation=15)
    for i, v in enumerate(summary["RMSE"]):
        axes[0].text(i, v + 0.003, f'{v:.4f}', ha='center', fontsize=8)

    axes[1].bar(x_pos, summary["MAE"], color=colors, edgecolor='black')
    axes[1].set_title("MAE (↓ καλύτερο)")
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(short_labels, fontsize=9, rotation=15)
    for i, v in enumerate(summary["MAE"]):
        axes[1].text(i, v + 0.003, f'{v:.4f}', ha='center', fontsize=8)

    x = np.arange(len(models))
    axes[2].bar(x - 0.2, summary["F1@5"],  0.4, label='F1@5',  color=colors, edgecolor='black', alpha=0.85)
    axes[2].bar(x + 0.2, summary["F1@10"], 0.4, label='F1@10', color=colors, edgecolor='black', alpha=0.55)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(short_labels, fontsize=9)
    axes[2].set_title("F1 @5 και @10 (↑ καλύτερο)")
    p1 = mpatches.Patch(color='grey', alpha=0.85, label='@5')
    p2 = mpatches.Patch(color='grey', alpha=0.55, label='@10')
    axes[2].legend(handles=[p1, p2])

    plt.suptitle("Σύγκριση Μοντέλων — MovieLens 1M", fontsize=13, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)

    st.divider()
    st.subheader("Ανάλυση Αποτελεσμάτων")
    sw_rmse = summary.loc[summary['Μοντέλο'].str.startswith('Switching'), 'RMSE'].values[0]
    st.markdown(f"""
    - **Content-Based** (RMSE={cb_res['rmse']:.4f}): Βασίζεται αποκλειστικά στα genres — απλό αλλά αδύναμο, αγνοεί τα patterns άλλων χρηστών
    - **Collaborative SVD** (RMSE={cf_res['rmse']:.4f}): Εκμεταλλεύεται το interaction history, σημαντικά καλύτερο στο RMSE και F1
    - **Weighted Hybrid α={BEST_ALPHA}** (RMSE={hyb_res['rmse']:.4f}): Η καλύτερη προσέγγιση — το CB προσθέτει μικρή αλλά σταθερή βελτίωση
    - **Switching Hybrid thr=50** (RMSE={sw_rmse:.4f}): 36.1% χρήστες (<50 ratings στο train) → CB, 63.9% → CF. Ελαφρώς χειρότερο από pure CF — δείχνει ότι για engaged users το CF κυριαρχεί
    - Το χαμηλό α={BEST_ALPHA} δείχνει ότι με πλούσιο interaction history **το CF κυριαρχεί**, το CB έχει μεγαλύτερη αξία σε cold-start σενάρια
    """)
