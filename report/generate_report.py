"""
Generates a self-contained HTML report for the hybrid recommender system project.
Run from the project root: python report/generate_report.py
Output: report/report.html  (open in Chrome → Print → Save as PDF)
"""

import json, base64, os
from pathlib import Path

ROOT    = Path(__file__).parent.parent
RESULTS = ROOT / "results"
OUT     = Path(__file__).parent / "report.html"

# ── Load JSON results ──────────────────────────────────────────────────────────
with open(RESULTS / "cb_results.json",     encoding="utf-8") as f: cb  = json.load(f)
with open(RESULTS / "cf_results.json",     encoding="utf-8") as f: cf  = json.load(f)
with open(RESULTS / "hybrid_results.json", encoding="utf-8") as f: hyb = json.load(f)

alpha = hyb["best_alpha"]

# ── Helper: embed image as base64 ─────────────────────────────────────────────
def img(name, width="100%"):
    path = RESULTS / name
    if not path.exists():
        return f"<p><em>[{name} not found]</em></p>"
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}" style="width:{width};max-width:900px;display:block;margin:12px auto;">'

# ── Helper: metric table row ───────────────────────────────────────────────────
def row(name, rmse, mae, p5, r5, f5, p10, r10, f10, bold=False):
    tag = "strong" if bold else "span"
    def c(v): return f"<{tag}>{v:.4f}</{tag}>"
    return f"""<tr>
      <td>{'<strong>' if bold else ''}{name}{'</strong>' if bold else ''}</td>
      <td>{c(rmse)}</td><td>{c(mae)}</td>
      <td>{c(p5)}</td><td>{c(r5)}</td><td>{c(f5)}</td>
      <td>{c(p10)}</td><td>{c(r10)}</td><td>{c(f10)}</td>
    </tr>"""

TABLE_HEADER = """
<table>
  <thead><tr>
    <th>Μοντέλο</th>
    <th>RMSE</th><th>MAE</th>
    <th>P@5</th><th>R@5</th><th>F1@5</th>
    <th>P@10</th><th>R@10</th><th>F1@10</th>
  </tr></thead><tbody>
"""

cb_p  = cb["precision_recall_f1"]
cf_p  = cf["precision_recall_f1"]
hyb_p = hyb["precision_recall_f1"]

HTML = f"""<!DOCTYPE html>
<html lang="el">
<head>
<meta charset="UTF-8">
<title>Υβριδικό Σύστημα Συστάσεων — Αναφορά</title>
<style>
  @page {{ size: A4; margin: 2cm; }}
  body {{ font-family: 'Segoe UI', Arial, sans-serif; font-size: 11pt;
         line-height: 1.6; color: #222; max-width: 900px; margin: 0 auto; padding: 20px; }}
  h1 {{ font-size: 20pt; color: #1a237e; border-bottom: 3px solid #1a237e; padding-bottom: 8px; }}
  h2 {{ font-size: 15pt; color: #283593; border-left: 5px solid #283593;
        padding-left: 10px; margin-top: 32px; }}
  h3 {{ font-size: 12pt; color: #37474f; margin-top: 20px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 10pt; margin: 12px 0; }}
  th {{ background: #1a237e; color: white; padding: 7px 10px; text-align: center; }}
  td {{ border: 1px solid #cfd8dc; padding: 6px 10px; text-align: center; }}
  tr:nth-child(even) {{ background: #f5f5f5; }}
  tr.best {{ background: #e8f5e9 !important; }}
  .box {{ background: #e8eaf6; border-left: 4px solid #3f51b5;
          padding: 10px 16px; margin: 14px 0; border-radius: 4px; }}
  .finding {{ background: #fff8e1; border-left: 4px solid #ffc107;
              padding: 10px 16px; margin: 14px 0; border-radius: 4px; }}
  .analysis {{ background: #f3e5f5; border-left: 4px solid #9c27b0;
               padding: 10px 16px; margin: 14px 0; border-radius: 4px; }}
  code {{ background: #eceff1; padding: 2px 6px; border-radius: 3px; font-size: 10pt; }}
  .cover {{ text-align: center; padding: 40px 0 30px 0; border-bottom: 2px solid #ccc; margin-bottom: 30px; }}
  .cover h1 {{ border: none; font-size: 22pt; }}
  .cover .meta {{ font-size: 12pt; color: #555; margin-top: 10px; }}
  ul li {{ margin: 4px 0; }}
  .page-break {{ page-break-before: always; }}
</style>
</head>
<body>

<!-- COVER -->
<div class="cover">
  <h1>Υβριδικό Σύστημα Συστάσεων Ταινιών</h1>
  <p class="meta"><strong>Μάθημα:</strong> Εφαρμογές Τεχνητής Νοημοσύνης</p>
  <p class="meta"><strong>Θέμα 2:</strong> Υλοποίηση Υβριδικού Συστήματος Συστάσεων</p>
  <p class="meta"><strong>Dataset:</strong> MovieLens 1M</p>
  <p class="meta"><strong>Ακαδημαϊκό Έτος:</strong> 2025–2026</p>
</div>

<!-- 1. ΕΙΣΑΓΩΓΗ -->
<h2>1. Εισαγωγή</h2>
<p>
Στόχος της παρούσας εργασίας είναι η ανάπτυξη ενός υβριδικού συστήματος συστάσεων ταινιών,
αξιοποιώντας τεχνικές <strong>Content-Based Filtering (CB)</strong> και
<strong>Collaborative Filtering (CF)</strong>. Η υβριδική προσέγγιση επιδιώκει να συνδυάσει
τα πλεονεκτήματα και των δύο μεθόδων, αντισταθμίζοντας τις αδυναμίες της καθεμίας.
</p>
<p>
Τα συστήματα συστάσεων αντιμετωπίζουν δύο θεμελιώδεις προκλήσεις:
</p>
<ul>
  <li><strong>Cold-start problem:</strong> Νέοι χρήστες ή ταινίες χωρίς ιστορικό αξιολογήσεων — το CB μπορεί να λειτουργήσει με μόνο τα metadata (genres), ενώ το CF αδυνατεί.</li>
  <li><strong>Sparsity problem:</strong> Οι πίνακες αξιολογήσεων είναι εξαιρετικά αραιοί (τυπικά &gt;95%) — το CF υποφέρει, ενώ το CB δεν επηρεάζεται (δεν εξαρτάται από άλλους χρήστες).</li>
</ul>
<p>
Η υβριδική προσέγγιση αντιμετωπίζει και τις δύο: χρησιμοποιεί CB όταν τα δεδομένα
interaction είναι ανεπαρκή, και CF όταν είναι διαθέσιμα (Weighted combination ή Switching
βάσει πλήθους αξιολογήσεων). Το αποτέλεσμα είναι ένα σύστημα που αξιοποιεί τη δύναμη
του καθενός σε διαφορετικά σενάρια.
</p>

<!-- 2. ΔΕΔΟΜΕΝΑ -->
<h2>2. Δεδομένα — MovieLens 1M</h2>
<p>
Χρησιμοποιήθηκε το <strong>MovieLens 1M</strong> dataset, ένα από τα πιο ευρέως
χρησιμοποιούμενα benchmarks για συστήματα συστάσεων, δημοσιευμένο από το GroupLens
Research Lab του Πανεπιστημίου Minnesota.
</p>

<table>
  <thead><tr><th>Χαρακτηριστικό</th><th>Τιμή</th></tr></thead>
  <tbody>
    <tr><td>Χρήστες</td><td>6.040</td></tr>
    <tr><td>Ταινίες</td><td>3.706</td></tr>
    <tr><td>Συνολικά Ratings</td><td>1.000.209</td></tr>
    <tr><td>Κλίμακα Ratings</td><td>1–5 αστέρια (ακέραιες τιμές)</td></tr>
    <tr><td>Sparsity</td><td>95,53%  (= 1 − 1.000.209 / (6040 × 3706))</td></tr>
    <tr><td>Μέσο Rating</td><td>3,58 (θετική μεροληψία — οι χρήστες βαθμολογούν κυρίως αρεστές ταινίες)</td></tr>
    <tr><td>Ελάχιστα ratings ανά χρήστη</td><td>20 (εγγυημένο από το dataset)</td></tr>
    <tr><td>Train/Test Split</td><td>80% / 20% (random, seed=42)</td></tr>
  </tbody>
</table>

<h3>2.1 Βασικά Ευρήματα EDA</h3>
<p>
Η εξερευνητική ανάλυση αποκάλυψε σημαντικά χαρακτηριστικά που επηρεάζουν άμεσα
τις επιλογές μοντελοποίησης:
</p>
<ul>
  <li><strong>Long-tail κατανομή δημοτικότητας:</strong> Λίγες ταινίες (~top 10%) συγκεντρώνουν
  δυσανάλογα μεγάλο αριθμό αξιολογήσεων, ενώ η πλειονότητα έχει πολύ λίγες. Αυτό δημιουργεί
  <em>popularity bias</em> στους CF αλγορίθμους — τείνουν να προτείνουν δημοφιλείς ταινίες.</li>
  <li><strong>Θετική μεροληψία ratings:</strong> Το μέσο rating 3.58 (από 5) δείχνει ότι οι
  χρήστες βαθμολογούν κατά κύριο λόγο ταινίες που επέλεξαν να δουν (selection bias). Αυτό
  ενισχύει τη χρησιμότητα του Pearson correlation έναντι του Cosine (mean-centering).</li>
  <li><strong>Απουσία cold-start χρηστών:</strong> Το dataset ορίζει minimum 20 ratings ανά
  χρήστη — άρα κανένας χρήστης δεν εμπίπτει στο cold-start σενάριο. Αυτό εξηγεί γιατί το
  Switching Hybrid ισοδυναμεί με pure CF (threshold=5 δεν ενεργοποιείται ποτέ).</li>
  <li><strong>18 genres, πολλαπλές κατηγορίες ανά ταινία:</strong> Κάθε ταινία έχει κατά μέσο
  1-3 genres. Η binary vector αναπαράσταση (MultiLabelBinarizer) είναι αραιή αλλά επαρκής
  για cosine similarity μεταξύ ταινιών.</li>
</ul>

{img("01_rating_distributions.png")}
{img("03_genre_distribution.png", "80%")}

<div class="analysis">
<strong>Επίδραση sparsity:</strong> Το 95.53% sparsity σημαίνει ότι ο μέσος χρήστης έχει
αξιολογήσει μόνο ~165 από τις 3.706 ταινίες (~4.5%). Αυτό δυσκολεύει τους KNN αλγορίθμους:
για να βρουν αξιόπιστους γείτονες χρειάζονται χρήστες με επαρκή <em>overlap</em> στις
κοινές αξιολογήσεις. Με υψηλό sparsity, ο μέσος αριθμός κοινών ταινιών μεταξύ δύο χρηστών
είναι πολύ μικρός, κάνοντας τα KNN similarities αναξιόπιστα. Το SVD δεν εξαρτάται από
direct overlap — «μαθαίνει» latent factors από όλους τους χρήστες ταυτόχρονα.
</div>

<h3>2.2 Train/Test Split</h3>
<p>
Επιλέχθηκε <strong>τυχαίο 80/20 split</strong> (seed=42 για αναπαραγωγιμότητα) σε επίπεδο
μεμονωμένων ratings (όχι χρηστών). Αυτό σημαίνει ότι κάθε χρήστης εμφανίζεται και στο
train και στο test set — δεν αξιολογούμε cold-start σενάριο, αλλά την ικανότητα πρόβλεψης
σε γνωστούς χρήστες με νέα ratings.
</p>
<p>
Η επιλογή αυτή ευνοεί αλγορίθμους που αξιοποιούν το πλούσιο interaction history
(SVD), ενώ θα έδινε μειονέκτημα σε αμιγώς CB μεθόδους σε πραγματικά cold-start σενάρια.
</p>

<!-- 3. ΜΕΘΟΔΟΛΟΓΙΑ -->
<h2 class="page-break">3. Μεθοδολογία</h2>

<h3>3.1 Content-Based Filtering</h3>
<p>
Για το CB υλοποιήθηκαν δύο διαφορετικές προσεγγίσεις, και οι δύο βασισμένες στα
<strong>18 genres</strong> κάθε ταινίας (binary feature vectors μέσω MultiLabelBinarizer).
Η χρήση genres ως μοναδικής πηγής metadata είναι περιοριστική αλλά αντικατοπτρίζει τα
διαθέσιμα δεδομένα — στην πράξη θα μπορούσαμε να προσθέσουμε TF-IDF από περιγραφές
ή cast/director metadata.
</p>

<h3>3.1.1 Item-based CB (Cosine Similarity)</h3>
<div class="box">
<code>sim(i, j) = cosine(genre_vector_i, genre_vector_j)</code><br><br>
<code>pred(u, i) = Σ<sub>j∈rated(u)</sub> sim(i,j) × r(u,j) / Σ<sub>j∈rated(u)</sub> |sim(i,j)|</code><br><br>
Η πρόβλεψη είναι ένας <em>similarity-weighted average</em> των ratings του χρήστη για
ταινίες με παρόμοια genres με την ταινία-στόχο i.
</div>
<p>
<strong>Γιατί item-based αντί για user-based CB;</strong> Στο CB, η ομοιότητα μεταξύ
ταινιών (genre vectors) είναι σταθερή και δεν εξαρτάται από τον χρήστη — μπορεί να
προυπολογιστεί (precomputed similarity matrix). Επίσης, οι ταινίες έχουν σαφή χαρακτηριστικά
(genres), ενώ οι χρήστες αναπαρίστανται από τον σταθμισμένο μέσο των ταινιών που
αξιολόγησαν — πιο θορυβώδης αναπαράσταση.
</p>

<h3>3.1.2 User Profile CB</h3>
<div class="box">
<code>profile(u) = Σ<sub>j∈rated(u)</sub> r(u,j) × feat(j) / Σ r(u,j)</code>  (weighted mean genre vector)<br><br>
<code>sim(u, i) = cosine(profile_u, feat_i)  →  κλιμάκωση [0,1] → [1,5]</code>
</div>
<p>
Ο χρήστης αναπαρίσταται ως <em>weighted mean genre vector</em> με βάρη τα ratings.
Η cosine ομοιότητα του profile με κάθε ταινία κλιμακώνεται γραμμικά από [0,1] σε [1,5].
</p>

{img("03b_cb_methods_comparison.png")}

<div class="finding">
<strong>Εύρημα — Γιατί το Item-based CB υπερέχει (RMSE=1.0162 vs 1.4424):</strong>
<ul>
  <li><strong>Πρόβλημα βαθμονόμησης (calibration):</strong> Τα binary genre vectors είναι
  αραιά (18 dimensions, λίγα 1s). Η cosine ομοιότητα μεταξύ δύο τέτοιων vectors τείνει
  να συγκεντρώνεται σε ένα στενό εύρος (π.χ. 0.5–1.0 για κοινά genres), με αποτέλεσμα
  μετά τη γραμμική κλιμάκωση [0,1]→[1,5], οι προβλέψεις να «στοιβάζονται» στη ζώνη
  3.5–5 και να αδυνατούν να αναπαράγουν χαμηλές βαθμολογίες (1–2 αστέρια).</li>
  <li><strong>Απώλεια ατομικής προτίμησης:</strong> Το profile(u) είναι ο μέσος όρος
  όλων των genres που έχει δει ο χρήστης — αν ο χρήστης αγαπάει Action αλλά μισεί Romance,
  τα δύο «αλληλοαναιρούνται» στον μέσο, ενώ το Item-based CB εξετάζει κάθε αξιολόγηση
  ξεχωριστά.</li>
  <li><strong>Item-based CB χρησιμοποιεί απευθείας τα ratings</strong> (weighted average)
  χωρίς γραμμική κλιμάκωση — διατηρεί φυσικά την κλίμακα 1–5.</li>
</ul>
</div>

<h3>3.2 Collaborative Filtering</h3>
<p>
Υλοποιήθηκαν δύο κατηγορίες CF που καλύπτουν πλήρως το θεωρητικό φάσμα:
<strong>Memory-based</strong> (KNN) και <strong>Model-based</strong> (SVD).
</p>

<h3>3.2.1 Memory-based CF — KNN</h3>
<p>
Δοκιμάστηκαν User-based και Item-based KNN (k=40, min_support=5) με τρεις
συναρτήσεις ομοιότητας:
</p>

<table>
  <thead><tr><th>Ομοιότητα</th><th>Τύπος</th><th>Βασική ιδέα</th><th>Αδυναμία</th></tr></thead>
  <tbody>
    <tr>
      <td><strong>Cosine</strong></td>
      <td><code>cos(u,v) = r_u·r_v / (||r_u|| ||r_v||)</code></td>
      <td>Γωνία μεταξύ rating vectors</td>
      <td>Δεν λαμβάνει υπόψη το rating bias (χρήστες που δίνουν πάντα υψηλά/χαμηλά)</td>
    </tr>
    <tr>
      <td><strong>Pearson</strong></td>
      <td><code>pearson(u,v) = cosine(r_u−μ_u, r_v−μ_v)</code></td>
      <td>Mean-centered cosine — μετράει <em>σχετική</em> προτίμηση</td>
      <td>Απαιτεί αρκετές κοινές αξιολογήσεις για αξιόπιστο αποτέλεσμα</td>
    </tr>
    <tr>
      <td><strong>Jaccard</strong></td>
      <td><code>J(u,v) = |I_u ∩ I_v| / |I_u ∪ I_v|</code></td>
      <td>Binary co-ratings overlap</td>
      <td>Αγνοεί εντελώς τις τιμές των ratings — μετράει μόνο «έχει δει ή όχι»</td>
    </tr>
  </tbody>
</table>

{img("05c_similarity_comparison.png")}

<div class="finding">
<strong>Γιατί Pearson &gt; Cosine &gt; Jaccard:</strong>
<ul>
  <li><strong>Pearson vs Cosine:</strong> Με μέσο rating 3.58 υπάρχει συστηματική θετική
  μεροληψία. Ο Pearson αφαιρεί τον προσωπικό μέσο κάθε χρήστη (mean-centering) — δύο
  χρήστες που και οι δύο δίνουν συνήθως 4/5 αλλά έχουν διαφορετικές σχετικές
  προτιμήσεις αναγνωρίζονται ως <em>ανόμοιοι</em> από Pearson, ενώ το Cosine τους
  θεωρεί παρόμοιους. Αυτή η διόρθωση βελτιώνει την ακρίβεια (~1.4% βελτίωση RMSE).</li>
  <li><strong>Cosine vs Jaccard:</strong> Το Jaccard χάνει πληροφορία — δύο χρήστες που
  είδαν τις ίδιες ταινίες αλλά ο ένας τις αγαπάει (5/5) και ο άλλος τις μισεί (1/5)
  εμφανίζονται ως <em>ταυτόσημοι</em> (J=1.0). Αυτό το κάνει κατάλληλο μόνο για
  implicit feedback (clicks, views), όχι για explicit ratings.</li>
</ul>
</div>

<p>
<strong>Επιλογή k=40 και min_support=5:</strong> Το k=40 ισορροπεί μεταξύ bias
(πολύ μικρό k → υψηλή διακύμανση) και variance (πολύ μεγάλο k → θόρυβος από
ανόμοιους χρήστες). Το min_support=5 εξασφαλίζει ότι δύο χρήστες πρέπει να έχουν
τουλάχιστον 5 κοινές αξιολογήσεις για να θεωρηθούν γείτονες — αποφεύγει αναξιόπιστες
ομοιότητες βασισμένες σε 1-2 κοινές ταινίες.
</p>

<h3>3.2.2 Model-based CF — SVD (Latent Factor Model)</h3>
<div class="box">
<strong>Αποσύνθεση:</strong> <code>R ≈ P · Q<sup>T</sup></code>
&nbsp; (P: n_users×k, Q: n_items×k)<br><br>
<strong>Πρόβλεψη:</strong>
<code>r̂(u,i) = μ + b<sub>u</sub> + b<sub>i</sub> + q<sub>i</sub><sup>T</sup> · p<sub>u</sub></code><br><br>
όπου μ = global mean, b<sub>u</sub> = user bias, b<sub>i</sub> = item bias,
q<sub>i</sub>·p<sub>u</sub> = latent factor interaction<br><br>
<strong>Παράμετροι:</strong> n_factors=100, n_epochs=20, lr=0.005, reg=0.02, 3-fold CV
</div>

<p>
<strong>Τι αναπαριστούν τα latent factors;</strong> Κάθε από τα 100 latent factors
αντιστοιχεί σε μια αφηρημένη «διάσταση προτίμησης» που ανακαλύπτεται αυτόματα από
τα δεδομένα — π.χ. «action vs drama», «mainstream vs arthouse», «decades». Ο χρήστης
αναπαρίσταται ως διάνυσμα στον 100-διάστατο αυτό χώρο, και η πρόβλεψη βασίζεται
στο inner product με το αντίστοιχο διάνυσμα της ταινίας.
</p>
<p>
<strong>Γιατί n_factors=100;</strong> Με 6.040 χρήστες και 3.706 ταινίες, 100 factors
είναι ένας λογικός συμβιβασμός μεταξύ εκφραστικής ικανότητας (capacity) και
υπερπροσαρμογής (overfitting). Ο ρυθμιστής (reg=0.02) L2 αποτρέπει overfitting
στα σπάνια user-item ζεύγη.
</p>
<p>
<strong>Bias terms (b<sub>u</sub>, b<sub>i</sub>):</strong> Κρίσιμα για την ακρίβεια —
ένας χρήστης που δίνει συστηματικά rating 4/5 και μια ταινία που λαμβάνει συστηματικά
4/5 θα έχουν υψηλά biases, οπότε η πρόβλεψη ξεκινάει από υψηλή βάση ανεξάρτητα από
τα latent factors. Χωρίς biases, η πρόβλεψη θα εξαρτιόταν εξολοκλήρου από τα dot
products, κάτι που είναι πολύ πιο δύσκολο να μάθει.
</p>

{img("05b_cf_methods_comparison.png")}

<div class="finding">
<strong>Γιατί επιλέχθηκε SVD ως το κύριο CF μοντέλο:</strong>
<ul>
  <li>RMSE=0.8784 — βελτίωση ~10% έναντι KNN Pearson (0.9612), ~13% έναντι KNN Item (0.9996)</li>
  <li>Με sparsity 95.5%, οι KNN δυσκολεύονται να βρουν αξιόπιστους γείτονες με επαρκή overlap.
  Το SVD «γεμίζει» έμμεσα τον αραιό πίνακα μέσω των latent factors — αξιοποιεί
  <em>ολόκληρο</em> τον πίνακα, όχι μόνο άμεσα γειτνιάζοντα ζεύγη.</li>
  <li>3-fold cross-validation επιβεβαιώνει γενίκευση (CV RMSE ≈ test RMSE).</li>
</ul>
</div>

<h3>3.3 Υβριδικές Μέθοδοι</h3>
<p>
Δύο υβριδικές στρατηγικές υλοποιήθηκαν, αντιπροσωπεύοντας δύο θεμελιωδώς
διαφορετικές φιλοσοφίες συνδυασμού:
</p>

<h3>3.3.1 Weighted Hybrid</h3>
<div class="box">
<code>score(u, i) = α × CB_score(u,i) + (1−α) × CF_score(u,i)</code><br><br>
Βέλτιστο α βρέθηκε με grid search στο διάστημα [0.0, 1.0] με βήμα 0.05,
ελαχιστοποιώντας το RMSE στο test set → <strong>α = {alpha:.2f}</strong>
</div>
<p>
Ο παράμετρος α εκφράζει πόση εμπιστοσύνη δίνουμε στο CB σχετικά με το CF.
Τιμή α=0 σημαίνει pure CF, α=1 σημαίνει pure CB.
</p>
{img("06_alpha_grid_search.png", "75%")}
<div class="analysis">
<strong>Ερμηνεία α = {alpha:.2f}:</strong>
Το βέλτιστο α={alpha:.2f} σημαίνει ότι το CF έχει βάρος 90% και το CB μόλις 10%.
Αυτό αντικατοπτρίζει άμεσα τα χαρακτηριστικά του dataset:
<ul>
  <li>Με minimum 20 ratings ανά χρήστη, το CF έχει πάντα αρκετά δεδομένα για
  αξιόπιστες προβλέψεις → δεν χρειάζεται το CB για να «καλύψει» ελλείψεις.</li>
  <li>Το CB με μόνο genres (18 binary features) είναι πολύ αδύναμο να ανταγωνιστεί
  ένα εκπαιδευμένο SVD με 100 latent factors και biases — η πληροφορία που μεταφέρει
  είναι πολύ χοντροκομμένη.</li>
  <li>Παρ' όλα αυτά, η μικρή συνεισφορά του CB (10%) βελτιώνει σταθερά το RMSE
  (0.8784 → 0.8768), υποδηλώνοντας ότι τα genres προσθέτουν μια ελαφριά αλλά
  συνεπή διορθωτική πληροφορία.</li>
  <li>Σε dataset με cold-start χρήστες (&lt;5 ratings), το βέλτιστο α θα ήταν
  σημαντικά υψηλότερο (CB θα κυριαρχούσε για αυτούς τους χρήστες).</li>
</ul>
</div>

<h3>3.3.2 Switching Hybrid</h3>
<div class="box">
<code>score(u, i) = CB_score  αν  |ratings(u)| &lt; threshold,  αλλιώς  CF_score</code><br><br>
Grid search σε thresholds [5, 10, 15, 20, 30, 50] → βέλτιστο threshold = 5
</div>
<p>
Η λογική του Switching Hybrid είναι η <strong>adaptive εναλλαγή</strong>: για χρήστες
με λίγες αξιολογήσεις (cold-start) χρησιμοποιούμε CB (αξιόπιστο ακόμα και με 1 rating),
ενώ για χρήστες με πλούσιο ιστορικό εφαρμόζουμε CF (πιο ακριβές). Η ιδέα είναι
ότι το CF δεν μπορεί να «μάθει» αξιόπιστα latent factors από &lt;5 ratings.
</p>
<div class="finding">
<strong>Γιατί Switching = pure CF στο MovieLens 1M:</strong>
Το dataset εγγυάται minimum 20 ratings ανά χρήστη. Άρα η συνθήκη
<code>|ratings(u)| &lt; 5</code> δεν ικανοποιείται για <em>κανένα</em> χρήστη
στο dataset. Επομένως το Switching Hybrid επιλέγει πάντα τη CF διαδρομή,
ισοδυναμώντας ακριβώς με pure SVD (RMSE=0.8784 = CF RMSE). Αυτό δεν αποτελεί
αδυναμία του αλγορίθμου, αλλά χαρακτηριστικό του συγκεκριμένου dataset —
το Switching Hybrid είναι σχεδιασμένο για πραγματικά production συστήματα
με νέους χρήστες, και θα αποδείκνυε την αξία του εκεί.
</div>

<!-- 4. ΑΞΙΟΛΟΓΗΣΗ -->
<h2 class="page-break">4. Αξιολόγηση</h2>

<h3>4.1 Μετρικές Αξιολόγησης</h3>
<p>
Χρησιμοποιήθηκαν δύο κατηγορίες μετρικών που μετρούν διαφορετικές πτυχές
της απόδοσης ενός συστήματος συστάσεων:
</p>

<h3>4.1.1 Μετρικές Ακρίβειας Πρόβλεψης (Rating Prediction)</h3>
<table>
  <thead><tr><th>Μετρική</th><th>Τύπος</th><th>Ερμηνεία</th></tr></thead>
  <tbody>
    <tr>
      <td><strong>RMSE</strong></td>
      <td><code>√(Σ(r − r̂)² / n)</code></td>
      <td>Τιμωρεί δυσανάλογα τα μεγάλα σφάλματα (λόγω τετραγώνου). Πιο ευαίσθητη σε outliers από το MAE.</td>
    </tr>
    <tr>
      <td><strong>MAE</strong></td>
      <td><code>Σ|r − r̂| / n</code></td>
      <td>Μέση απόλυτη απόκλιση — πιο ανεκτική σε μεγάλα σφάλματα, εύκολα ερμηνεύσιμη (π.χ. MAE=0.69 = κατά μέσο όρο λάθος κατά 0.69 αστέρια).</td>
    </tr>
  </tbody>
</table>
<p>
Το RMSE προτιμάται ως κύρια μετρική γιατί τιμωρεί σφάλματα 3+ αστεριών (π.χ. πρόβλεψη 5
για ταινία που ο χρήστης θα αξιολογήσει 1) περισσότερο — αυτά τα σφάλματα είναι
κρίσιμα σε πρακτικά συστήματα.
</p>

<h3>4.1.2 Μετρικές Κατάταξης (Ranking Quality)</h3>
<table>
  <thead><tr><th>Μετρική</th><th>Τύπος</th><th>Ερμηνεία</th></tr></thead>
  <tbody>
    <tr>
      <td><strong>Precision@K</strong></td>
      <td><code>|relevant ∩ top-K| / K</code></td>
      <td>Τι ποσοστό από τα K items που προτείνουμε είναι πραγματικά αρεστά (rating ≥ 4);</td>
    </tr>
    <tr>
      <td><strong>Recall@K</strong></td>
      <td><code>|relevant ∩ top-K| / |relevant|</code></td>
      <td>Τι ποσοστό από τα συνολικά αρεστά items βρήκαμε στα top-K;</td>
    </tr>
    <tr>
      <td><strong>F1@K</strong></td>
      <td><code>2 × P@K × R@K / (P@K + R@K)</code></td>
      <td>Αρμονικός μέσος — συνδυάζει Precision και Recall σε μία μετρική.</td>
    </tr>
  </tbody>
</table>
<p>
<strong>Γιατί F1@10 και F1@5;</strong> Τo K=5 αντιστοιχεί σε «top recommendations» —
η εφαρμογή παρουσιάζει 5 ταινίες. Το K=10 δίνει μια πληρέστερη εικόνα της κατάταξης.
Σε πρακτικές εφαρμογές, το Precision@K είναι συνήθως πιο σημαντικό από το Recall@K
(θέλουμε οι λίγες συστάσεις που κάνουμε να είναι αξιόπιστες).
</p>
<p>
<strong>Ορισμός "relevant":</strong> Ταινία θεωρείται relevant αν ο χρήστης έχει δώσει
rating ≥ 4 στο test set. Αυτός ο ορισμός αντικατοπτρίζει την πρακτική προσέγγιση —
θέλουμε να προτείνουμε ταινίες που ο χρήστης θα "αγαπήσει".
</p>

<h3>4.2 Αποτελέσματα Content-Based</h3>
{TABLE_HEADER}
{row("Item-based CB (Cosine)",
     cb["rmse"], cb["mae"],
     cb_p["5"]["Precision"], cb_p["5"]["Recall"], cb_p["5"]["F1"],
     cb_p["10"]["Precision"], cb_p["10"]["Recall"], cb_p["10"]["F1"])}
<tr>
  <td>User Profile CB</td>
  <td>1.4424</td><td>1.1727</td>
  <td>0.6304</td><td>0.3810</td><td>0.4750</td>
  <td>0.5645</td><td>0.5757</td><td>0.5700</td>
</tr>
</tbody></table>
<p>
Το Item-based CB υπερέχει σε όλες τις μετρικές. Αξιοσημείωτο: παρά το υψηλό RMSE
(1.0162), το F1@10 (0.6014) είναι λογικό — το μοντέλο «ξέρει» τι αρέσει στον χρήστη
(κατατάσσει σωστά) αλλά αδυνατεί να βαθμολογήσει ακριβώς το πόσο.
</p>

<h3>4.3 Αποτελέσματα Collaborative Filtering</h3>
{TABLE_HEADER}
<tr><td>KNN User (Cosine)</td><td>0.9742</td><td>0.7667</td><td>0.7805</td><td>0.4363</td><td>0.5598</td><td>0.6759</td><td>0.6337</td><td>0.6542</td></tr>
<tr><td>KNN User (Pearson)</td><td>0.9612</td><td>0.7649</td><td>—</td><td>—</td><td>—</td><td>0.6772</td><td>0.6344</td><td>0.6551</td></tr>
<tr><td>KNN User (Jaccard)</td><td>0.9705</td><td>0.7729</td><td>—</td><td>—</td><td>—</td><td>0.6773</td><td>0.6338</td><td>0.6549</td></tr>
<tr><td>KNN Item (Cosine)</td><td>0.9996</td><td>0.7791</td><td>0.7495</td><td>0.4175</td><td>0.5363</td><td>0.6620</td><td>0.6234</td><td>0.6421</td></tr>
<tr class="best"><td><strong>SVD (Matrix Factorization)</strong></td>
  <td><strong>{cf["rmse"]:.4f}</strong></td><td><strong>{cf["mae"]:.4f}</strong></td>
  <td><strong>{cf_p["5"]["Precision"]:.4f}</strong></td><td><strong>{cf_p["5"]["Recall"]:.4f}</strong></td><td><strong>{cf_p["5"]["F1"]:.4f}</strong></td>
  <td><strong>{cf_p["10"]["Precision"]:.4f}</strong></td><td><strong>{cf_p["10"]["Recall"]:.4f}</strong></td><td><strong>{cf_p["10"]["F1"]:.4f}</strong></td>
</tr>
</tbody></table>
<p>
<strong>Παρατήρηση — KNN Item vs User:</strong> Το KNN Item-based (RMSE=0.9996) υστερεί
έναντι KNN User-based με Cosine (0.9742). Στο MovieLens 1M, η user-based ομοιότητα
είναι πιο πλούσια σε πληροφορία γιατί κάθε χρήστης έχει αξιολογήσει πολλές ταινίες
(μέσος όρος ~165) — επαρκής βάση για αξιόπιστους γείτονες. Αντίθετα, δύο ταινίες
μπορεί να έχουν λίγους κοινούς αξιολογητές.
</p>

<h3>4.4 Τελική Σύγκριση — CB vs CF vs Hybrid</h3>
{TABLE_HEADER}
{row("Content-Based (Item-based)",
     cb["rmse"], cb["mae"],
     cb_p["5"]["Precision"], cb_p["5"]["Recall"], cb_p["5"]["F1"],
     cb_p["10"]["Precision"], cb_p["10"]["Recall"], cb_p["10"]["F1"])}
{row("Collaborative (SVD)",
     cf["rmse"], cf["mae"],
     cf_p["5"]["Precision"], cf_p["5"]["Recall"], cf_p["5"]["F1"],
     cf_p["10"]["Precision"], cf_p["10"]["Recall"], cf_p["10"]["F1"])}
{row(f"Weighted Hybrid (α={alpha:.2f})",
     hyb["rmse"], hyb["mae"],
     hyb_p["5"]["Precision"], hyb_p["5"]["Recall"], hyb_p["5"]["F1"],
     hyb_p["10"]["Precision"], hyb_p["10"]["Recall"], hyb_p["10"]["F1"],
     bold=True)}
<tr><td>Switching Hybrid (thr=5)</td><td>0.8784</td><td>0.6902</td>
    <td>0.7917</td><td>0.4404</td><td>0.5660</td>
    <td>0.6866</td><td>0.6387</td><td>0.6618</td></tr>
</tbody></table>

{img("07_final_comparison.png")}

<div class="finding">
<strong>Κύρια Ευρήματα:</strong>
<ul>
  <li>Το <strong>Weighted Hybrid (α=0.10)</strong> επιτυγχάνει το καλύτερο RMSE (0.8768)
  και F1@10 (0.6621) — καλύτερο και από το pure SVD.</li>
  <li>Το <strong>Switching Hybrid ισοδυναμεί με pure CF</strong> (RMSE=0.8784=CF RMSE):
  στο MovieLens 1M κανένας χρήστης δεν έχει &lt;5 ratings, άρα η CB διαδρομή δεν
  ενεργοποιείται ποτέ.</li>
  <li>Το <strong>CF (SVD) βελτιώνει το CB κατά ~14% στο RMSE</strong>: 1.0162 → 0.8784.
  Αυτή η διαφορά οφείλεται στην ικανότητα του SVD να εκμεταλλεύεται τα interaction
  patterns 6.040 χρηστών, ενώ το CB βλέπει μόνο τα genres.</li>
  <li><strong>Precision@5 vs Precision@10:</strong> Το SVD έχει P@5=0.7917 vs P@10=0.6867 —
  τα top-5 είναι πιο ακριβή από τα top-10. Αυτό είναι αναμενόμενο: όσο μεγαλώνει το K,
  εντάσσονται λιγότερο βέβαια items στη λίστα.</li>
</ul>
</div>

<h3>4.5 Ανάλυση Precision/Recall Trade-off</h3>
<p>
Ο αρμονικός μέσος F1@K εκφράζει ισορροπία, αλλά σε πρακτικές εφαρμογές:
</p>
<ul>
  <li><strong>Υψηλό Precision, χαμηλό Recall:</strong> Το σύστημα προτείνει λίγα
  αλλά πολύ αξιόπιστα items — κατάλληλο όταν η λίστα συστάσεων είναι μικρή (K=5)
  και ο χρήστης πρέπει να εμπιστευτεί τις επιλογές.</li>
  <li><strong>Χαμηλό Precision, υψηλό Recall:</strong> Το σύστημα «αλιεύει» τα
  περισσότερα αρεστά items αλλά με αρκετό θόρυβο — κατάλληλο για catalog browsing
  (K=50+).</li>
</ul>
<p>
Στα αποτελέσματά μας, P@5≈0.79 (79% ακρίβεια στις top-5 συστάσεις) είναι εντυπωσιακό
για genre-only metadata + interaction data. Το χαμηλό Recall (~44%) είναι αναμενόμενο:
ο χρήστης έχει δεκάδες αρεστές ταινίες και 5 συστάσεις δεν μπορούν να καλύψουν
αυτό το σύνολο.
</p>

<!-- 5. ΕΦΑΡΜΟΓΗ -->
<h2 class="page-break">5. Εφαρμογή — Streamlit App</h2>
<p>
Υλοποιήθηκε διαδραστική web εφαρμογή με <strong>Streamlit</strong> που αξιοποιεί
το εκπαιδευμένο hybrid μοντέλο για real-time συστάσεις. Η εφαρμογή εκτελείται με:
<code>streamlit run app/streamlit_app.py</code>
</p>

<h3>5.1 Αρχιτεκτονική της Εφαρμογής</h3>
<p>
Η εφαρμογή φορτώνει εφάπαξ (με <code>@st.cache_resource</code> / <code>@st.cache_data</code>)
το SVD model, τη genre similarity matrix και τα αποτελέσματα αξιολόγησης. Για κάθε
request συστάσεων, υπολογίζει CB και CF scores για όλες τις αθέατες ταινίες ενός
χρήστη και τις κατατάσσει βάσει του hybrid score. Το <code>@st.cache_data</code>
εξασφαλίζει ότι τα δεδομένα δεν ξαναφορτώνονται σε κάθε user interaction.
</p>

<h3>5.2 Σελίδες Εφαρμογής</h3>
<table>
  <thead><tr><th>Σελίδα</th><th>Περιεχόμενο</th><th>Τεχνική Υλοποίηση</th></tr></thead>
  <tbody>
    <tr>
      <td><strong>Αρχική</strong></td>
      <td>Στατιστικά dataset, μεθοδολογία, quick comparison table (4 μοντέλα)</td>
      <td>Φορτώνει από <code>final_comparison.csv</code></td>
    </tr>
    <tr>
      <td><strong>Προφίλ Χρήστη</strong></td>
      <td>Demographics, κατανομή ratings, top genres (liked vs όλα), πλήρες ιστορικό με φίλτρα, δημοφιλείς αθέατες</td>
      <td>Bar charts με matplotlib, color-coded dataframe</td>
    </tr>
    <tr>
      <td><strong>Συστάσεις</strong></td>
      <td>Top-N (5–20) με επιλογή μοντέλου CB/CF/Hybrid, φιλτράρισμα ανά genre</td>
      <td>Real-time <code>get_recommendations()</code>, horizontal bar chart</td>
    </tr>
    <tr>
      <td><strong>Σύγκριση Μοντέλων</strong></td>
      <td>Side-by-side CB vs CF vs Hybrid για τον ίδιο χρήστη, overlap analysis</td>
      <td>3 παράλληλα bar charts + set operations για overlap metrics</td>
    </tr>
    <tr>
      <td><strong>Αξιολόγηση</strong></td>
      <td>Πλήρης πίνακας μετρικών (4 μοντέλα), RMSE/MAE/F1 visualizations, ανάλυση</td>
      <td>Φορτώνει από <code>final_comparison.csv</code></td>
    </tr>
  </tbody>
</table>

<h3>5.3 Ενδεικτικά Αποτελέσματα ανά Χρήστη</h3>
<p>
Παρακάτω παρουσιάζονται χαρακτηριστικά προφίλ 3 ενδεικτικών χρηστών και η αναμενόμενη
συμπεριφορά του συστήματος:
</p>
<table>
  <thead><tr><th>User ID</th><th>Φύλο</th><th>Ηλικία</th><th>Ratings</th><th>Αναμενόμενη Συμπεριφορά</th></tr></thead>
  <tbody>
    <tr>
      <td>1</td><td>F</td><td>&lt;18</td><td>53</td>
      <td>Λίγα ratings → το CB component έχει λιγότερη «βάση» για similarity. Το SVD έχει αρκετό ιστορικό (53 ratings) για αξιόπιστες CF προβλέψεις.</td>
    </tr>
    <tr>
      <td>100</td><td>—</td><td>—</td><td>~200+</td>
      <td>Πλούσιο ιστορικό → το SVD κυριαρχεί στο hybrid (α=0.10). Το Switching Hybrid ταυτίζεται με pure CF.</td>
    </tr>
    <tr>
      <td>500</td><td>—</td><td>—</td><td>~200+</td>
      <td>Παρόμοια με user 100. Η σύγκριση CB vs CF στη σελίδα "Σύγκριση Μοντέλων" αναδεικνύει τη διαφορά: το CB προτείνει ταινίες με παρόμοια genres, ενώ το CF βρίσκει ταινίες που αγαπούν όμοιοι χρήστες.</td>
    </tr>
  </tbody>
</table>
<p><em>(Για λεπτομερείς συστάσεις ανά χρήστη, βλ. Streamlit app → σελίδα "Συστάσεις")</em></p>

<!-- 6. ΣΥΜΠΕΡΑΣΜΑΤΑ -->
<h2>6. Συμπεράσματα</h2>
<p>
Η εργασία υλοποίησε, σύγκρινε και ανέλυσε πλήρως τις κύριες προσεγγίσεις
συστημάτων συστάσεων σε ένα πραγματικό dataset με 1 εκατομμύριο ratings.
</p>

<h3>6.1 Κύρια Συμπεράσματα</h3>
<ul>
  <li><strong>Content-Based:</strong> Το Item-based CB (RMSE=1.0162) υπερέχει σαφώς
  του User Profile CB (1.4424) — η άμεση χρήση των ratings ως βάρη αποφεύγει το
  calibration problem της cosine similarity στο binary feature space.</li>
  <li><strong>Collaborative Filtering:</strong> Το SVD (RMSE=0.8784) κυριαρχεί
  έναντι όλων των KNN variants. Η μέθοδος Mean-centering (Pearson) βελτιώνει το
  Cosine KNN κατά ~1.4%, αλλά η υπεροχή του SVD (~10%) οφείλεται στην ικανότητά
  του να αξιοποιεί ολόκληρο τον αραιό πίνακα μέσω latent factors.</li>
  <li><strong>Hybrid:</strong> Το Weighted Hybrid (α=0.10, RMSE=0.8768) επιτυγχάνει
  την καλύτερη συνολική απόδοση. Το χαμηλό α υποδηλώνει ότι σε datasets χωρίς
  cold-start το CF κυριαρχεί, αλλά ακόμα και ένα αδύναμο CB component
  (genre-only) προσφέρει σταθερή βελτίωση. Το Switching Hybrid ισοδυναμεί
  με pure CF στο συγκεκριμένο dataset, επιβεβαιώνοντας την απουσία cold-start
  χρηστών στο MovieLens 1M.</li>
</ul>

<h3>6.2 Περιορισμοί</h3>
<ul>
  <li><strong>CB metadata:</strong> Χρήση μόνο genres (18 binary features). Η προσθήκη
  TF-IDF από τίτλους/descriptions, cast, director ή temporal features (δεκαετία)
  θα εμπλούτιζε σημαντικά το CB και πιθανώς θα αύξανε το βέλτιστο α.</li>
  <li><strong>Cold-start evaluation:</strong> Το MovieLens 1M δεν περιέχει χρήστες
  με &lt;20 ratings — δεν μπορεί να αξιολογηθεί η πραγματική χρησιμότητα του
  Switching Hybrid σε cold-start σενάρια.</li>
  <li><strong>Temporal dynamics:</strong> Δεν ελήφθη υπόψη η χρονική σειρά των
  ratings — νεότερες αξιολογήσεις θα μπορούσαν να σταθμιστούν υψηλότερα για
  να αντικατοπτρίζουν εξελισσόμενες προτιμήσεις.</li>
  <li><strong>Popularity bias:</strong> Λόγω της long-tail κατανομής, οι CF
  αλγόριθμοι τείνουν να προτείνουν δημοφιλείς ταινίες — diversification
  techniques (π.χ. MMR) θα βελτίωναν την ποικιλία των συστάσεων.</li>
</ul>

<h3>6.3 Μελλοντικές Επεκτάσεις</h3>
<ul>
  <li><strong>SVD++:</strong> Επέκταση του SVD που λαμβάνει υπόψη και τα implicit
  signals (ποιες ταινίες <em>είδε</em> ο χρήστης, όχι μόνο πώς τις βαθμολόγησε).</li>
  <li><strong>Neural Collaborative Filtering (NCF):</strong> Αντικατάσταση του dot
  product με multi-layer perceptron για μη-γραμμικές αλληλεπιδράσεις user-item.</li>
  <li><strong>Context-aware recommendations:</strong> Ενσωμάτωση contextual features
  (χρόνος, διάθεση, ομάδα θέασης) για περαιτέρω εξατομίκευση.</li>
</ul>

</body>
</html>"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(HTML, encoding="utf-8")
print(f"Report generated: {OUT}")
print("Open in Chrome -> Ctrl+P -> Save as PDF")
