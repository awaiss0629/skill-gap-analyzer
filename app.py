import streamlit as st
import pandas as pd
import numpy as np
import re
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Skill Gap Analyzer", layout="wide")
st.title("🧠 Skill Gap Analysis — TF-IDF + K-Means")
st.caption("Upload: Intern_Skills.csv, Job_Descriptions.csv (optional: Skills_Ontology.csv)")

with st.expander("Required CSV schema", expanded=False):
    st.markdown(
        """
        **Intern_Skills.csv** → `intern_id`, `name`, `skills_text`  
        **Job_Descriptions.csv** → `job_id`, `title`, `description`  
        **Skills_Ontology.csv (optional)** → `skill_name`, `category`, `synonyms` (semicolon-separated)
        """
    )

# -----------------------------
# Upload inputs
# -----------------------------
c1, c2, c3 = st.columns(3)
with c1:
    intern_file = st.file_uploader("Intern_Skills.csv", type=["csv"])
with c2:
    jobs_file = st.file_uploader("Job_Descriptions.csv", type=["csv"])
with c3:
    onto_file = st.file_uploader("Skills_Ontology.csv (optional)", type=["csv"])

# -----------------------------
# Helpers
# -----------------------------
def norm_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9+#/\.\- ,]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def build_synonym_map(onto_df: pd.DataFrame | None) -> dict:
    synmap = {}
    if onto_df is None or onto_df.empty:
        return synmap
    for _, row in onto_df.iterrows():
        base = norm_text(row.get("skill_name", ""))
        syns = norm_text(row.get("synonyms", "")).split(";")
        keys = [base] + [s.strip() for s in syns if s.strip()]
        for k in keys:
            if k:
                synmap[k] = base if base else k
    return synmap

def normalize_with_synonyms(text_series: pd.Series, synonym_map: dict) -> pd.Series:
    if not synonym_map:
        return text_series.astype(str).map(norm_text)

    def _map_text(t: str) -> str:
        txt = norm_text(t)
        # If short comma-separated list, split on commas; else by whitespace
        tokens = [x.strip() for x in txt.split(",")] if ("," in txt and len(txt) < 400) else txt.split()
        mapped = []
        for tok in tokens:
            k = synonym_map.get(norm_text(tok), norm_text(tok))
            if k:
                mapped.append(k)
        return " ".join(sorted(set(mapped))) if mapped else txt

    return text_series.astype(str).map(_map_text)

def fit_tfidf(corpus: pd.Series):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    X = vec.fit_transform(corpus)
    return vec, X

# IMPORTANT: Do NOT cache this on sparse matrices
def compute_similarity(_X_intern, _X_jobs):
    # Streamlit cache can't hash scipy.sparse; using underscored args avoids hashing
    return cosine_similarity(_X_intern, _X_jobs)

# -----------------------------
# Main
# -----------------------------
if intern_file is None or jobs_file is None:
    st.info("Please upload **Intern_Skills.csv** and **Job_Descriptions.csv** to begin.")
    st.stop()

# Read CSVs
interns = pd.read_csv(intern_file)
jobs = pd.read_csv(jobs_file)
onto_df = pd.read_csv(onto_file) if onto_file is not None else None

# Validate schemas
need_intern = {"intern_id", "name", "skills_text"}
need_job = {"job_id", "title", "description"}

missing_i = need_intern - set(interns.columns)
missing_j = need_job - set(jobs.columns)
if missing_i:
    st.error(f"Intern_Skills.csv is missing columns: {missing_i}")
    st.stop()
if missing_j:
    st.error(f"Job_Descriptions.csv is missing columns: {missing_j}")
    st.stop()

# Normalize & (optional) map synonyms
synmap = build_synonym_map(onto_df)
interns["skills_norm"] = normalize_with_synonyms(interns["skills_text"], synmap)
jobs["desc_norm"] = normalize_with_synonyms(jobs["description"], synmap)

# Vectorize (shared vocab)
corpus = pd.concat([interns["skills_norm"], jobs["desc_norm"]], ignore_index=True)
vec, X_all = fit_tfidf(corpus)
X_intern = X_all[: len(interns)]
X_jobs = X_all[len(interns) :]

# -----------------------------
# Similarity section
# -----------------------------
st.subheader("🔍 Intern ↔ Job Similarity")

top_k = st.slider("Top-K jobs per intern", 1, 10, 3)
sims = compute_similarity(X_intern, X_jobs)

rows = []
for i in range(X_intern.shape[0]):
    top_idx = np.argsort(-sims[i])[:top_k]
    for rank, j in enumerate(top_idx, start=1):
        rows.append(
            {
                "intern_id": interns.iloc[i].get("intern_id", i + 1),
                "intern_name": interns.iloc[i].get("name", f"Intern {i+1}"),
                "job_id": jobs.iloc[j].get("job_id", j + 1),
                "job_title": jobs.iloc[j].get("title", f"Job {j+1}"),
                "similarity": round(float(sims[i, j]), 4),
            }
        )
matches_df = pd.DataFrame(rows)
st.dataframe(matches_df, use_container_width=True)

buf = StringIO()
matches_df.to_csv(buf, index=False)
st.download_button("⬇️ Download Matches CSV", data=buf.getvalue(),
                   file_name="intern_job_top_matches.csv", mime="text/csv")

# -----------------------------
# K-Means on jobs
# -----------------------------
st.subheader("📊 Job Demand Clusters (K-Means)")
k = st.slider("Number of clusters (K)", 2, 12, 5)
km = KMeans(n_clusters=k, n_init=10, random_state=42)
job_labels = km.fit_predict(X_jobs)

jobs_clusters = jobs.copy()
jobs_clusters["cluster"] = job_labels
st.dataframe(jobs_clusters[["job_id", "title", "cluster"]].sort_values("cluster"),
             use_container_width=True)

# Top terms per cluster (approx.)
st.markdown("**Top terms per cluster (approximate):**")
terms = np.array(vec.get_feature_names_out())
centroids = km.cluster_centers_
for c in range(k):
    top_terms = terms[np.argsort(-centroids[c])[:8]]
    st.write(f"Cluster {c}: {', '.join(top_terms)}")

# -----------------------------
# Simple skill gaps (per best match)
# -----------------------------
st.subheader("🧩 Skill Gaps (per best-match job)")

def tokens_set(text: str):
    return set([t for t in norm_text(text).replace(",", " ").split() if t])

vocab_tokens = set([t for t in terms if len(t.split()) == 1])
gap_rows = []
for i in range(len(interns)):
    intern_name = interns.iloc[i].get("name", f"Intern {i+1}")
    intern_tokens = tokens_set(interns.iloc[i]["skills_norm"])
    j = int(np.argmax(sims[i]))  # best job for this intern
    job_title = jobs.iloc[j].get("title", f"Job {j+1}")
    job_tokens = tokens_set(jobs.iloc[j]["desc_norm"]) & vocab_tokens
    missing = sorted(list(job_tokens - intern_tokens))[:15]
    gap_rows.append(
        {
            "intern_name": intern_name,
            "best_job_match": job_title,
            "missing_skill_tokens": ", ".join(missing) if missing else "(none)",
        }
    )

gap_df = pd.DataFrame(gap_rows)
st.dataframe(gap_df, use_container_width=True)

buf2 = StringIO()
gap_df.to_csv(buf2, index=False)
st.download_button("⬇️ Download Skill Gaps CSV", data=buf2.getvalue(),
                   file_name="skill_gaps.csv", mime="text/csv")

# -----------------------------
# Footer
# -----------------------------
st.markdown(
    """
    ---
    **Tips**  
    • If results are noisy, try a different **K** or enrich `Skills_Ontology.csv` synonyms.  
    • Keep skill texts concise for cleaner TF-IDF features.  
    • Use the download buttons to export results.
    """
)
