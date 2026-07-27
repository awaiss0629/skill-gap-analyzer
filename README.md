# Skill Gap Analyzer

A Streamlit app that matches interns to job roles and identifies skill gaps using NLP and clustering techniques.

## Features
- Converts raw text into vectors using TF-IDF (uni/bi-grams)
- Matches interns to roles using cosine similarity
- Discovers job-demand themes via K-Means clustering
- Highlights missing skills per intern vs. best-fit role
- Exports match and gap results as downloadable CSVs

## Tech Stack
Python, Streamlit, scikit-learn, pandas, NumPy

## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Input Data
Upload three CSVs (samples provided in `sample_data/`):
- `Intern_Skills.csv` — intern skill profiles
- `Job_Descriptions.csv` — job role descriptions
- `Skills_Ontology.csv` — optional skills reference/synonym mapping

## Output
- `intern_job_top_matches.csv` — best-fit role matches per intern
- `skill_gaps.csv` — identified skill gaps per intern vs. matched role

## License
See [LICENSE](LICENSE) for details.