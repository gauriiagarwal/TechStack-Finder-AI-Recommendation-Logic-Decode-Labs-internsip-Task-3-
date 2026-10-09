# 🎯 TechStack Finder — AI Recommendation Logic

> A content-based career recommender that matches your **skills, interests and career goals** to the best-fitting tech roles, using **TF-IDF** and **Cosine Similarity**, wrapped in a desktop-style **Streamlit** interface with an interactive **3D skill map**.

**Project 3 · Artificial Intelligence · DecodeLabs Industrial Training Kit (Batch 2026)**
**Project by Gauri Agarwal**

---

## ✨ What it does

You tell the app what you know and what excites you. It returns your **Top 3 career matches** with an exact match percentage, the skills you already have for each role, and the skills you still need to learn.

| Step | What you do | What the app does |
|---|---|---|
| 1 | Open the **Add Subjects** folder and pick skills (min. 3) | Reads your skills as the user profile |
| 2 | Open the **Interests** folder | Expands each interest into related skills |
| 3 | Scroll down, open **Career Interests** | Adds the skills of the roles you want to explore |
| 4 | Click **Proceed ▶** | Scores every role and shows 3 result pop-ups (🥇🥈🥉) |
| 5 | Click **Generate 3D graph** | Draws you and all roles in an interactive 3D skill map |

---

## 🧠 How it works

The engine follows the **Input → Process → Output** model and a strict 4-step ranking pipeline.

```
Ingestion  ->  Scoring  ->  Sorting  ->  Filtering (Top-N)
```

1. **Ingestion** – user skills are cleaned, lower-cased and mapped to the dataset vocabulary (for example `ml` becomes `machine learning`, `k8s` becomes `kubernetes`). Unknown skills are ignored and reported back.
2. **Vector mapping + TF-IDF** – the user profile and every job role become numeric vectors over one shared skill vocabulary.
   - `TF = count of term in document / total terms in document`
   - `IDF = log(total documents / documents containing the term)`
   - Rare, specific skills (Terraform, PyTorch) get a high weight. Skills that appear in many roles (Python, Git) get a lower one.
3. **Cosine similarity** – each role is scored against the user vector:

   `cos(θ) = (A · B) / (‖A‖ ‖B‖)`

   It compares direction, not size, so a long skill list does not beat a focused one. Scores range from 0 to 1 and are shown as percentages.
4. **Sorting and filtering** – scores are sorted in descending order and cut to the Top 3 to avoid choice overload.

**Cold start:** if none of the entered skills is recognised, the engine falls back to roles with the broadest skill sets instead of crashing on a zero vector. Because the engine is content-based, newly added roles can be recommended immediately.

### The 3D skill map

Each role is a 58-dimensional vector. The app uses PCA (via NumPy SVD) to squeeze them into 3 axes and plots them with Plotly. Your profile is the pink diamond, your top 3 roles are gold, silver and bronze. The projection is approximate, so **trust the percentages**, which are the exact scores.

---

## 🛠️ Tech stack

| Layer | Technology |
|---|---|
| Language | Python 3 |
| Recommendation engine | Pure Python (`csv`, `math`) — TF-IDF and cosine similarity written from scratch, no ML library |
| Web app | Streamlit (pills, dialogs, session state, tabs) |
| 3D visualisation | Plotly (`Scatter3d`, animated camera rotation) |
| Dimensionality reduction | NumPy (SVD-based PCA) |
| UI styling | Custom HTML + CSS injected into Streamlit (desktop-OS look: menu bar, window, folders, dock) |
| Data | `raw_skills.csv` — 17 job roles, 58 unique skills |

---

## 📁 Project structure

```
internhsip 3/
├── app.py              # Streamlit desktop-style UI + 3D graph
├── recommender.py      # Core engine (TF-IDF, cosine similarity, pipeline) + CLI mode
├── raw_skills.csv      # Dataset: role, skills (separated by ;)
├── requirements.txt    # streamlit, plotly
└── README.md
```

---

## 🚀 Getting started

**Requirements:** Python 3.10+ and a recent version of Streamlit.

```bash
# 1. go to the project folder
cd "path/to/internhsip 3"

# 2. install dependencies
pip install -r requirements.txt

# 3. run the app
streamlit run app.py
```

On Windows, if `python` or `streamlit` is not recognised, use the launcher:

```powershell
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

The app opens at `http://localhost:8501`.

### Command-line mode (no UI)

```bash
python recommender.py Python "Cloud Computing" Automation
```

```
Your profile : python, cloud, automation

Top 3 career matches:
  1. Cloud Engineer               match:  52.2%
     matched skills: cloud, automation
  2. Cloud Architect              match:  47.9%
     matched skills: cloud, automation
  3. DevOps Engineer              match:  12.9%
     matched skills: automation, python
```

Run `python recommender.py` with no arguments to enter skills interactively.

---

## 🗂️ Dataset format

`raw_skills.csv` has two columns. Add your own roles by appending rows.

```csv
role,skills
Data Scientist,python;sql;machine learning;statistics;data analysis;pandas;scikit-learn;visualization
DevOps Engineer,aws;docker;kubernetes;ci/cd;linux;terraform;git;automation;python
```

Keep skill names consistent across rows. A mismatch such as "Web Design" and "Frontend Development" breaks the similarity math. Add shortcuts to the `ALIASES` dictionary in `recommender.py` to map alternate spellings.

---

## ⚠️ Limitations

- The dataset is small (17 roles) and hand-written, so results reflect that data, not the whole job market.
- Skills are matched by exact name. There is no semantic understanding of synonyms beyond the alias table.
- The 3D map is a 3-axis approximation of a 58-dimensional space.
- Choosing a role under **Career Interests** adds its skills to your profile, so that role is more likely to rank high.

## 🔮 Future scope

- Let users **rate** how strongly they know each skill, using weights instead of 0/1.
- Load a larger real-world job dataset and update it automatically.
- Add **collaborative filtering** and compare it with the content-based approach.
- Add learning-path suggestions (courses and projects) for the missing skills.
- Deploy on Streamlit Community Cloud.

---

## 🙏 Acknowledgements

Built as **Project 3: AI Recommendation Logic** in the **DecodeLabs** Industrial Training Kit (Batch 2026).

**Author:** Gauri Agarwal
