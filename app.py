"""
Tech Stack Recommender (Project 3) - "desktop OS" Streamlit UI.
Run:  py -m streamlit run app.py
Needs recommender.py and raw_skills.csv in the same folder, plus: pip install streamlit plotly
"""
import os

import numpy as np
import streamlit as st

from recommender import build_vocab, compute_idf, load_items, recommend, tfidf_vector

try:
    import plotly.graph_objects as go
except ImportError:  # app still works, only the 3D graph needs plotly
    go = None

st.set_page_config(
    page_title="TechStack Finder",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "raw_skills.csv")

# Interests are broad ideas; each one expands into skills from the dataset
INTERESTS = {
    "🌐 Building websites": ["html", "css", "javascript", "react", "web design"],
    "🤖 AI & Data": ["machine learning", "python", "statistics", "data analysis"],
    "☁️ Cloud & Infrastructure": ["aws", "cloud", "docker", "kubernetes", "linux"],
    "🔐 Cybersecurity": ["security", "networking", "cryptography", "penetration testing"],
    "📱 Mobile apps": ["kotlin", "swift", "flutter", "ui design"],
    "⚙️ Automation & Testing": ["automation", "testing", "selenium", "ci/cd"],
    "🗄️ Databases": ["sql", "database design", "performance tuning"],
    "🧩 Problem solving": ["algorithms", "data structures", "mathematics"],
}

# Skill groups only organise the "Add Subjects" popup into tabs
SKILL_GROUPS = {
    "💻 Programming": ["python", "java", "javascript", "typescript", "kotlin", "swift", "bash",
                       "algorithms", "data structures", "git", "apis", "microservices",
                       "testing", "selenium"],
    "📊 Data & AI": ["sql", "pandas", "statistics", "mathematics", "data analysis", "visualization",
                     "excel", "tableau", "machine learning", "deep learning", "nlp", "tensorflow",
                     "pytorch", "scikit-learn", "mlops", "spark", "etl", "kafka", "airflow",
                     "data warehousing"],
    "☁️ Cloud & DevOps": ["aws", "azure", "cloud", "docker", "kubernetes", "terraform", "ci/cd",
                          "linux", "automation", "virtualization", "networking", "troubleshooting"],
    "🌐 Web & Mobile": ["html", "css", "react", "node.js", "web design", "ui design", "flutter",
                        "firebase"],
    "🔐 Security & DB": ["security", "cryptography", "penetration testing", "database design",
                         "performance tuning", "backup"],
}

ROLE_BLURBS = {
    "Data Scientist": "Finds patterns in data and builds predictive models.",
    "Machine Learning Engineer": "Turns ML models into reliable, deployed products.",
    "Data Analyst": "Turns raw data into dashboards and decisions.",
    "Data Engineer": "Builds the pipelines that move and clean data.",
    "DevOps Engineer": "Automates how software is built, shipped and run.",
    "Cloud Architect": "Designs secure, scalable systems on the cloud.",
    "Cloud Engineer": "Builds and runs cloud infrastructure.",
    "System Administrator": "Keeps servers and networks healthy.",
    "Backend Developer": "Builds server-side logic, APIs and databases.",
    "Frontend Developer": "Builds what users see and click in the browser.",
    "Full Stack Developer": "Works on both the frontend and the backend.",
    "Mobile App Developer": "Builds Android and iOS apps.",
    "Cybersecurity Analyst": "Protects systems and hunts for vulnerabilities.",
    "QA Automation Engineer": "Writes automated tests to catch bugs early.",
    "AI Research Engineer": "Experiments with new deep-learning and NLP ideas.",
    "Software Engineer": "Designs and builds software on strong CS basics.",
    "Database Administrator": "Keeps databases fast, safe and backed up.",
}

MEDALS = ["🥇", "🥈", "🥉"]
TOP_COLORS = ["#ffd700", "#c9d1d9", "#e08a4b"]  # gold, silver, bronze


@st.cache_data
def load_engine():
    items = load_items(CSV_PATH)
    vocab = build_vocab(items)
    idf = compute_idf(items, vocab)
    return items, vocab, idf


items, vocab, idf = load_engine()
ROLES = list(items)

# skills not covered by a group go to an "Other" tab (keeps the app working if the CSV changes)
GROUPS = {k: [s for s in v if s in vocab] for k, v in SKILL_GROUPS.items()}
_covered = {s for v in GROUPS.values() for s in v}
_other = [s for s in vocab if s not in _covered]
if _other:
    GROUPS["✨ Other"] = _other

# ------------------------------------------------------------------ state
DEFAULTS = dict(subjects=[], extra="", interests=[], careers=[], results=None,
                known=[], unknown=[], popup_idx=0, show_graph=False)
for _k, _v in DEFAULTS.items():
    st.session_state.setdefault(_k, _v)


def parse_typed(text):
    return [x.strip() for x in (text or "").split(",") if x.strip()]


# ------------------------------------------------------------------ styling
CSS = """<style>
#MainMenu, footer, [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] {display:none !important;}
.stApp {background: radial-gradient(900px 500px at 12% 8%, rgba(255,120,200,.55), transparent 60%), radial-gradient(900px 600px at 92% 18%, rgba(80,200,255,.50), transparent 60%), radial-gradient(1100px 700px at 50% 105%, rgba(140,90,255,.70), transparent 60%), linear-gradient(135deg,#1b1744,#2b1f6b 45%,#0f4c75); background-attachment: fixed;}
.menubar {position:fixed;top:0;left:0;right:0;height:30px;background:rgba(15,12,40,.55);backdrop-filter:blur(14px);color:#fff;font:600 13px/30px 'Segoe UI',system-ui,sans-serif;display:flex;gap:22px;padding:0 18px;z-index:999;}
.menubar .right {margin-left:auto;opacity:.75;font-weight:500;}
.dock {position:fixed;bottom:14px;left:50%;transform:translateX(-50%);display:flex;gap:16px;padding:10px 20px;background:rgba(255,255,255,.18);border:1px solid rgba(255,255,255,.35);border-radius:22px;backdrop-filter:blur(16px);z-index:999;font-size:26px;}
.dock span {transition:transform .2s ease;cursor:default;}
.dock span:hover {transform:translateY(-9px) scale(1.25);}
[data-testid="stMainBlockContainer"], .block-container {max-width:1000px !important;margin:52px auto 110px auto !important;padding:0 2.2rem 2.2rem 2.2rem !important;background:rgba(247,248,255,.95);border-radius:16px;border:1px solid rgba(255,255,255,.6);box-shadow:0 30px 80px rgba(8,6,30,.55), 0 0 0 1px rgba(0,0,0,.18);}
.titlebar {margin:0 -2.2rem 1.2rem -2.2rem;height:44px;border-radius:16px 16px 0 0;background:linear-gradient(180deg,#f3f4fb,#dfe2f1);border-bottom:1px solid #c9cde3;display:flex;align-items:center;padding:0 16px;position:relative;}
.dot {width:13px;height:13px;border-radius:50%;margin-right:8px;display:inline-block;}
.titlebar .title {position:absolute;left:0;right:0;text-align:center;font:600 14px 'Segoe UI',system-ui,sans-serif;color:#3b3f5c;pointer-events:none;}
.hero {text-align:center;margin:6px 0 14px 0;}
.hero h1 {font:800 2.1rem 'Segoe UI',system-ui,sans-serif !important;margin:0 !important;padding:0 !important;background:linear-gradient(90deg,#6c5ce7,#ff4d9d);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.hero p {margin:6px 0 0 0;color:#5b5f86;font:500 1rem 'Segoe UI',system-ui,sans-serif;}
.slabel {font:800 .78rem 'Segoe UI',system-ui,sans-serif;letter-spacing:.14em;color:#6c5ce7;margin:14px 0 4px 2px;}
.st-key-area1, .st-key-area2 {background:rgba(120,110,255,.07);border:2px dashed rgba(90,80,200,.25);border-radius:14px;padding:34px 10px 18px 10px;min-height:290px;}
[class*="st-key-folder_"], .st-key-proceed, .st-key-gen3d, .st-key-reset {display:flex !important;flex-direction:row !important;justify-content:center !important;width:100% !important;}
[class*="st-key-folder_"] [data-testid="stButton"], .st-key-proceed [data-testid="stButton"], .st-key-gen3d [data-testid="stButton"], .st-key-reset [data-testid="stButton"] {display:flex !important;justify-content:center !important;width:100% !important;}
[class*="st-key-folder_"] {--f1:#ffe08a;--f2:#f7b733;--tab:#e3a11c;--emoji:"📁";}
.st-key-folder_subjects {--f1:#ffe08a;--f2:#f7b733;--tab:#e3a11c;--emoji:"🎓";}
.st-key-folder_interests {--f1:#ffb4c8;--f2:#ff6a88;--tab:#e8506f;--emoji:"💡";}
.st-key-folder_careers {--f1:#9fe7ff;--f2:#4facfe;--tab:#3b8fe0;--emoji:"🧭";}
[class*="st-key-folder_"] button {position:relative;width:210px;height:140px;margin-top:18px;border:0 !important;border-radius:4px 14px 14px 14px !important;background:linear-gradient(180deg,var(--f1),var(--f2)) !important;box-shadow:inset 0 2px 0 rgba(255,255,255,.65), 0 16px 30px rgba(40,30,120,.35);display:flex;align-items:flex-end;justify-content:center;padding:0 0 14px 0 !important;transition:transform .18s ease, box-shadow .18s ease;}
[class*="st-key-folder_"] button::before {content:"";position:absolute;top:-16px;left:0;width:84px;height:20px;background:var(--tab);border-radius:10px 10px 0 0;}
[class*="st-key-folder_"] button::after {content:var(--emoji);position:absolute;top:22px;left:50%;transform:translateX(-50%);font-size:44px;line-height:1;}
[class*="st-key-folder_"] button:hover {transform:translateY(-6px) scale(1.04);box-shadow:inset 0 2px 0 rgba(255,255,255,.65), 0 24px 40px rgba(40,30,120,.45);}
[class*="st-key-folder_"] button p {font-weight:800;font-size:1.05rem;color:#3a2a00 !important;}
.fbadge {display:block;text-align:center;margin-top:12px;font:600 13px 'Segoe UI',system-ui,sans-serif;color:#6b6f96;}
.fbadge.ok {color:#188a3e;}
.st-key-proceed button, .st-key-gen3d button {width:320px !important;background:linear-gradient(135deg,#6c5ce7,#a363ff) !important;border:0 !important;border-radius:14px !important;padding:.85rem 2.4rem !important;box-shadow:0 12px 26px rgba(108,92,231,.45);transition:transform .15s ease;}
.st-key-proceed button:hover, .st-key-gen3d button:hover {transform:translateY(-3px);}
.st-key-proceed button p, .st-key-gen3d button p {color:#fff !important;font-weight:800;font-size:1.05rem;}
.st-key-reset button {width:200px !important;background:rgba(0,0,0,.05) !important;border:1px solid rgba(0,0,0,.12) !important;border-radius:12px !important;}
.st-key-reset button p {color:#3b3f5c !important;}
.hint {text-align:center;color:#6b6f96;font:500 .85rem 'Segoe UI',system-ui,sans-serif;margin:8px 0 14px 0;}
.rstrip {display:flex;gap:12px;flex-wrap:wrap;justify-content:center;margin:12px 0;}
.rchip {background:#fff;border:1px solid #e1e3f5;border-radius:14px;padding:10px 16px;box-shadow:0 6px 16px rgba(60,50,160,.12);font:700 .95rem 'Segoe UI',system-ui,sans-serif;color:#1d1f3a;}
.rchip b {color:#6c5ce7;}
.st-key-graphwin {background:#0f1226;border-radius:14px;overflow:hidden;border:1px solid #2b2f5a;box-shadow:0 16px 40px rgba(15,18,38,.5);padding:0 !important;gap:0 !important;margin-top:10px;}
.mini-title {margin-bottom:1rem;background:linear-gradient(180deg,#2a2e57,#1b1e3d);color:#cfd3ff;font:600 13px 'Segoe UI',system-ui,sans-serif;padding:9px 14px;display:flex;align-items:center;gap:8px;}
.mini-title .dot {width:11px;height:11px;margin:0;}
.statusbar {margin:26px -2.2rem -2.2rem -2.2rem;padding:8px 18px;background:#e7e9f7;border-top:1px solid #d3d6ee;border-radius:0 0 16px 16px;font:500 12px 'Segoe UI',system-ui,sans-serif;color:#5b5f86;display:flex;justify-content:space-between;}
section[role="dialog"] {border-radius:16px !important;box-shadow:0 30px 80px rgba(8,6,30,.55) !important;}
section[role="dialog"] button[data-testid="stBaseButton-primary"] {background:linear-gradient(135deg,#6c5ce7,#a363ff) !important;border:0 !important;color:#fff !important;border-radius:10px !important;font-weight:700;}
section[role="dialog"] button[data-variant="pills"][aria-pressed="true"] {background:#6c5ce7 !important;border-color:#6c5ce7 !important;color:#fff !important;}
section[role="dialog"] [data-testid="stTab"][aria-selected="true"] {color:#a99cff !important;border-bottom:2px solid #6c5ce7 !important;}
.pbar {height:10px;border-radius:99px;background:rgba(128,128,160,.25);overflow:hidden;margin:6px 0 14px 0;}
.pbar div {height:100%;border-radius:99px;background:linear-gradient(90deg,#6c5ce7,#ff4d9d);}
.doc-head {background:rgba(108,92,231,.14);border-left:4px solid #6c5ce7;padding:8px 12px;border-radius:8px;font:600 .9rem 'Segoe UI',system-ui,sans-serif;margin-bottom:8px;}
.res-card {text-align:center;padding:6px 0 0 0;}
.res-rank {font:700 .85rem 'Segoe UI',system-ui,sans-serif;opacity:.7;letter-spacing:.08em;}
.res-role {font:800 1.7rem 'Segoe UI',system-ui,sans-serif;margin:4px 0;}
.res-pct {font:900 3rem 'Segoe UI',system-ui,sans-serif;background:linear-gradient(90deg,#6c5ce7,#ff4d9d);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
</style>"""

MENUBAR = ('<div class="menubar"><span>🎯 TechStack Finder</span><span>File</span><span>Edit</span>'
           '<span>View</span><span>Help</span><span class="right">Batch 2026 · DecodeLabs</span></div>'
           '<div class="dock"><span>📁</span><span>🧭</span><span>🌐</span><span>🎯</span>'
           '<span>📊</span><span>⚙️</span></div>')

TITLEBAR = ('<div class="titlebar"><span class="dot" style="background:#ff5f57"></span>'
            '<span class="dot" style="background:#febc2e"></span><span class="dot" style="background:#28c840"></span>'
            '<div class="title">📁 TechStack Finder — Project 3 · AI Recommendation Logic</div></div>')


def folder_glow_css():
    glow = []
    if st.session_state.subjects or parse_typed(st.session_state.extra):
        glow.append(".st-key-folder_subjects button")
    if st.session_state.interests:
        glow.append(".st-key-folder_interests button")
    if st.session_state.careers:
        glow.append(".st-key-folder_careers button")
    if not glow:
        return ""
    return "<style>" + ",".join(glow) + "{outline:4px solid #34c759;outline-offset:4px;}</style>"


def plural(n, word):
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def badge(n, word):
    if n:
        return f'<span class="fbadge ok">✔ {plural(n, word)} selected</span>'
    return '<span class="fbadge">Empty · click to open</span>'


# ------------------------------------------------------------------ popups (the "little docs")
@st.dialog("🎓 Add Subjects", width="large")
def subjects_dialog():
    st.markdown('<div class="doc-head">📄 subjects.doc — pick what you know (at least 3 in total)</div>',
                unsafe_allow_html=True)
    tabs = st.tabs(list(GROUPS))
    chosen = []
    for tab, (group, options) in zip(tabs, GROUPS.items()):
        with tab:
            sel = st.pills(
                group, options, selection_mode="multi",
                default=[s for s in st.session_state.subjects if s in options],
                key=f"pills_{group}", label_visibility="collapsed",
            )
            chosen += list(sel or [])
    extra = st.text_input("➕ Something missing? Type extra skills (comma separated)",
                          value=st.session_state.extra, key="extra_input",
                          placeholder="e.g. ml, k8s, cloud computing")
    total = len(chosen) + len(parse_typed(extra))
    st.caption(f"{total} selected")
    if st.button("Save & close ✔", type="primary", key="save_subjects"):
        st.session_state.subjects = sorted(set(chosen))
        st.session_state.extra = extra
        st.rerun()


@st.dialog("💡 Interests", width="large")
def interests_dialog():
    st.markdown('<div class="doc-head">📄 interests.doc — what excites you?</div>', unsafe_allow_html=True)
    sel = st.pills("Interests", list(INTERESTS), selection_mode="multi",
                   default=st.session_state.interests, key="pills_interests",
                   label_visibility="collapsed")
    st.caption(f"{len(sel or [])} selected")
    if st.button("Save & close ✔", type="primary", key="save_interests"):
        st.session_state.interests = list(sel or [])
        st.rerun()


@st.dialog("🧭 Career Interests", width="large")
def careers_dialog():
    st.markdown('<div class="doc-head">📄 careers.doc — which roles would you like to explore?</div>',
                unsafe_allow_html=True)
    sel = st.pills("Careers", ROLES, selection_mode="multi",
                   default=st.session_state.careers, key="pills_careers",
                   label_visibility="collapsed")
    for role in (sel or []):
        st.caption(f"**{role}** — {ROLE_BLURBS.get(role, '')}")
    if st.button("Save & close ✔", type="primary", key="save_careers"):
        st.session_state.careers = list(sel or [])
        st.rerun()


def _nav(step):
    st.session_state.popup_idx += step


@st.dialog("🏆 Your match")
def results_dialog():
    res = st.session_state.results
    i = st.session_state.popup_idx
    r = res[i]
    st.markdown(
        f'<div class="res-card"><div class="res-rank">{MEDALS[i]} MATCH {i + 1} OF {len(res)}</div>'
        f'<div class="res-role">{r["role"]}</div><div class="res-pct">{r["pct"]}%</div></div>',
        unsafe_allow_html=True)
    st.markdown(f'<div class="pbar"><div style="width:{min(max(r["pct"], 0), 100)}%"></div></div>', unsafe_allow_html=True)
    st.write(ROLE_BLURBS.get(r["role"], ""))
    if r["matched"]:
        st.markdown("**✅ You already have**  \n" + " ".join(f":green-badge[{s}]" for s in r["matched"]))
    if r["missing"]:
        st.markdown("**📚 Skills to learn**  \n" + " ".join(f":orange-badge[{s}]" for s in r["missing"]))
    if i == len(res) - 1 and st.session_state.unknown:
        st.caption("Not in our dataset, so ignored: " + ", ".join(sorted(set(st.session_state.unknown))))
    left, right = st.columns(2)
    with left:
        if i > 0:
            st.button("‹ Back", key="res_back", on_click=_nav, args=(-1,))
    with right:
        if i < len(res) - 1:
            st.button("Next ›", key="res_next", type="primary", on_click=_nav, args=(1,))
        elif st.button("Finish ✓", key="res_done", type="primary"):
            st.rerun()


# ------------------------------------------------------------------ 3D graph
def build_3d_figure(known, results):
    names = ROLES
    X = np.array([tfidf_vector(items[r], vocab, idf) for r in names], dtype=float)
    u = np.array(tfidf_vector(known, vocab, idf), dtype=float)

    def unit(v):
        n = np.linalg.norm(v)
        return v / n if n else v

    X = np.array([unit(row) for row in X])
    u = unit(u)
    cos = X @ u

    # squeeze the skill space (58 dimensions) down to 3 axes with PCA. We centre the roles first,
    # otherwise every role sits in one tight clump. The % scores stay exact (computed above).
    centre = X.mean(axis=0)
    _, _, vt = np.linalg.svd(X - centre, full_matrices=False)
    basis = vt[:3].T
    P = (X - centre) @ basis
    up = (u - centre) @ basis
    lim = float(max(np.abs(P).max(), np.abs(up).max())) * 1.2

    top_names = [r["role"] for r in results]
    others = [i for i, n in enumerate(names) if n not in top_names]
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=P[others, 0], y=P[others, 1], z=P[others, 2], mode="markers",
        marker=dict(size=6, color="#6f76b8", opacity=0.85, line=dict(width=1, color="#aab0f0")),
        text=[f"{names[i]} · {cos[i] * 100:.1f}%" for i in others],
        hovertemplate="%{text}<extra></extra>", name="Other roles"))

    for rank, r in enumerate(results):
        i = names.index(r["role"])
        color = TOP_COLORS[rank]
        fig.add_trace(go.Scatter3d(
            x=[0, P[i, 0]], y=[0, P[i, 1]], z=[0, P[i, 2]], mode="lines",
            line=dict(color=color, width=5), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter3d(
            x=[P[i, 0]], y=[P[i, 1]], z=[P[i, 2]], mode="markers+text",
            marker=dict(size=13, color=color, line=dict(width=2, color="#ffffff")),
            text=[f"#{rank + 1} {r['role']}"],
            textposition=["top center", "top right", "top left"][rank],
            textfont=dict(color="#ffffff", size=13),
            hovertemplate=f"{r['role']} · {r['pct']}%<extra></extra>", name=f"#{rank + 1} {r['role']}"))

    fig.add_trace(go.Scatter3d(
        x=[0, up[0]], y=[0, up[1]], z=[0, up[2]], mode="lines",
        line=dict(color="#ff4d9d", width=7), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter3d(
        x=[up[0]], y=[up[1]], z=[up[2]], mode="markers+text",
        marker=dict(size=15, color="#ff4d9d", symbol="diamond", line=dict(width=2, color="#ffffff")),
        text=["You"], textposition="bottom center", textfont=dict(color="#ffb3d3", size=14),
        hovertemplate="Your profile<extra></extra>", name="You"))
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0], mode="markers", marker=dict(size=4, color="#ffffff"),
        hoverinfo="skip", showlegend=False))

    axis = dict(showbackground=False, gridcolor="rgba(255,255,255,.12)", zeroline=False,
                color="#8f95d0", showticklabels=False)
    frames = [go.Frame(layout=dict(scene_camera=dict(
        eye=dict(x=1.6 * np.cos(a), y=1.6 * np.sin(a), z=0.8)))) for a in np.linspace(0, 2 * np.pi, 90)]
    fig.frames = frames
    fig.update_layout(
        height=560, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="#0f1226", plot_bgcolor="#0f1226",
        scene=dict(xaxis=dict(title="", range=[-lim, lim], **axis),
                   yaxis=dict(title="", range=[-lim, lim], **axis),
                   zaxis=dict(title="", range=[-lim, lim], **axis),
                   aspectmode="cube", bgcolor="#0f1226",
                   camera=dict(eye=dict(x=1.5, y=1.1, z=0.8))),
        legend=dict(font=dict(color="#cfd3ff"), bgcolor="rgba(0,0,0,0)", x=0.01, y=0.99),
        updatemenus=[dict(
            type="buttons", showactive=False, x=0.01, y=0.02, xanchor="left", yanchor="bottom",
            bgcolor="#6c5ce7", font=dict(color="#ffffff"),
            buttons=[
                dict(label="▶ Rotate", method="animate",
                     args=[None, dict(frame=dict(duration=60, redraw=True), transition=dict(duration=0),
                                      fromcurrent=True, mode="immediate")]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False), transition=dict(duration=0),
                                        mode="immediate")]),
            ])],
    )
    return fig


# ------------------------------------------------------------------ page
# one single markdown block, so no extra grey gap appears above the window title bar
st.markdown(CSS + folder_glow_css() + "\n" + MENUBAR + "\n" + TITLEBAR, unsafe_allow_html=True)
st.markdown(
    '<div class="hero"><h1>Find your perfect tech career 🚀</h1>'
    '<p>Open the folders, pick what fits you, then hit <b>Proceed</b>.</p></div>',
    unsafe_allow_html=True)

n_sub = len(st.session_state.subjects) + len(parse_typed(st.session_state.extra))
n_int = len(st.session_state.interests)
n_car = len(st.session_state.careers)

st.markdown('<div class="slabel">STEP 1 · ABOUT YOU — click a folder to open it</div>', unsafe_allow_html=True)
with st.container(key="area1"):
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Add Subjects", key="folder_subjects"):
            subjects_dialog()
        st.markdown(badge(n_sub, "skill"), unsafe_allow_html=True)
    with col2:
        if st.button("Interests", key="folder_interests"):
            interests_dialog()
        st.markdown(badge(n_int, "interest"), unsafe_allow_html=True)

st.markdown('<div class="slabel">STEP 2 · SCROLL DOWN — add your career interests</div>', unsafe_allow_html=True)
with st.container(key="area2"):
    if st.button("Career Interests", key="folder_careers"):
        careers_dialog()
    st.markdown(badge(n_car, "career"), unsafe_allow_html=True)

st.markdown('<div class="slabel">STEP 3 · GET YOUR MATCHES</div>', unsafe_allow_html=True)
if st.button("Proceed ▶", key="proceed"):
    if n_sub < 3:
        st.toast("Please add at least 3 subjects first 📁", icon="⚠️")
    else:
        profile = list(st.session_state.subjects) + parse_typed(st.session_state.extra)
        for name in st.session_state.interests:
            profile += INTERESTS[name]
        for role in st.session_state.careers:
            profile += items[role]
        top, known, unknown = recommend(profile, items, vocab, idf)
        if not known:
            st.toast("None of those skills are in our dataset. Try picking from the folders.", icon="🤔")
        else:
            st.session_state.results = [
                dict(role=role, pct=round(score * 100, 1),
                     matched=[s for s in items[role] if s in known],
                     missing=[s for s in items[role] if s not in known])
                for role, score in top
            ]
            st.session_state.known = known
            st.session_state.unknown = unknown
            st.session_state.popup_idx = 0
            st.session_state.show_graph = False
            results_dialog()
st.markdown('<div class="hint">Needs at least 3 subjects. Interests and careers make it sharper.</div>',
            unsafe_allow_html=True)

if st.session_state.results:
    chips = "".join(
        f'<div class="rchip">{MEDALS[i]} {r["role"]} <b>{r["pct"]}%</b></div>'
        for i, r in enumerate(st.session_state.results))
    st.markdown(f'<div class="slabel">YOUR TOP 3</div><div class="rstrip">{chips}</div>', unsafe_allow_html=True)

    if st.button("📊 Generate 3D graph", key="gen3d"):
        st.session_state.show_graph = True

    if st.session_state.show_graph:
        if go is None:
            st.info("The 3D graph needs plotly. Run:  py -m pip install plotly  and restart the app.")
        else:
            with st.container(key="graphwin"):
                st.markdown(
                    '<div class="mini-title"><span class="dot" style="background:#ff5f57"></span>'
                    '<span class="dot" style="background:#febc2e"></span>'
                    '<span class="dot" style="background:#28c840"></span>'
                    '📊 skill_space_3d.plot — drag to rotate, scroll to zoom</div>',
                    unsafe_allow_html=True)
                st.plotly_chart(build_3d_figure(st.session_state.known, st.session_state.results),
                                config={"displaylogo": False})
            st.markdown('<div class="hint">The pink diamond is you, gold/silver/bronze are your top 3 roles. '
                        'The map squeezes 58 skills into 3 axes, so distances are approximate; '
                        'the percentages are the exact scores.</div>',
                        unsafe_allow_html=True)

    if st.button("🔄 Start over", key="reset"):
        for k, v in DEFAULTS.items():
            st.session_state[k] = v
        st.rerun()

st.markdown(
    f'<div class="statusbar"><span>{plural(n_sub, "skill")} · {plural(n_int, "interest")} · {plural(n_car, "career")}</span>'
    '<span>TF-IDF + Cosine Similarity · pure Python engine</span></div>',
    unsafe_allow_html=True)
