from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dashboard_data"

st.set_page_config(
    page_title="Network Intrusion Detection | Naman Saini",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------- Data ----------
@st.cache_data
def load_data():
    return (
        pd.read_csv(DATA / "model_performance.csv"),
        pd.read_csv(DATA / "cicids_class_distribution.csv"),
        pd.read_csv(DATA / "unsw_test_distribution.csv"),
        pd.read_csv(DATA / "unsw_rf_attack_specific.csv"),
    )

performance, cicids_dist, unsw_dist, unsw_rf = load_data()
cicids = performance[performance["dataset"] == "CICIDS2017"].copy()
unsw = performance[performance["dataset"] == "UNSW-NB15"].copy()

# ---------- Visual system ----------
st.markdown(
    """
    <style>
        .stApp {
            background: #f6f8fb;
        }

        .block-container {
            max-width: 1280px;
            padding-top: 2.2rem;
            padding-bottom: 3rem;
        }

        header[data-testid="stHeader"] {
            background: transparent;
        }

        #MainMenu, footer {
            visibility: hidden;
        }

        h1, h2, h3 {
            letter-spacing: -0.02em;
        }

        .hero {
            background: linear-gradient(135deg, #0f172a 0%, #172554 58%, #1e3a5f 100%);
            border-radius: 22px;
            padding: 34px 38px 30px 38px;
            margin-bottom: 22px;
            box-shadow: 0 16px 40px rgba(15, 23, 42, 0.12);
        }

        .hero-kicker {
            color: #93c5fd;
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            margin-bottom: 10px;
        }

        .hero-title {
            color: white;
            font-size: 2.05rem;
            line-height: 1.14;
            font-weight: 750;
            margin: 0 0 10px 0;
        }

        .hero-sub {
            color: #cbd5e1;
            font-size: 0.98rem;
            max-width: 900px;
            line-height: 1.55;
            margin-bottom: 18px;
        }

        .pill {
            display: inline-block;
            color: #dbeafe;
            background: rgba(255,255,255,0.09);
            border: 1px solid rgba(255,255,255,0.13);
            border-radius: 999px;
            padding: 6px 11px;
            margin-right: 7px;
            margin-bottom: 5px;
            font-size: 0.78rem;
        }

        .metric-card {
            background: white;
            border: 1px solid #e6eaf0;
            border-radius: 16px;
            padding: 20px 20px 17px 20px;
            min-height: 120px;
            box-shadow: 0 7px 20px rgba(15, 23, 42, 0.04);
        }

        .metric-label {
            color: #64748b;
            font-size: 0.76rem;
            font-weight: 650;
            text-transform: uppercase;
            letter-spacing: 0.055em;
            margin-bottom: 10px;
        }

        .metric-value {
            color: #0f172a;
            font-size: 1.95rem;
            font-weight: 760;
            line-height: 1;
        }

        .metric-note {
            color: #64748b;
            font-size: 0.82rem;
            margin-top: 10px;
        }

        .section-label {
            color: #64748b;
            text-transform: uppercase;
            font-size: 0.72rem;
            letter-spacing: 0.10em;
            font-weight: 750;
            margin-bottom: 4px;
        }

        .section-title {
            color: #0f172a;
            font-size: 1.35rem;
            font-weight: 750;
            margin-bottom: 4px;
        }

        .section-copy {
            color: #64748b;
            font-size: 0.91rem;
            line-height: 1.5;
            margin-bottom: 12px;
        }

        .takeaway {
            background: white;
            border: 1px solid #e6eaf0;
            border-left: 4px solid #2563eb;
            border-radius: 14px;
            padding: 20px 20px;
            margin-bottom: 12px;
        }

        .takeaway strong {
            color: #0f172a;
        }

        .takeaway p {
            color: #475569;
            margin: 7px 0 0 0;
            line-height: 1.52;
            font-size: 0.91rem;
        }

        .caution {
            background: #fffaf0;
            border: 1px solid #fde7b1;
            border-radius: 14px;
            padding: 16px 18px;
            color: #7c4a03;
            line-height: 1.5;
            font-size: 0.88rem;
        }

        div[data-testid="stTabs"] button {
            font-weight: 650;
            padding-left: 18px;
            padding-right: 18px;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid #e6eaf0;
            border-radius: 14px;
            overflow: hidden;
        }

        .mini-card {
            background: white;
            border: 1px solid #e6eaf0;
            border-radius: 14px;
            padding: 16px 18px;
            min-height: 118px;
        }

        .mini-card h4 {
            color: #0f172a;
            margin: 0 0 8px 0;
            font-size: 0.95rem;
        }

        .mini-card p {
            color: #64748b;
            font-size: 0.84rem;
            line-height: 1.45;
            margin: 0;
        }

        .footer-note {
            color: #94a3b8;
            font-size: 0.76rem;
            text-align: center;
            padding-top: 28px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

BLUE = "#2563eb"
NAVY = "#0f172a"
AMBER = "#d97706"
MUTED = "#94a3b8"

# ---------- Helpers ----------
def metric_card(label, value, note):
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-note">{note}</div>
    </div>
    """

def section_head(kicker, title, copy):
    st.markdown(
        f"""
        <div class="section-label">{kicker}</div>
        <div class="section-title">{title}</div>
        <div class="section-copy">{copy}</div>
        """,
        unsafe_allow_html=True,
    )

def clean_layout(fig, height=430):
    fig.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="white",
        font=dict(family="Arial, sans-serif", color="#334155", size=12),
        margin=dict(l=10, r=20, t=20, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            title=None,
        ),
    )
    fig.update_xaxes(gridcolor="#e9eef5", zeroline=False)
    fig.update_yaxes(gridcolor="#f3f5f8", zeroline=False)
    return fig

# ---------- Hero ----------
st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">MSc Artificial Intelligence · University of Bath · 2026</div>
        <div class="hero-title">Network Intrusion Detection — ML vs Deep Learning</div>
        <div class="hero-sub">
            Comparative evaluation of seven machine-learning and deep-learning models on CICIDS2017 and UNSW-NB15,
            with emphasis on class-balanced performance, minority-attack behaviour and dataset dependence.
        </div>
        <span class="pill">7 models</span>
        <span class="pill">2 benchmark datasets</span>
        <span class="pill">ANOVA top-40 features</span>
        <span class="pill">Macro F1 primary metric</span>
        <span class="pill">Dissertation mark: 72%</span>
    </div>
    """,
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)
m1.markdown(metric_card("Best model", "Random Forest", "Ranked #1 by Macro F1 on both datasets"), unsafe_allow_html=True)
m2.markdown(metric_card("CICIDS2017 Macro F1", "0.892", "Random Forest · primary benchmark"), unsafe_allow_html=True)
m3.markdown(metric_card("UNSW-NB15 Macro F1", "0.448", "Random Forest · official test set"), unsafe_allow_html=True)
m4.markdown(metric_card("UNSW highest accuracy", "0.757", "XGBoost · weighted F1 = 0.756"), unsafe_allow_html=True)

st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

overview_tab, classes_tab, method_tab = st.tabs(
    ["Overview", "Class performance", "Methodology & limitations"]
)

# ---------- Overview ----------
with overview_tab:
    section_head(
        "Headline comparison",
        "Random Forest leads on the primary metric",
        "Macro F1 is used as the main ranking criterion because it gives every attack class equal weight.",
    )

    cross = performance.pivot(index="model", columns="dataset", values="macro_f1").reset_index()
    order = (
        cross.assign(avg=cross[["CICIDS2017", "UNSW-NB15"]].mean(axis=1))
        .sort_values("avg", ascending=True)["model"]
        .tolist()
    )

    chart_col, insight_col = st.columns([1.85, 0.85], gap="large")

    with chart_col:
        fig = go.Figure()
        fig.add_bar(
            y=cross["model"],
            x=cross["CICIDS2017"],
            name="CICIDS2017",
            orientation="h",
            marker_color=BLUE,
            hovertemplate="%{y}<br>CICIDS2017 Macro F1: %{x:.3f}<extra></extra>",
        )
        fig.add_bar(
            y=cross["model"],
            x=cross["UNSW-NB15"],
            name="UNSW-NB15",
            orientation="h",
            marker_color=AMBER,
            hovertemplate="%{y}<br>UNSW-NB15 Macro F1: %{x:.3f}<extra></extra>",
        )
        fig.update_layout(barmode="group", xaxis_range=[0, 1], bargap=0.26)
        fig.update_yaxes(categoryorder="array", categoryarray=order, title="")
        fig.update_xaxes(title="Macro F1-score")
        st.plotly_chart(clean_layout(fig, 455), use_container_width=True, config={"displayModeBar": False})

    with insight_col:
        st.markdown(
            """
            <div class="takeaway">
                <strong>01 · Most stable ranking</strong>
                <p>Random Forest ranked first by Macro F1 on both datasets.</p>
            </div>
            <div class="takeaway">
                <strong>02 · Accuracy is not enough</strong>
                <p>XGBoost reached the highest UNSW-NB15 accuracy, but Random Forest had the stronger class-balanced Macro F1.</p>
            </div>
            <div class="takeaway">
                <strong>03 · Neural models lagged</strong>
                <p>CNN and LSTM were not well matched to these engineered flow-level tabular representations under the tested training budgets.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="caution">
            <strong>Do not read the two datasets as a direct difficulty comparison.</strong>
            CICIDS2017 used an 80:20 split of a capped sample, while UNSW-NB15 retained its official test set.
            The cross-dataset view is useful for model ranking and robustness, not for claiming that one dataset is
            numerically “harder” than the other.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)

    section_head(
        "Interactive detail",
        "Inspect any metric within one dataset",
        "Switch the dataset and evaluation metric without leaving the overview.",
    )

    f1, f2, spacer = st.columns([0.8, 0.9, 2.3])
    with f1:
        dataset = st.selectbox("Dataset", ["CICIDS2017", "UNSW-NB15"], key="dataset_main")
    with f2:
        metric_label = st.selectbox(
            "Metric",
            ["Macro F1", "Accuracy", "Macro Recall", "Macro Precision", "Weighted F1"],
            key="metric_main",
        )

    metric_map = {
        "Macro F1": "macro_f1",
        "Accuracy": "accuracy",
        "Macro Recall": "macro_recall",
        "Macro Precision": "macro_precision",
        "Weighted F1": "weighted_f1",
    }
    metric = metric_map[metric_label]
    selected = performance[performance["dataset"] == dataset].sort_values(metric, ascending=False)

    bcol, tcol = st.columns([1.25, 1], gap="large")
    with bcol:
        fig2 = px.bar(
            selected.sort_values(metric, ascending=True),
            x=metric,
            y="model",
            orientation="h",
            text=metric,
        )
        fig2.update_traces(
            marker_color=BLUE,
            texttemplate="%{text:.3f}",
            textposition="outside",
            cliponaxis=False,
            hovertemplate=f"%{{y}}<br>{metric_label}: %{{x:.3f}}<extra></extra>",
        )
        fig2.update_layout(showlegend=False, xaxis_range=[0, 1])
        fig2.update_yaxes(title="")
        fig2.update_xaxes(title=metric_label)
        st.plotly_chart(clean_layout(fig2, 405), use_container_width=True, config={"displayModeBar": False})

    with tcol:
        display = selected[
            ["model", "accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]
        ].copy()
        display.columns = [
            "Model", "Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Weighted F1"
        ]
        st.dataframe(
            display.style.format({
                "Accuracy": "{:.3f}",
                "Macro Precision": "{:.3f}",
                "Macro Recall": "{:.3f}",
                "Macro F1": "{:.3f}",
                "Weighted F1": "{:.3f}",
            }),
            use_container_width=True,
            hide_index=True,
            height=346,
        )
        st.caption(
            "Macro F1 is the dissertation's primary ranking metric; weighted F1 and accuracy are retained for context."
        )

# ---------- Class performance ----------
with classes_tab:
    section_head(
        "Minority-class behaviour",
        "Aggregate scores hide very different class-level outcomes",
        "This view shows why Macro F1 and per-class recall matter in imbalanced intrusion detection.",
    )

    left, right = st.columns([1, 1], gap="large")

    with left:
        dataset_dist = st.radio(
            "Test support",
            ["CICIDS2017", "UNSW-NB15"],
            index=1,
            horizontal=True,
            key="support_dataset",
        )

        if dataset_dist == "CICIDS2017":
            dist = cicids_dist.rename(columns={"test_count": "support"}).copy()
            note = "Heartbleed (2), SQL Injection (4) and Infiltration (7) have extremely small test support."
        else:
            dist = unsw_dist.rename(columns={"count": "support"}).copy()
            note = "Worms has only 44 official test examples; Normal alone has 37,000."

        dist = dist.sort_values("support", ascending=True)
        fig3 = px.bar(dist, x="support", y="class", orientation="h")
        fig3.update_traces(
            marker_color=BLUE,
            hovertemplate="%{y}<br>Test support: %{x:,}<extra></extra>",
        )
        fig3.update_yaxes(title="")
        fig3.update_xaxes(title="Test-set support", type="log")
        st.plotly_chart(clean_layout(fig3, 500), use_container_width=True, config={"displayModeBar": False})
        st.caption(note + " The logarithmic axis keeps rare and dominant classes visible in the same chart.")

    with right:
        class_metric = st.selectbox(
            "UNSW-NB15 Random Forest class metric",
            ["F1-score", "Recall", "Precision"],
            key="class_metric",
        )
        cmap = {"F1-score": "f1_score", "Recall": "recall", "Precision": "precision"}
        cmetric = cmap[class_metric]

        rf = unsw_rf.sort_values(cmetric, ascending=True)
        fig4 = px.bar(rf, x=cmetric, y="class", orientation="h", text=cmetric)
        fig4.update_traces(
            marker_color=AMBER,
            texttemplate="%{text:.3f}",
            textposition="outside",
            cliponaxis=False,
            hovertemplate=f"%{{y}}<br>{class_metric}: %{{x:.3f}}<br>Support: %{{customdata[0]:,}}<extra></extra>",
            customdata=rf[["support"]],
        )
        fig4.update_layout(showlegend=False, xaxis_range=[0, 1])
        fig4.update_yaxes(title="")
        fig4.update_xaxes(title=class_metric)
        st.plotly_chart(clean_layout(fig4, 500), use_container_width=True, config={"displayModeBar": False})

    c1, c2, c3 = st.columns(3)
    c1.markdown(
        """
        <div class="mini-card">
            <h4>Strongest UNSW class</h4>
            <p><strong>Generic</strong> — F1 0.981 with recall 0.962 and support 18,871.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c2.markdown(
        """
        <div class="mini-card">
            <h4>Weak minority detection</h4>
            <p><strong>Analysis</strong> — F1 0.028; <strong>Backdoor</strong> — F1 0.055.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c3.markdown(
        """
        <div class="mini-card">
            <h4>Operational interpretation</h4>
            <p>An undetected attack is usually more serious than a false alarm, although excessive false alarms can still reduce operational usefulness.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------- Methodology ----------
with method_tab:
    section_head(
        "Experimental design",
        "What was actually compared",
        "The dashboard preserves the dissertation's methodological boundaries instead of presenting the benchmark as deployment-ready evidence.",
    )

    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("### CICIDS2017")
        st.dataframe(
            pd.DataFrame(
                [
                    ["Experimental sample", "31,193"],
                    ["Train / test", "24,954 / 6,239"],
                    ["Split", "80:20 stratified"],
                    ["Feature selection", "ANOVA F-score · top 40"],
                    ["Imbalance handling", "Capping + class weighting where supported"],
                    ["SMOTE", "Not used"],
                    ["Best Macro F1", "Random Forest · 0.892"],
                ],
                columns=["Setting", "Value"],
            ),
            use_container_width=True,
            hide_index=True,
        )

    with b:
        st.markdown("### UNSW-NB15")
        st.dataframe(
            pd.DataFrame(
                [
                    ["Training sample", "60,000"],
                    ["Official test set", "82,332 retained"],
                    ["Split", "Official train/test design"],
                    ["Feature selection", "ANOVA F-score · top 40"],
                    ["Imbalance handling", "Training down-sampling + model-specific weighting"],
                    ["SMOTE", "Not used"],
                    ["Best Macro F1", "Random Forest · 0.448"],
                ],
                columns=["Setting", "Value"],
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    st.markdown("### Interpretation boundaries")

    limitations = [
        ("Different test designs", "The two datasets are not used for a direct numerical difficulty comparison."),
        ("Capped CICIDS2017 sample", "Sampling changes the original class prevalence and may remove some majority-class variation."),
        ("Pre-split CICIDS imputation", "Median imputation before splitting creates a limited preprocessing-leakage risk."),
        ("Reduced XGBoost training subset", "The reported CICIDS2017 XGBoost run used 6,068 stratified training rows because of runtime constraints."),
        ("Unequal neural optimisation", "UNSW-NB15 MLP, CNN and LSTM used one epoch with batch size 8192."),
        ("No controlled ablation", "Feature selection and class weighting were not isolated experimentally, so causal credit is avoided."),
    ]

    rows = st.columns(3)
    for i, (title, body) in enumerate(limitations):
        with rows[i % 3]:
            st.markdown(
                f"""
                <div class="mini-card" style="margin-bottom:12px;">
                    <h4>{title}</h4>
                    <p>{body}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

st.markdown(
    """
    <div class="footer-note">
        Naman Saini · MSc Artificial Intelligence · University of Bath · 2026 · Dissertation mark: 72%
    </div>
    """,
    unsafe_allow_html=True,
)
