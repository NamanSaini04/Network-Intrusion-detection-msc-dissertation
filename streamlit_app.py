from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dashboard_data"

st.set_page_config(
    page_title="Network Intrusion Detection | MSc Dissertation",
    page_icon="🛡️",
    layout="wide",
)

@st.cache_data
def load_data():
    performance = pd.read_csv(DATA / "model_performance.csv")
    cicids_dist = pd.read_csv(DATA / "cicids_class_distribution.csv")
    unsw_dist = pd.read_csv(DATA / "unsw_test_distribution.csv")
    unsw_rf = pd.read_csv(DATA / "unsw_rf_attack_specific.csv")
    return performance, cicids_dist, unsw_dist, unsw_rf

performance, cicids_dist, unsw_dist, unsw_rf = load_data()

st.title("🛡️ Network Intrusion Detection — Dissertation Results Dashboard")
st.caption(
    "MSc Artificial Intelligence, University of Bath · Naman Saini · "
    "Comparative ML/DL evaluation on CICIDS2017 and UNSW-NB15"
)

st.info(
    "Interpretation note: CICIDS2017 and UNSW-NB15 used different sampling and test designs. "
    "Absolute scores are therefore not treated as directly comparable measures of dataset difficulty. "
    "The cross-dataset view is used to inspect ranking stability, minority-class behaviour and dataset dependence."
)

cicids = performance[performance["dataset"] == "CICIDS2017"].copy()
unsw = performance[performance["dataset"] == "UNSW-NB15"].copy()

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Models evaluated", "7")
k2.metric("Datasets", "2")
k3.metric("Selected features", "40")
k4.metric("Best CICIDS2017 Macro F1", f"{cicids['macro_f1'].max():.3f}", "Random Forest")
k5.metric("Best UNSW-NB15 Macro F1", f"{unsw['macro_f1'].max():.3f}", "Random Forest")

st.divider()

st.subheader("Cross-dataset model ranking")
cross = performance.pivot(index="model", columns="dataset", values="macro_f1").reset_index()
order = (
    cross.assign(rank_score=cross[["CICIDS2017", "UNSW-NB15"]].mean(axis=1))
    .sort_values("rank_score", ascending=True)["model"]
    .tolist()
)
fig = go.Figure()
fig.add_bar(
    y=cross["model"], x=cross["CICIDS2017"],
    name="CICIDS2017", orientation="h",
    customdata=cross[["CICIDS2017"]],
    hovertemplate="%{y}<br>CICIDS2017 Macro F1: %{x:.3f}<extra></extra>",
)
fig.add_bar(
    y=cross["model"], x=cross["UNSW-NB15"],
    name="UNSW-NB15", orientation="h",
    hovertemplate="%{y}<br>UNSW-NB15 Macro F1: %{x:.3f}<extra></extra>",
)
fig.update_layout(
    barmode="group",
    height=470,
    xaxis_title="Macro F1-score",
    yaxis_title="",
    xaxis_range=[0, 1],
    legend_title_text="Dataset",
    margin=dict(l=10, r=10, t=20, b=10),
)
fig.update_yaxes(categoryorder="array", categoryarray=order)
st.plotly_chart(fig, use_container_width=True)

st.markdown(
    "**Key result:** Random Forest ranked first by Macro F1 on both benchmarks. "
    "XGBoost achieved the highest UNSW-NB15 accuracy and weighted F1, but Random Forest produced the stronger "
    "class-balanced Macro F1 result."
)

st.divider()

left, right = st.columns([1, 1.65])
with left:
    st.subheader("Explore one dataset")
    dataset = st.selectbox("Dataset", ["CICIDS2017", "UNSW-NB15"])
    metric_label = st.selectbox(
        "Ranking metric",
        ["Macro F1", "Accuracy", "Macro Recall", "Macro Precision", "Weighted F1"],
        index=0,
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
    best = selected.iloc[0]
    st.metric(
        f"Top model by {metric_label}",
        best["model"],
        f"{best[metric]:.3f}",
    )
    st.caption(
        "Macro F1 is the dissertation's primary ranking metric because it gives each class equal weight "
        "and is less likely than accuracy or weighted F1 to hide weak minority-class performance."
    )

with right:
    fig2 = px.bar(
        selected.sort_values(metric, ascending=True),
        x=metric,
        y="model",
        orientation="h",
        text=metric,
        title=f"{dataset}: {metric_label} by model",
    )
    fig2.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig2.update_layout(
        height=430,
        xaxis_title=metric_label,
        yaxis_title="",
        xaxis_range=[0, 1],
        showlegend=False,
        margin=dict(l=10, r=40, t=50, b=10),
    )
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("Full model performance table")
display_cols = [
    "model", "accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"
]
table = selected[display_cols].rename(columns={
    "model": "Model",
    "accuracy": "Accuracy",
    "macro_precision": "Macro Precision",
    "macro_recall": "Macro Recall",
    "macro_f1": "Macro F1",
    "weighted_f1": "Weighted F1",
})
st.dataframe(
    table.style.format({
        "Accuracy": "{:.3f}",
        "Macro Precision": "{:.3f}",
        "Macro Recall": "{:.3f}",
        "Macro F1": "{:.3f}",
        "Weighted F1": "{:.3f}",
    }),
    use_container_width=True,
    hide_index=True,
)

st.divider()

st.subheader("Class imbalance and test support")
d1, d2 = st.columns(2)

with d1:
    dist_dataset = st.radio(
        "Class-distribution view",
        ["CICIDS2017 test partition", "UNSW-NB15 official test set"],
        horizontal=True,
    )
    if dist_dataset.startswith("CICIDS"):
        dist = cicids_dist.copy()
        title = "CICIDS2017 test support"
        support_col = "test_count"
        note = (
            "CICIDS2017 test support is extremely small for Heartbleed (2), SQL Injection (4) "
            "and Infiltration (7), so strong conclusions for those classes are avoided."
        )
    else:
        dist = unsw_dist.copy()
        title = "UNSW-NB15 official test support"
        support_col = "count"
        note = (
            "The official UNSW-NB15 test set is dominated by Normal, Generic and Exploits. "
            "Worms has only 44 examples, making its class-specific metrics particularly unstable."
        )

    dist_plot = dist.sort_values(support_col, ascending=True)
    fig3 = px.bar(
        dist_plot,
        x=support_col,
        y="class",
        orientation="h",
        title=title,
        labels={support_col: "Test records", "class": ""},
        log_x=True,
    )
    fig3.update_layout(height=520, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig3, use_container_width=True)
    st.caption(note)

with d2:
    st.markdown("#### UNSW-NB15 Random Forest: attack-specific performance")
    rf_long = unsw_rf.melt(
        id_vars=["class", "support"],
        value_vars=["precision", "recall", "f1_score"],
        var_name="metric",
        value_name="score",
    )
    metric_names = {
        "precision": "Precision",
        "recall": "Recall",
        "f1_score": "F1-score",
    }
    rf_long["metric"] = rf_long["metric"].map(metric_names)
    fig4 = px.bar(
        rf_long,
        x="score",
        y="class",
        color="metric",
        barmode="group",
        orientation="h",
        title="Per-class precision, recall and F1",
        labels={"score": "Score", "class": "", "metric": "Metric"},
    )
    fig4.update_layout(
        height=520,
        xaxis_range=[0, 1],
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig4, use_container_width=True)
    st.caption(
        "Random Forest is strong on Generic, Reconnaissance and Exploits, but weak on Analysis and Backdoor. "
        "This is why the dashboard does not rely on overall accuracy alone."
    )

st.divider()

st.subheader("What the results mean")
i1, i2, i3 = st.columns(3)
with i1:
    st.markdown("#### 🌳 Tree ensembles")
    st.write(
        "Random Forest produced the strongest Macro F1 on both benchmarks. "
        "Its performance is consistent with tree ensembles being well suited to structured, engineered flow-level features."
    )
with i2:
    st.markdown("#### ⚡ XGBoost")
    st.write(
        "XGBoost remained competitive and achieved the highest UNSW-NB15 accuracy and weighted F1, "
        "but it did not exceed Random Forest on the primary class-balanced Macro F1 criterion."
    )
with i3:
    st.markdown("#### 🧠 CNN / LSTM")
    st.write(
        "The neural results are interpreted cautiously. Flow-level tabular features do not provide genuine image-like "
        "spatial structure or true temporal sequences, and neural training budgets were constrained."
    )

st.divider()

st.subheader("Experimental design and limitations")
design = pd.DataFrame([
    {
        "Aspect": "Test design",
        "CICIDS2017": "80:20 stratified split of capped experimental sample",
        "UNSW-NB15": "Official test set retained",
    },
    {
        "Aspect": "Training sample",
        "CICIDS2017": "24,954 training records after capping/split",
        "UNSW-NB15": "60,000 proportional stratified training records",
    },
    {
        "Aspect": "Feature selection",
        "CICIDS2017": "ANOVA F-score, top 40",
        "UNSW-NB15": "ANOVA F-score, top 40",
    },
    {
        "Aspect": "Imbalance handling",
        "CICIDS2017": "Capped sampling + class weighting where supported",
        "UNSW-NB15": "Training down-sampling + model-specific weighting",
    },
    {
        "Aspect": "SMOTE",
        "CICIDS2017": "Discussed, not used",
        "UNSW-NB15": "Discussed, not used",
    },
])
st.dataframe(design, use_container_width=True, hide_index=True)

with st.expander("Important limitations — read before interpreting the charts", expanded=True):
    st.markdown(
        """
- **No direct dataset-difficulty claim:** CICIDS2017 and UNSW-NB15 used different evaluation designs.
- **CICIDS2017 class prevalence changed** because capped sampling was used.
- **CICIDS2017 median imputation occurred before the 80:20 split** in the supplied implementation, creating a limited preprocessing-leakage risk.
- The reported **CICIDS2017 XGBoost run used a reduced 6,068-row stratified training subset** because of runtime constraints.
- Neural optimisation was **not standardised across datasets**; the UNSW-NB15 MLP, CNN and LSTM runs used one epoch with batch size 8192.
- Feature selection and class weighting were **not isolated with controlled ablation studies**.
- These are benchmark experiments, **not evidence of deployment readiness on live enterprise traffic**.
        """
    )

st.divider()
st.caption(
    "Source: Naman Saini, MSc Artificial Intelligence dissertation, University of Bath (2026). "
    "Dashboard values reproduce the submitted result tables; interpretation retains the dissertation's stated limitations."
)
