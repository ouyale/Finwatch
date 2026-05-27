"""
FinWatch - Consumer Financial Vulnerability Dashboard
======================================================
Streamlit dashboard for:
  • Live portfolio vulnerability overview
  • Intervention tier breakdown
  • Drift monitoring (PSI)
  • Monthly fairness audit results

Run with:
    streamlit run dashboard/app.py
or via docker-compose (port 8501).
"""

import os
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# -- Config --------------------------------------------------------------------

API_URL = os.getenv("API_URL", "http://localhost:8000")
ARTEFACT_DIR = Path(os.getenv("ARTEFACT_DIR", "data/processed"))

TIER_COLOURS = {
    "ESCALATE": "#E63946",   # red
    "OUTREACH":  "#F4A261",  # amber
    "MONITOR":   "#2A9D8F",  # teal
}

st.set_page_config(
    page_title="FinWatch | Vulnerability Intelligence",
    page_icon="🛡️",
    layout="wide",
)

# -- Sidebar ------------------------------------------------------------------─

st.sidebar.image(
    "https://img.shields.io/badge/FinWatch-v0.1.0-blue",
    use_column_width=True,
)
st.sidebar.title("FinWatch")
st.sidebar.caption("Consumer Vulnerability Early Warning")

page = st.sidebar.radio(
    "Navigate",
    ["📊 Portfolio Overview", "🔍 Score a Customer", "📈 Drift Monitor", "⚖️ Fairness Audit"],
)

# -- Helpers ------------------------------------------------------------------─


def api_health() -> dict:
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.json()
    except Exception:
        return {"status": "unreachable"}


def load_scored_portfolio() -> pd.DataFrame:
    """Load latest scored portfolio from processed data directory."""
    path = ARTEFACT_DIR / "latest_scored.parquet"
    if path.exists():
        return pd.read_parquet(path)
    # demo data when no real scores exist yet
    import numpy as np
    rng = np.random.default_rng(99)
    n = 500
    scores = rng.beta(2, 5, n)
    tiers = pd.cut(
        scores,
        bins=[-0.001, 0.40, 0.70, 1.0],
        labels=["MONITOR", "OUTREACH", "ESCALATE"],
    )
    return pd.DataFrame({
        "SK_ID_CURR": rng.integers(100000, 999999, n),
        "vulnerability_score": scores,
        "tier": tiers,
        "scored_at": datetime.utcnow().isoformat(),
    })


def load_psi_report() -> pd.DataFrame:
    """Load latest PSI report."""
    path = ARTEFACT_DIR / "latest_psi_report.parquet"
    if path.exists():
        return pd.read_parquet(path)
    # demo
    return pd.DataFrame({
        "feature": ["EXT_SOURCE_2", "credit_to_income_ratio", "AMT_ANNUITY", "DAYS_BIRTH"],
        "psi": [0.05, 0.12, 0.22, 0.08],
        "status": ["OK", "WARN", "ALERT", "OK"],
        "action": ["None", "Monitor", "Review / retrain", "None"],
    })


def load_fairness_log() -> pd.DataFrame:
    """Load monthly fairness audit log."""
    path = ARTEFACT_DIR / "fairness_log.parquet"
    if path.exists():
        return pd.read_parquet(path)
    # demo
    return pd.DataFrame({
        "month": ["2024-01", "2024-02", "2024-03", "2024-04"],
        "column": ["CODE_GENDER"] * 4,
        "dir": [0.92, 0.89, 0.85, 0.91],
        "passed": [True, True, True, True],
    })


# -- Health indicator ----------------------------------------------------------

health = api_health()
status_colour = "🟢" if health.get("status") == "ok" else "🔴"
st.sidebar.markdown(f"{status_colour} API: **{health.get('status', 'unknown')}**")
if health.get("model_version"):
    st.sidebar.caption(f"Model: {health['model_version']}")

# -- Page: Portfolio Overview --------------------------------------------------

if page == "📊 Portfolio Overview":
    st.title("Portfolio Vulnerability Overview")
    st.caption(f"Refreshed: {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC")

    df = load_scored_portfolio()

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)
    total = len(df)
    esc = (df["tier"] == "ESCALATE").sum()
    out = (df["tier"] == "OUTREACH").sum()
    mon = (df["tier"] == "MONITOR").sum()

    col1.metric("Total Customers Scored", f"{total:,}")
    col2.metric("🔴 ESCALATE", f"{esc:,}", f"{esc/total:.1%}")
    col3.metric("🟠 OUTREACH", f"{out:,}", f"{out/total:.1%}")
    col4.metric("🟢 MONITOR", f"{mon:,}", f"{mon/total:.1%}")

    st.markdown("---")

    col_left, col_right = st.columns([1, 2])

    with col_left:
        # Donut chart
        tier_counts = df["tier"].value_counts().reset_index()
        tier_counts.columns = ["tier", "count"]
        fig_donut = px.pie(
            tier_counts,
            values="count",
            names="tier",
            hole=0.55,
            color="tier",
            color_discrete_map=TIER_COLOURS,
            title="Tier Distribution",
        )
        fig_donut.update_traces(textposition="outside", textinfo="percent+label")
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_right:
        # Score distribution histogram
        fig_hist = px.histogram(
            df,
            x="vulnerability_score",
            color="tier",
            color_discrete_map=TIER_COLOURS,
            nbins=50,
            title="Vulnerability Score Distribution",
            labels={"vulnerability_score": "Score", "count": "Customers"},
        )
        fig_hist.add_vline(x=0.40, line_dash="dash", line_color="orange", annotation_text="Outreach threshold")
        fig_hist.add_vline(x=0.70, line_dash="dash", line_color="red", annotation_text="Escalate threshold")
        st.plotly_chart(fig_hist, use_container_width=True)

    # Top ESCALATE table
    st.subheader("Highest-Risk Customers (ESCALATE)")
    top_esc = (
        df[df["tier"] == "ESCALATE"]
        .sort_values("vulnerability_score", ascending=False)
        .head(20)
        .reset_index(drop=True)
    )
    st.dataframe(
        top_esc[["SK_ID_CURR", "vulnerability_score", "tier"]].style.format(
            {"vulnerability_score": "{:.3f}"}
        ),
        use_container_width=True,
    )

# -- Page: Score a Customer ----------------------------------------------------

elif page == "🔍 Score a Customer":
    st.title("Score a Single Customer")
    st.info("Provide customer attributes below. The model will return a vulnerability score, tier, and top explanatory factors.")

    with st.form("score_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            customer_id = st.text_input("Customer ID", value="CUST001")
            amt_income = st.number_input("Annual Income (£)", value=35000, step=1000)
            amt_credit = st.number_input("Credit Amount (£)", value=200000, step=5000)

        with col2:
            amt_annuity = st.number_input("Annual Annuity (£)", value=12000, step=500)
            amt_goods = st.number_input("Goods Price (£)", value=180000, step=5000)
            days_birth = st.number_input("Days Since Birth (negative)", value=-12000, step=100)

        with col3:
            days_employed = st.number_input("Days Employed (negative)", value=-1500, step=100)
            ext_source_2 = st.slider("EXT_SOURCE_2", 0.0, 1.0, 0.5)
            gender = st.selectbox("Gender", ["M", "F"])

        submitted = st.form_submit_button("Score Customer")

    if submitted:
        payload = {
            "SK_ID_CURR": customer_id,
            "AMT_INCOME_TOTAL": amt_income,
            "AMT_CREDIT": amt_credit,
            "AMT_ANNUITY": amt_annuity,
            "AMT_GOODS_PRICE": amt_goods,
            "DAYS_BIRTH": days_birth,
            "DAYS_EMPLOYED": days_employed,
            "EXT_SOURCE_2": ext_source_2,
            "CODE_GENDER": gender,
        }

        with st.spinner("Scoring..."):
            try:
                resp = requests.post(
                    f"{API_URL}/score/single",
                    json=payload,
                    timeout=15,
                )
                if resp.status_code == 200:
                    result = resp.json()

                    tier = result["tier"]
                    score = result["vulnerability_score"]
                    colour = TIER_COLOURS[tier]

                    st.markdown(
                        f"""
                        <div style='background:{colour}22; border-left:6px solid {colour};
                                    padding:16px; border-radius:6px;'>
                            <h2 style='color:{colour}; margin:0'>{tier}</h2>
                            <p style='font-size:1.4rem; margin:4px 0'>
                                Vulnerability score: <b>{score:.3f}</b>
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.subheader("Top Contributing Factors")
                    shap_data = result.get("top_shap_features", [])
                    if shap_data:
                        shap_df = pd.DataFrame(shap_data)
                        fig_shap = px.bar(
                            shap_df.sort_values("shap_value"),
                            x="shap_value",
                            y="feature",
                            orientation="h",
                            color="direction",
                            color_discrete_map={"increases_risk": "#E63946", "decreases_risk": "#2A9D8F"},
                            title="SHAP Feature Contributions",
                        )
                        st.plotly_chart(fig_shap, use_container_width=True)
                    else:
                        st.info("No SHAP data returned.")

                else:
                    st.error(f"API error {resp.status_code}: {resp.text}")

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to FinWatch API. Is it running? (docker-compose up)")

# -- Page: Drift Monitor ------------------------------------------------------─

elif page == "📈 Drift Monitor":
    st.title("Feature Drift Monitor (PSI)")
    st.caption("Population Stability Index - measures how much the distribution of each feature has shifted since training.")

    df_psi = load_psi_report()

    # Threshold guide
    cols = st.columns(3)
    cols[0].metric("PSI < 0.10", "No change ✅")
    cols[1].metric("0.10 – 0.20", "Monitor ⚠️")
    cols[2].metric("PSI > 0.20", "Retrain 🔴")

    st.markdown("---")

    # Colour-coded table
    def colour_status(val):
        colours = {"OK": "background-color:#2A9D8F22", "WARN": "background-color:#F4A26122", "ALERT": "background-color:#E6394622"}
        return colours.get(val, "")

    styled = df_psi.style.applymap(colour_status, subset=["status"])
    st.dataframe(styled, use_container_width=True)

    # Bar chart
    fig_psi = px.bar(
        df_psi.sort_values("psi", ascending=False),
        x="feature",
        y="psi",
        color="status",
        color_discrete_map={"OK": "#2A9D8F", "WARN": "#F4A261", "ALERT": "#E63946"},
        title="PSI by Feature",
    )
    fig_psi.add_hline(y=0.10, line_dash="dash", line_color="orange", annotation_text="Warn threshold")
    fig_psi.add_hline(y=0.20, line_dash="dash", line_color="red", annotation_text="Alert threshold")
    st.plotly_chart(fig_psi, use_container_width=True)

    retrain_needed = (df_psi["psi"] >= 0.20).any()
    if retrain_needed:
        st.error("⚠️ One or more features exceed PSI 0.20 - retraining recommended.")
    else:
        st.success("✅ All features within acceptable drift bounds.")

# -- Page: Fairness Audit ------------------------------------------------------

elif page == "⚖️ Fairness Audit":
    st.title("Monthly Fairness Audit")
    st.caption(
        "Disparate Impact Ratio (DIR) per protected group. "
        "FCA Consumer Duty compliance requires DIR ≥ 0.80 (4/5ths rule). "
        "Models failing this gate are not deployed."
    )

    df_fair = load_fairness_log()

    # Latest month card
    if not df_fair.empty:
        latest = df_fair[df_fair["month"] == df_fair["month"].max()]
        for _, row in latest.iterrows():
            passed_icon = "✅" if row["passed"] else "❌"
            st.metric(
                label=f"{passed_icon} {row['column']} - DIR (latest: {row['month']})",
                value=f"{row['dir']:.3f}",
                delta=f"{'PASSED' if row['passed'] else 'FAILED'} - threshold 0.80",
            )

    st.markdown("---")

    # Trend chart
    fig_dir = px.line(
        df_fair,
        x="month",
        y="dir",
        color="column",
        markers=True,
        title="Disparate Impact Ratio Over Time",
        labels={"dir": "DIR", "month": "Month"},
    )
    fig_dir.add_hline(y=0.80, line_dash="dash", line_color="red", annotation_text="FCA minimum (0.80)")
    st.plotly_chart(fig_dir, use_container_width=True)

    # Raw table
    st.subheader("Audit Log")
    st.dataframe(df_fair, use_container_width=True)

    st.info(
        "**How DIR is computed:** For each protected attribute, we divide the "
        "ESCALATE rate of the disadvantaged group by that of the most-advantaged group. "
        "A ratio below 0.80 constitutes prohibited disparate impact under the 4/5ths rule."
    )
