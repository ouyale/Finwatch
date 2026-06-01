"""
FinWatch - Consumer Financial Vulnerability Dashboard
======================================================
Streamlit dashboard for:
  - Live portfolio vulnerability overview
  - Intervention tier breakdown
  - Drift monitoring (PSI)
  - Monthly fairness audit results

Run with:
    streamlit run dashboard/app.py
or via docker-compose (port 8501).
"""

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# -- Config --------------------------------------------------------------------

API_URL = os.getenv("API_URL", "http://localhost:8000")
ARTEFACT_DIR = Path(os.getenv("ARTEFACT_DIR", "data/processed"))

# Lloyds-inspired palette - light theme
GREEN     = "#006A4D"   # Lloyds dark green - primary brand colour
GREEN_LT  = "#00A877"   # lighter green for hover/accents
AMBER     = "#D4700A"   # outreach
RED       = "#B91C1C"   # escalate
GREY_TEXT = "#475569"
BORDER    = "#E2E8F0"
CARD_BG   = "#FFFFFF"
PAGE_BG   = "#F1F5F9"

TIER_COLOURS = {
    "ESCALATE": RED,
    "OUTREACH":  AMBER,
    "MONITOR":   GREEN,
}

# Load real thresholds from artefact dir if available
_thresh_path = ARTEFACT_DIR / "thresholds.json"
if _thresh_path.exists():
    with open(_thresh_path) as _f:
        _thresh = json.load(_f)
    THRESHOLD_ESCALATE = _thresh.get("threshold_escalate", 0.10)
    THRESHOLD_OUTREACH = _thresh.get("threshold_outreach", 0.06)
else:
    THRESHOLD_ESCALATE = 0.10
    THRESHOLD_OUTREACH = 0.06

# -- Page config ---------------------------------------------------------------

st.set_page_config(
    page_title="FinWatch",
    page_icon="shield",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- Global CSS ----------------------------------------------------------------

st.markdown(
    f"""
    <style>
    /* Hide Streamlit default header and footer */
    header[data-testid="stHeader"] {{ display: none !important; }}
    #MainMenu {{ display: none !important; }}
    footer {{ display: none !important; }}
    [data-testid="stToolbar"] {{ display: none !important; }}

    /* Backgrounds */
    .stApp {{ background-color: {PAGE_BG}; }}
    [data-testid="stSidebar"] {{
        background-color: {CARD_BG};
        border-right: 1px solid {BORDER};
    }}

    /* Remove top padding */
    .block-container {{ padding-top: 1.8rem; padding-bottom: 2rem; }}

    /* Metric cards */
    [data-testid="metric-container"] {{
        background: {CARD_BG};
        border: 1px solid {BORDER};
        border-radius: 8px;
        padding: 14px 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }}
    [data-testid="metric-container"] label {{
        color: {GREY_TEXT} !important;
        font-size: 0.75rem !important;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        font-weight: 600 !important;
    }}
    [data-testid="metric-container"] [data-testid="stMetricValue"] {{
        color: #1A202C !important;
        font-size: 1.7rem !important;
        font-weight: 700 !important;
    }}

    /* Headings */
    h1, h2, h3 {{ color: #1A202C !important; }}
    h2 {{ font-size: 1.3rem !important; font-weight: 700 !important; }}
    h3 {{ font-size: 1rem !important; font-weight: 600 !important; }}

    /* Dividers */
    hr {{ border-color: {BORDER} !important; margin: 1rem 0; }}

    /* Text */
    p, li {{ color: #374151; }}
    .stCaption {{ color: {GREY_TEXT} !important; font-size: 0.78rem !important; }}

    /* Forms */
    [data-testid="stForm"] {{
        background: {CARD_BG};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }}

    /* Buttons */
    .stFormSubmitButton button {{
        background-color: {GREEN} !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        letter-spacing: 0.03em;
    }}
    .stFormSubmitButton button:hover {{
        background-color: {GREEN_LT} !important;
    }}

    /* Dataframe */
    [data-testid="stDataFrame"] {{ border-radius: 8px; overflow: hidden; }}

    /* Radio buttons - sidebar nav */
    [data-testid="stRadio"] label {{
        color: #374151 !important;
        font-size: 0.9rem !important;
        padding: 5px 0;
    }}
    [data-testid="stRadio"] label:hover {{
        color: {GREEN} !important;
    }}
    [data-testid="stRadio"] [data-checked="true"] label {{
        color: {GREEN} !important;
        font-weight: 600 !important;
    }}
    [data-testid="stRadio"] [data-checked="true"] span[data-baseweb="radio"] div {{
        background-color: {GREEN} !important;
        border-color: {GREEN} !important;
    }}

    /* Alerts */
    .stAlert {{ border-radius: 8px !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -- Sidebar -------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        f"""
        <div style='padding:12px 0 24px 0;'>
            <div style='font-size:1.5rem;font-weight:800;letter-spacing:-0.02em;'>
                <span style='color:{GREEN};'>Fin</span><span style='color:#FFFFFF;'>Watch</span>
            </div>
            <div style='font-size:0.68rem;color:{GREY_TEXT};letter-spacing:0.1em;
                        text-transform:uppercase;margin-top:2px;color:#64748B;'>
                Vulnerability Intelligence
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "",
        ["Portfolio Overview", "Score a Customer", "Drift Monitor", "Fairness Audit"],
        format_func=lambda x: {
            "Portfolio Overview": "  Portfolio Overview",
            "Score a Customer":   "  Score a Customer",
            "Drift Monitor":      "  Drift Monitor",
            "Fairness Audit":     "  Fairness Audit",
        }[x],
    )

    st.markdown(f"<hr style='border-color:{BORDER};margin:16px 0'/>", unsafe_allow_html=True)

    # API health indicator
    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        api_ok = health.get("status") == "ok"
    except Exception:
        health = {}
        api_ok = False

    dot   = GREEN if api_ok else RED
    label = "API online" if api_ok else "API offline"
    st.markdown(
        f"<p style='font-size:0.78rem;color:{GREY_TEXT};margin:0;'>"
        f"<span style='color:{dot};'>&#9679;</span> {label}</p>",
        unsafe_allow_html=True,
    )
    if health.get("model_version"):
        st.markdown(
            f"<p style='font-size:0.72rem;color:{GREY_TEXT};opacity:0.6;margin:2px 0;'>"
            f"Model v{health['model_version']}</p>",
            unsafe_allow_html=True,
        )

    st.markdown(
        f"<p style='font-size:0.68rem;color:#CBD5E0;margin-top:32px;line-height:1.6;'>"
        "FCA Consumer Duty aligned<br/>UK GDPR Art.22 compliant<br/>PRA SS1/23 compatible</p>",
        unsafe_allow_html=True,
    )


# -- Helpers -------------------------------------------------------------------


def dark_chart(fig, title="", height=380):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#F8FAFC",
        font=dict(color="#374151", family="Inter, sans-serif", size=11),
        title=dict(text=title, font=dict(color="#1A202C", size=13), x=0),
        xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER),
        yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=GREY_TEXT)),
        margin=dict(t=44, b=36, l=40, r=20),
        height=height,
    )
    return fig


def load_scored_portfolio() -> pd.DataFrame:
    path = ARTEFACT_DIR / "latest_scored.parquet"
    if path.exists():
        return pd.read_parquet(path)
    rng = np.random.default_rng(99)
    n = 500
    scores = rng.beta(1.5, 18, n)
    tiers = pd.cut(
        scores,
        bins=[-0.001, THRESHOLD_OUTREACH, THRESHOLD_ESCALATE, 1.0],
        labels=["MONITOR", "OUTREACH", "ESCALATE"],
    )
    return pd.DataFrame({
        "SK_ID_CURR": rng.integers(100000, 999999, n),
        "vulnerability_score": scores,
        "tier": tiers,
        "scored_at": datetime.utcnow().isoformat(),
    })


def load_psi_report() -> pd.DataFrame:
    path = ARTEFACT_DIR / "latest_psi_report.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame({
        "feature": ["EXT_SOURCE_2", "credit_to_income_ratio", "AMT_ANNUITY", "DAYS_BIRTH"],
        "psi":     [0.05, 0.12, 0.22, 0.08],
        "status":  ["OK", "WARN", "ALERT", "OK"],
        "action":  ["None", "Monitor", "Review / retrain", "None"],
    })


def load_fairness_log() -> pd.DataFrame:
    path = ARTEFACT_DIR / "fairness_log.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame({
        "month":  ["2024-01", "2024-02", "2024-03", "2024-04"] * 2,
        "column": ["CODE_GENDER"] * 4 + ["NAME_FAMILY_STATUS"] * 4,
        "dir":    [0.92, 0.89, 0.95, 0.91, 0.88, 0.85, 0.87, 0.86],
        "passed": [True] * 8,
    })


def stat_card(label, value, colour=GREEN):
    return (
        f"<div style='background:{CARD_BG};border:1px solid {BORDER};border-radius:8px;"
        f"padding:16px 20px;text-align:center;'>"
        f"<div style='font-size:0.72rem;color:{GREY_TEXT};text-transform:uppercase;"
        f"letter-spacing:0.06em;font-weight:600;'>{label}</div>"
        f"<div style='font-size:1.8rem;font-weight:800;color:{colour};margin:4px 0;'>{value}</div>"
        f"</div>"
    )


# -- Portfolio Overview --------------------------------------------------------

if page == "Portfolio Overview":
    st.markdown("## Portfolio Vulnerability Overview")
    st.caption(f"Demo data  |  {datetime.utcnow().strftime('%d %b %Y %H:%M')} UTC")

    df = load_scored_portfolio()
    total = len(df)
    esc = (df["tier"] == "ESCALATE").sum()
    out = (df["tier"] == "OUTREACH").sum()
    mon = (df["tier"] == "MONITOR").sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers Scored", f"{total:,}")
    c2.metric("ESCALATE", f"{esc:,}", f"{esc/total:.1%}", delta_color="inverse")
    c3.metric("OUTREACH",  f"{out:,}", f"{out/total:.1%}", delta_color="inverse")
    c4.metric("MONITOR",   f"{mon:,}", f"{mon/total:.1%}")

    st.markdown("<hr/>", unsafe_allow_html=True)

    left, right = st.columns([1, 2])

    with left:
        tier_vc = df["tier"].value_counts().reset_index()
        tier_vc.columns = ["tier", "count"]
        fig_donut = px.pie(
            tier_vc, values="count", names="tier",
            hole=0.62, color="tier", color_discrete_map=TIER_COLOURS,
        )
        fig_donut.update_traces(
            textposition="outside", textinfo="percent+label",
            textfont=dict(color="#C5D0DE", size=11),
        )
        fig_donut = dark_chart(fig_donut, "Tier Distribution", height=340)
        fig_donut.update_layout(
            showlegend=False,
            margin=dict(t=44, b=60, l=60, r=60),
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with right:
        fig_hist = px.histogram(
            df, x="vulnerability_score", color="tier",
            color_discrete_map=TIER_COLOURS, nbins=50, barmode="overlay",
            labels={"vulnerability_score": "Vulnerability Score"},
            opacity=0.85,
        )
        fig_hist.add_vline(
            x=THRESHOLD_OUTREACH, line_dash="dot", line_color=AMBER,
            annotation_text=f"Outreach {THRESHOLD_OUTREACH}",
            annotation_font_color=AMBER, annotation_position="top right",
        )
        fig_hist.add_vline(
            x=THRESHOLD_ESCALATE, line_dash="dot", line_color=RED,
            annotation_text=f"Escalate {THRESHOLD_ESCALATE}",
            annotation_font_color=RED, annotation_position="top right",
        )
        fig_hist = dark_chart(fig_hist, "Vulnerability Score Distribution", height=340)
        st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("#### Highest-Risk Customers")
    top_esc = (
        df[df["tier"] == "ESCALATE"]
        .sort_values("vulnerability_score", ascending=False)
        .head(15)
        .reset_index(drop=True)
    )
    top_esc.index += 1
    st.dataframe(
        top_esc[["SK_ID_CURR", "vulnerability_score", "tier"]].style
        .format({"vulnerability_score": "{:.4f}"}),
        use_container_width=True,
    )


# -- Score a Customer ----------------------------------------------------------

elif page == "Score a Customer":
    st.markdown("## Score a Customer")
    st.caption(
        "Enter known customer attributes. Any field left at its default will use "
        "a population-average value - you do not need all 122 raw fields."
    )

    with st.form("score_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Identity**")
            customer_id     = st.number_input("Customer ID", value=100001, step=1)
            gender          = st.selectbox("Gender", ["M", "F"])
            name_income_type = st.selectbox(
                "Income Type",
                ["Working", "Commercial associate", "Pensioner", "State servant"],
            )

        with col2:
            st.markdown("**Financials**")
            amt_income   = st.number_input("Annual Income (£)", value=35000,  step=1000)
            amt_credit   = st.number_input("Credit Amount (£)", value=200000, step=5000)
            amt_annuity  = st.number_input("Annual Annuity (£)", value=12000,  step=500)
            amt_goods    = st.number_input("Goods Price (£)", value=180000, step=5000)

        with col3:
            st.markdown("**Bureau & Employment**")
            ext_source_2  = st.slider("Credit Bureau Score (EXT_SOURCE_2)", 0.0, 1.0, 0.5, 0.01)
            days_birth    = st.number_input("Days Since Birth (negative)", value=-12000, step=365)
            days_employed = st.number_input("Days Employed (negative)", value=-1500, step=100)

        submitted = st.form_submit_button("Score Customer", use_container_width=True)

    if submitted:
        payload = {
            "SK_ID_CURR":      int(customer_id),
            "AMT_INCOME_TOTAL": amt_income,
            "AMT_CREDIT":       amt_credit,
            "AMT_ANNUITY":      amt_annuity,
            "AMT_GOODS_PRICE":  amt_goods,
            "DAYS_BIRTH":       int(days_birth),
            "DAYS_EMPLOYED":    int(days_employed),
            "EXT_SOURCE_2":     float(ext_source_2),
            "CODE_GENDER":      gender,
            "NAME_INCOME_TYPE": name_income_type,
        }

        with st.spinner("Scoring..."):
            try:
                resp = requests.post(f"{API_URL}/score/single", json=payload, timeout=15)
                if resp.status_code == 200:
                    result = resp.json()
                    tier   = result["tier"]
                    score  = result["vulnerability_score"]
                    colour = TIER_COLOURS[tier]

                    action = {
                        "ESCALATE": "Immediate referral to specialist vulnerability team",
                        "OUTREACH": "Proactive contact - offer support products or payment holiday",
                        "MONITOR":  "Continue standard monitoring",
                    }[tier]

                    st.markdown(
                        f"""
                        <div style='background:{colour}14;border-left:4px solid {colour};
                                    border-radius:0 8px 8px 0;padding:18px 22px;margin:16px 0;'>
                            <div style='font-size:0.72rem;color:{colour};font-weight:700;
                                        letter-spacing:0.1em;text-transform:uppercase;'>
                                Intervention Tier
                            </div>
                            <div style='font-size:2.2rem;font-weight:800;color:{colour};
                                        line-height:1.1;margin:6px 0 4px;'>
                                {tier}
                            </div>
                            <div style='font-size:0.85rem;color:{GREY_TEXT};margin-bottom:12px;'>
                                {action}
                            </div>
                            <div style='font-size:0.82rem;color:#C5D0DE;'>
                                Score: <b style='color:{colour}'>{score:.4f}</b>
                                &ensp;&bull;&ensp;
                                Escalate threshold: {result['threshold_escalate']:.2f}
                                &ensp;&bull;&ensp;
                                Outreach threshold: {result['threshold_outreach']:.2f}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    shap_data = result.get("top_shap_features", [])
                    if shap_data:
                        st.markdown("#### Top Explanatory Factors")
                        shap_df = pd.DataFrame(shap_data)
                        bar_colours = shap_df["direction"].map(
                            {"increases_risk": RED, "reduces_risk": GREEN}
                        )
                        fig_shap = go.Figure(go.Bar(
                            x=shap_df["shap_value"],
                            y=shap_df["feature"],
                            orientation="h",
                            marker_color=bar_colours,
                            text=shap_df["shap_value"].round(4).astype(str),
                            textposition="outside",
                            textfont=dict(color="#C5D0DE", size=11),
                        ))
                        fig_shap.add_vline(x=0, line_color=BORDER, line_width=1)
                        fig_shap = dark_chart(
                            fig_shap,
                            "SHAP Feature Contributions",
                            height=280,
                        )
                        fig_shap.update_layout(
                            xaxis_title="SHAP Value (positive = increases vulnerability risk)",
                            yaxis_title="",
                        )
                        st.plotly_chart(fig_shap, use_container_width=True)
                        st.caption(
                            "SHAP values show which features pushed this customer's score up or "
                            "down relative to the average customer. Provided under UK GDPR Article 22."
                        )
                else:
                    st.error(f"API error {resp.status_code}: {resp.text}")

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to the FinWatch API. Is docker-compose running?")


# -- Drift Monitor -------------------------------------------------------------

elif page == "Drift Monitor":
    st.markdown("## Feature Drift Monitor")
    st.caption(
        "Population Stability Index (PSI) measures how much each feature's distribution "
        "has shifted since training. A large shift means the model may be scoring customers "
        "in conditions it was not trained on."
    )

    df_psi = load_psi_report()

    c1, c2, c3 = st.columns(3)
    for col, label, val, clr in [
        (c1, "No action", "PSI < 0.10", GREEN),
        (c2, "Monitor",   "0.10 - 0.20", AMBER),
        (c3, "Retrain",   "PSI > 0.20",  RED),
    ]:
        col.markdown(
            f"<div style='background:{clr}12;border:1px solid {clr}40;"
            f"border-radius:8px;padding:12px;text-align:center;'>"
            f"<div style='color:{clr};font-weight:700;font-size:0.82rem;'>{val}</div>"
            f"<div style='color:{GREY_TEXT};font-size:0.75rem;margin-top:2px;'>{label}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br/>", unsafe_allow_html=True)

    colour_map = {"OK": GREEN, "WARN": AMBER, "ALERT": RED}
    fig_psi = go.Figure()
    for status in ["OK", "WARN", "ALERT"]:
        sub = df_psi[df_psi["status"] == status]
        if not sub.empty:
            fig_psi.add_trace(go.Bar(
                x=sub["feature"], y=sub["psi"],
                name=status, marker_color=colour_map[status],
            ))
    fig_psi.add_hline(y=0.10, line_dash="dot", line_color=AMBER,
                      annotation_text="Warn 0.10", annotation_font_color=AMBER)
    fig_psi.add_hline(y=0.20, line_dash="dot", line_color=RED,
                      annotation_text="Alert 0.20", annotation_font_color=RED)
    fig_psi = dark_chart(fig_psi, "PSI by Feature")
    fig_psi.update_layout(xaxis_title="Feature", yaxis_title="PSI", barmode="group")
    st.plotly_chart(fig_psi, use_container_width=True)

    st.markdown("#### Feature Summary")
    st.dataframe(df_psi, use_container_width=True)

    if (df_psi["psi"] >= 0.20).any():
        st.error("One or more features exceed PSI 0.20 - retraining recommended.")
    else:
        st.success("All features within acceptable drift bounds.")


# -- Fairness Audit ------------------------------------------------------------

elif page == "Fairness Audit":
    st.markdown("## Fairness Audit Log")
    st.caption(
        "Disparate Impact Ratio (DIR) per protected characteristic, tracked monthly. "
        "FCA Consumer Duty requires DIR >= 0.80 for all protected groups. "
        "Models failing this gate are blocked from deployment regardless of accuracy."
    )

    df_fair = load_fairness_log()

    if not df_fair.empty:
        latest_month = df_fair["month"].max()
        latest = df_fair[df_fair["month"] == latest_month]
        st.markdown(f"#### Latest results &nbsp; <span style='color:{GREY_TEXT};font-size:0.8rem;font-weight:400;'>({latest_month})</span>", unsafe_allow_html=True)

        cols = st.columns(max(len(latest), 1))
        for col, (_, row) in zip(cols, latest.iterrows()):
            clr = GREEN if row["passed"] else RED
            col.markdown(
                f"<div style='background:{CARD_BG};border:1px solid {clr}60;"
                f"border-radius:8px;padding:18px;text-align:center;'>"
                f"<div style='font-size:0.72rem;color:{GREY_TEXT};text-transform:uppercase;"
                f"letter-spacing:0.06em;font-weight:600;'>{row['column']}</div>"
                f"<div style='font-size:2.2rem;font-weight:800;color:{clr};line-height:1.1;"
                f"margin:6px 0 2px;'>{row['dir']:.3f}</div>"
                f"<div style='font-size:0.72rem;color:{clr};font-weight:600;'>"
                f"{'PASSED' if row['passed'] else 'FAILED'} &nbsp;|&nbsp; threshold 0.80</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<br/>", unsafe_allow_html=True)

    fig_dir = px.line(
        df_fair, x="month", y="dir", color="column",
        markers=True,
        color_discrete_sequence=[GREEN, AMBER],
        labels={"dir": "Disparate Impact Ratio", "month": "Month", "column": "Attribute"},
    )
    fig_dir.add_hline(
        y=0.80, line_dash="dot", line_color=RED,
        annotation_text="FCA minimum (0.80)", annotation_font_color=RED,
    )
    fig_dir.add_hrect(y0=0, y1=0.80, fillcolor=RED, opacity=0.04, line_width=0)
    fig_dir = dark_chart(fig_dir, "Disparate Impact Ratio Over Time")
    fig_dir.update_layout(yaxis=dict(range=[0.5, 1.05]))
    st.plotly_chart(fig_dir, use_container_width=True)

    st.markdown("#### Full Audit Log")
    st.dataframe(df_fair, use_container_width=True)

    st.markdown(
        f"<div style='background:{CARD_BG};border:1px solid {BORDER};border-radius:8px;"
        f"padding:16px 20px;margin-top:8px;font-size:0.82rem;color:{GREY_TEXT};line-height:1.6;'>"
        f"<b style='color:#C5D0DE;'>How DIR is calculated:</b> For each protected attribute, "
        "DIR = ESCALATE rate of the least-advantaged group divided by the rate of the "
        "most-advantaged group. A value below 0.80 constitutes prohibited disparate impact "
        "under the FCA 4/5ths rule. Only characteristics protected under the UK Equality "
        "Act 2010 are included as hard gate columns."
        "</div>",
        unsafe_allow_html=True,
    )
