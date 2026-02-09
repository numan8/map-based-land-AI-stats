import re
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report


# =========================
# Config
# =========================
st.set_page_config(page_title="Cash Sale Velocity (AI Stats)", layout="wide")

DATA_PATH = "Cash Sales - AI Stats.xlsx"   # keep this file in repo root
SHEET_NAME = 0  # first sheet


# =========================
# Helpers
# =========================
def _clean_money(x):
    """Convert money-ish values to float safely."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip()
    s = s.replace(",", "")
    s = re.sub(r"[^0-9.\-]", "", s)
    if s == "" or s == "." or s == "-":
        return np.nan
    try:
        return float(s)
    except:
        return np.nan


def extract_latlon(text):
    """
    Extract lat/lon from strings like:
    'Center of Lot, GPS Coordinates: 34.707235, -118.147...'
    Returns (lat, lon) or (nan, nan)
    """
    if pd.isna(text):
        return (np.nan, np.nan)
    s = str(text)

    # common pattern: "GPS Coordinates: <lat>, <lon>"
    m = re.search(r"GPS\s*Coordinates?\s*:\s*([-+]?\d{1,2}\.\d+)\s*,\s*([-+]?\d{1,3}\.\d+)", s, re.IGNORECASE)
    if m:
        return (float(m.group(1)), float(m.group(2)))

    # fallback: any "<lat>, <lon>" pair in text
    m2 = re.search(r"([-+]?\d{1,2}\.\d+)\s*,\s*([-+]?\d{1,3}\.\d+)", s)
    if m2:
        lat = float(m2.group(1))
        lon = float(m2.group(2))
        # sanity check
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            return (lat, lon)

    return (np.nan, np.nan)


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=SHEET_NAME, engine="openpyxl")

    # ---- normalize key columns (based on your sample headers) ----
    # Dates
    for c in ["PURCHASE DATE", "SALE DATE - start", "Total New Money Sale Date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    # Numeric fields
    money_cols = [
        "Purchase Price - amount",
        "Title Escrow Cost - amount",
        "Taxes We Paid - amount",
        "Total Purchase Price",
        "Cash Sales Price - amount",
        "Market Price - amount",
    ]
    for c in money_cols:
        if c in df.columns:
            df[c] = df[c].apply(_clean_money)

    # Acres
    if "Acres" in df.columns:
        df["Acres"] = df["Acres"].apply(_clean_money)

    # Basic derived metrics
    # Prefer SALE DATE - start as the sale date
    sale_date = df["SALE DATE - start"] if "SALE DATE - start" in df.columns else pd.NaT
    purchase_date = df["PURCHASE DATE"] if "PURCHASE DATE" in df.columns else pd.NaT

    df["sale_date"] = sale_date
    df["purchase_date"] = purchase_date

    df["days_to_sell"] = (df["sale_date"] - df["purchase_date"]).dt.days

    # Prices
    df["cash_sale_price"] = df["Cash Sales Price - amount"] if "Cash Sales Price - amount" in df.columns else np.nan
    df["total_purchase_price"] = df["Total Purchase Price"] if "Total Purchase Price" in df.columns else np.nan

    df["markup_multiple"] = df["cash_sale_price"] / df["total_purchase_price"]
    df["gross_profit"] = df["cash_sale_price"] - df["total_purchase_price"]

    # Commissions (your rule)
    # 4% of profit + 4% of profit + 10% of sale
    df["acq_agent_commission"] = 0.04 * df["gross_profit"]
    df["sales_agent_commission"] = 0.04 * df["gross_profit"]
    df["affiliate_commission"] = 0.10 * df["cash_sale_price"]

    df["net_profit"] = df["gross_profit"] - df["acq_agent_commission"] - df["sales_agent_commission"] - df["affiliate_commission"]
    df["net_margin_on_purchase"] = df["net_profit"] / df["total_purchase_price"]

    # Targets for velocity
    df["sell_30d"] = (df["days_to_sell"] <= 30).astype("float")
    df["sell_60d"] = (df["days_to_sell"] <= 60).astype("float")

    # Location
    if "County, State" in df.columns:
        df["county_state"] = df["County, State"].astype(str)
    else:
        df["county_state"] = ""

    if "Property Location or City" in df.columns:
        df["city"] = df["Property Location or City"].astype(str)
    else:
        df["city"] = ""

    # Extract lat/lon
    if "Directions to Property Below" in df.columns:
        latlon = df["Directions to Property Below"].apply(extract_latlon)
        df["lat"] = latlon.apply(lambda x: x[0])
        df["lon"] = latlon.apply(lambda x: x[1])
    else:
        df["lat"] = np.nan
        df["lon"] = np.nan

    # Keep only SOLD - CASH / CASH SALE rows if present
    if "Status" in df.columns:
        df["Status"] = df["Status"].astype(str)
    if "Sale Type (Push to Closing)" in df.columns:
        df["Sale Type (Push to Closing)"] = df["Sale Type (Push to Closing)"].astype(str)

    return df


def bin_success_curve(df, x_col, y_col, bins=12):
    d = df[[x_col, y_col]].dropna()
    if len(d) < 20:
        return None
    d = d.sort_values(x_col)
    d["bin"] = pd.qcut(d[x_col], q=bins, duplicates="drop")
    g = d.groupby("bin", observed=True).agg(
        x_mid=(x_col, "median"),
        rate=(y_col, "mean"),
        n=(y_col, "size"),
    ).reset_index(drop=True)
    return g


def train_prob_model(df, target_col):
    # minimal features that exist in your sample
    candidate_features = [
        "markup_multiple",
        "Acres",
        "total_purchase_price",
        "cash_sale_price",
        "Market Price - amount",
        "county_state",
        "city",
        "Zoning",
        "Water",
        "Road",
        "Power",
    ]
    features = [c for c in candidate_features if c in df.columns]

    d = df[features + [target_col]].dropna(subset=[target_col]).copy()
    d = d.dropna(subset=["markup_multiple", "total_purchase_price", "cash_sale_price"], how="any")

    if len(d) < 80:
        return None

    X = d[features]
    y = d[target_col].astype(int)

    num_cols = [c for c in features if pd.api.types.is_numeric_dtype(d[c])]
    cat_cols = [c for c in features if c not in num_cols]

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler())
            ]), num_cols),
            ("cat", Pipeline(steps=[
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols),
        ],
        remainder="drop"
    )

    model = LogisticRegression(max_iter=2000, class_weight="balanced")

    pipe = Pipeline(steps=[("pre", pre), ("clf", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y if y.nunique() > 1 else None
    )

    pipe.fit(X_train, y_train)
    p = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, p) if y_test.nunique() > 1 else np.nan

    yhat = (p >= 0.5).astype(int)
    cm = confusion_matrix(y_test, yhat)
    report = classification_report(y_test, yhat, output_dict=False)

    return {
        "pipe": pipe,
        "features": features,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "auc": auc,
        "cm": cm,
        "report": report,
        "n": len(d)
    }


# =========================
# Load
# =========================
df = load_data(DATA_PATH)

st.title("Cash Sale Velocity Dashboard (AI Stats)")
st.caption("Goal: maximize probability of selling in ≤30 or ≤60 days while keeping margins healthy (velocity of money / compounding cycles).")

# Basic sanity summary
colA, colB, colC, colD = st.columns(4)
with colA:
    st.metric("Rows", f"{len(df):,}")
with colB:
    st.metric("Median Days to Sell", f"{int(np.nanmedian(df['days_to_sell'])) if df['days_to_sell'].notna().any() else '—'}")
with colC:
    st.metric("Median Markup Multiple", f"{np.nanmedian(df['markup_multiple']):.2f}" if df["markup_multiple"].notna().any() else "—")
with colD:
    st.metric("Sell ≤30d Base Rate", f"{np.nanmean(df['sell_30d']):.2%}" if df["sell_30d"].notna().any() else "—")


# =========================
# Sidebar filters
# =========================
st.sidebar.header("Filters")

# Optional status filters
if "Status" in df.columns:
    status_vals = sorted(df["Status"].dropna().unique().tolist())
    default_status = [s for s in status_vals if "SOLD" in s.upper()] or status_vals
    status_sel = st.sidebar.multiselect("Status", status_vals, default=default_status)
else:
    status_sel = None

if "Sale Type (Push to Closing)" in df.columns:
    sale_type_vals = sorted(df["Sale Type (Push to Closing)"].dropna().unique().tolist())
    default_sale_type = [s for s in sale_type_vals if "CASH" in s.upper()] or sale_type_vals
    sale_type_sel = st.sidebar.multiselect("Sale Type", sale_type_vals, default=default_sale_type)
else:
    sale_type_sel = None

# numeric ranges
def safe_range(series, fallback=(0.0, 1.0)):
    s = series.dropna()
    if len(s) == 0:
        return fallback
    return (float(np.nanmin(s)), float(np.nanmax(s)))

ac_min, ac_max = safe_range(df.get("Acres", pd.Series(dtype=float)), (0.0, 10.0))
pp_min, pp_max = safe_range(df.get("total_purchase_price", pd.Series(dtype=float)), (0.0, 100000.0))
mk_min, mk_max = safe_range(df.get("markup_multiple", pd.Series(dtype=float)), (0.5, 5.0))
d_min, d_max = safe_range(df.get("days_to_sell", pd.Series(dtype=float)), (0.0, 365.0))

acres_rng = st.sidebar.slider("Acres", ac_min, ac_max, (ac_min, ac_max))
pp_rng = st.sidebar.slider("Total Purchase Price", pp_min, pp_max, (pp_min, pp_max))
mk_rng = st.sidebar.slider("Markup Multiple (sale / total purchase)", mk_min, mk_max, (mk_min, mk_max))
days_rng = st.sidebar.slider("Days to Sell", int(d_min), int(d_max), (int(d_min), int(d_max)))

county_vals = sorted(df["county_state"].dropna().unique().tolist()) if "county_state" in df.columns else []
county_sel = st.sidebar.multiselect("County, State", county_vals, default=[])

weight_mode = st.sidebar.selectbox(
    "Map Heat Weight",
    [
        "More weight = faster sale (1/days)",
        "More weight = higher markup multiple",
        "More weight = higher net profit",
        "More weight = higher cash sale price",
    ],
    index=0
)

# Apply filters
dff = df.copy()
if status_sel is not None:
    dff = dff[dff["Status"].isin(status_sel)]
if sale_type_sel is not None:
    dff = dff[dff["Sale Type (Push to Closing)"].isin(sale_type_sel)]

dff = dff[
    (dff.get("Acres", np.nan) >= acres_rng[0]) & (dff.get("Acres", np.nan) <= acres_rng[1]) &
    (dff["total_purchase_price"] >= pp_rng[0]) & (dff["total_purchase_price"] <= pp_rng[1]) &
    (dff["markup_multiple"] >= mk_rng[0]) & (dff["markup_multiple"] <= mk_rng[1]) &
    (dff["days_to_sell"] >= days_rng[0]) & (dff["days_to_sell"] <= days_rng[1])
]

if county_sel:
    dff = dff[dff["county_state"].isin(county_sel)]

# Keep mappable points
map_df = dff.dropna(subset=["lat", "lon"]).copy()


# =========================
# Tabs
# =========================
tab1, tab2, tab3 = st.tabs(["🗺️ Map (Heat)", "🎯 Sweet Spot (30/60 days)", "📋 Data Explorer"])


# =========================
# TAB 1: Map
# =========================
with tab1:
    st.subheader("Map: where fast cash sales happen (filter-driven heat)")
    st.write("Heat responds to your filters. Use it to spot counties/cities where the same pricing strategy sells faster.")

    if len(map_df) == 0:
        st.warning("No rows have usable GPS coordinates after filters. Make sure 'Directions to Property Below' contains GPS coordinates.")
    else:
        # weight selection
        if weight_mode.startswith("More weight = faster"):
            w = 1.0 / (map_df["days_to_sell"].clip(lower=1))
        elif "markup" in weight_mode:
            w = map_df["markup_multiple"]
        elif "net profit" in weight_mode:
            w = map_df["net_profit"].fillna(0)
        else:
            w = map_df["cash_sale_price"]

        map_df["weight"] = w.replace([np.inf, -np.inf], np.nan).fillna(0)

        # center map
        center_lat = float(map_df["lat"].median())
        center_lon = float(map_df["lon"].median())

        heat_layer = pdk.Layer(
            "HeatmapLayer",
            data=map_df,
            get_position=["lon", "lat"],
            get_weight="weight",
            radius_pixels=55,
            intensity=1.2,
            threshold=0.02,
        )

        # points layer (hover)
        pts = map_df.copy()
        pts["tooltip_days"] = pts["days_to_sell"].astype("Int64").astype(str)
        pts["tooltip_mk"] = pts["markup_multiple"].round(2).astype(str)

        point_layer = pdk.Layer(
            "ScatterplotLayer",
            data=pts,
            get_position=["lon", "lat"],
            get_radius=120,
            pickable=True,
            opacity=0.65,
        )

        tooltip = {
            "html": """
            <b>{Property Location or City}</b><br/>
            {County, State}<br/>
            Days: <b>{tooltip_days}</b><br/>
            Markup: <b>{tooltip_mk}x</b><br/>
            Purchase: ${Total Purchase Price}<br/>
            Sale: ${Cash Sales Price - amount}
            """,
            "style": {"backgroundColor": "white", "color": "black"}
        }

        view = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=7, pitch=0)

        st.pydeck_chart(
            pdk.Deck(
                map_style="mapbox://styles/mapbox/light-v9",
                initial_view_state=view,
                layers=[heat_layer, point_layer],
                tooltip=tooltip,
            )
        )

        st.caption(f"Mappable rows: {len(map_df):,} (of {len(dff):,} after filters)")

        # quick map-side stats
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sell ≤30d rate", f"{map_df['sell_30d'].mean():.2%}")
        c2.metric("Sell ≤60d rate", f"{map_df['sell_60d'].mean():.2%}")
        c3.metric("Median days", f"{int(map_df['days_to_sell'].median())}")
        c4.metric("Median markup", f"{map_df['markup_multiple'].median():.2f}x")


# =========================
# TAB 2: Sweet Spot
# =========================
with tab2:
    st.subheader("Sweet spot: markup multiple vs probability of selling fast")

    # empirical curves (binning)
    curve30 = bin_success_curve(dff, "markup_multiple", "sell_30d", bins=12)
    curve60 = bin_success_curve(dff, "markup_multiple", "sell_60d", bins=12)

    if curve30 is None or curve60 is None:
        st.warning("Not enough clean rows to draw stable curves. Need more rows with days_to_sell + prices.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=curve30["x_mid"], y=curve30["rate"],
            mode="lines+markers", name="P(sell ≤ 30 days)",
            hovertemplate="Markup=%{x:.2f}x<br/>Rate=%{y:.2%}<br/>n=%{text}",
            text=curve30["n"]
        ))
        fig.add_trace(go.Scatter(
            x=curve60["x_mid"], y=curve60["rate"],
            mode="lines+markers", name="P(sell ≤ 60 days)",
            hovertemplate="Markup=%{x:.2f}x<br/>Rate=%{y:.2%}<br/>n=%{text}",
            text=curve60["n"]
        ))
        fig.update_layout(
            xaxis_title="Markup Multiple (cash_sale_price / total_purchase_price)",
            yaxis_title="Observed Probability",
            yaxis_tickformat=".0%",
            height=420,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Sweet spot suggestion (simple rule): highest markup where rate stays above a threshold
        target_rate_30 = st.slider("Target success rate for ≤30d", 0.10, 0.95, 0.60, 0.05)
        target_rate_60 = st.slider("Target success rate for ≤60d", 0.10, 0.99, 0.80, 0.05)

        best30 = curve30[curve30["rate"] >= target_rate_30].sort_values("x_mid")
        best60 = curve60[curve60["rate"] >= target_rate_60].sort_values("x_mid")

        col1, col2 = st.columns(2)
        with col1:
            if len(best30) == 0:
                st.error("No markup band hits your ≤30d target rate (with current filters). Try lowering target or widening filters.")
            else:
                sweet_30 = float(best30["x_mid"].max())
                st.success(f"Sweet spot (≤30d @ ≥{target_rate_30:.0%}): **up to ~{sweet_30:.2f}x markup**")
                st.caption("Interpretation: above this, your observed ≤30d sell rate drops below target (given current filters).")

        with col2:
            if len(best60) == 0:
                st.error("No markup band hits your ≤60d target rate (with current filters). Try lowering target or widening filters.")
            else:
                sweet_60 = float(best60["x_mid"].max())
                st.success(f"Sweet spot (≤60d @ ≥{target_rate_60:.0%}): **up to ~{sweet_60:.2f}x markup**")
                st.caption("Interpretation: above this, your observed ≤60d sell rate drops below target (given current filters).")

    st.divider()
    st.subheader("Model-based probability (quick ML)")

    # Train two models on the filtered dataset (dff)
    with st.spinner("Training models for ≤30d and ≤60d..."):
        m30 = train_prob_model(dff, "sell_30d")
        m60 = train_prob_model(dff, "sell_60d")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Model: Sell ≤ 30 Days")
        if m30 is None:
            st.info("Not enough clean rows to train a stable model (need ~80+).")
        else:
            st.write(f"Rows used: **{m30['n']:,}** | AUC: **{m30['auc']:.3f}**")
            st.code(m30["report"])
            st.write("Confusion matrix @ 0.5 threshold:")
            st.write(pd.DataFrame(m30["cm"], index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]))

    with c2:
        st.markdown("#### Model: Sell ≤ 60 Days")
        if m60 is None:
            st.info("Not enough clean rows to train a stable model (need ~80+).")
        else:
            st.write(f"Rows used: **{m60['n']:,}** | AUC: **{m60['auc']:.3f}**")
            st.code(m60["report"])
            st.write("Confusion matrix @ 0.5 threshold:")
            st.write(pd.DataFrame(m60["cm"], index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]))

    st.divider()
    st.subheader("What to do with this (velocity of money)")

    st.markdown(
        """
- Treat **markup multiple** as your main “pricing lever”.
- Use the **sweet spot bands** above as your operational pricing policy:
  - **Aggressive mode (≤30d)**: cap markup at the ≤30d sweet spot
  - **Standard mode (≤60d)**: you can stretch up to the ≤60d sweet spot
- If you want “fly off the shelf”, pick a target **base-rate** (e.g., 70% ≤30d), then **cap markup at the highest level that still hits it**.
- Use the map to find **where you can stretch markup without losing speed** (county/city effects are real).
        """
    )


# =========================
# TAB 3: Data Explorer
# =========================
with tab3:
    st.subheader("Explore rows used in analysis")

    cols_show = [
        "county_state", "city", "Acres",
        "total_purchase_price", "cash_sale_price",
        "days_to_sell", "markup_multiple",
        "gross_profit", "net_profit", "net_margin_on_purchase",
        "purchase_date", "sale_date"
    ]
    cols_show = [c for c in cols_show if c in dff.columns]

    st.dataframe(
        dff[cols_show].sort_values("days_to_sell", ascending=True),
        use_container_width=True,
        height=520
    )

    # quick scatter: markup vs days
    st.markdown("#### Markup vs Days to Sell")
    sc = dff.dropna(subset=["markup_multiple", "days_to_sell", "total_purchase_price", "cash_sale_price"]).copy()
    if len(sc) > 0:
        fig2 = px.scatter(
            sc,
            x="markup_multiple",
            y="days_to_sell",
            hover_data=["county_state", "city", "total_purchase_price", "cash_sale_price", "net_profit"],
            trendline="lowess"
        )
        fig2.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Not enough clean rows for this plot.")
