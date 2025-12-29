import streamlit as st
import pandas as pd
import plotly.express as px
import time

# -----------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------
st.set_page_config(
    page_title="Hybrid AQI Forecasting Dashboard",
    layout="wide",
    page_icon="🌏"
)

# -----------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/test_with_anomalies.csv", parse_dates=["DateTime"])
    hybrid = pd.read_csv("data/hybrid_forecast.csv")
    df["Hybrid_AQI"] = hybrid["Hybrid_AQI"].head(len(df))
    return df

df = load_data()

# -----------------------------------------------------------
# HEADER
# -----------------------------------------------------------
st.title("🌏 Hybrid AQI Forecasting — Quantum Hybrid Model")
st.markdown("#### A real-time, spatio-temporal AQI monitoring dashboard")

# -----------------------------------------------------------
# TOP ROW — Summary + Trend Chart
# -----------------------------------------------------------
top_left, top_right = st.columns([1, 2])  # 1/3 + 2/3 layout

with top_left:
    st.subheader("📊 AQI Summary")
    total_anomalies = int(df["Anomaly"].sum())
    avg_aqi = round(df["AQI"].mean(), 2)
    peak_aqi = int(df["AQI"].max())

    c1, c2, c3 = st.columns(3)
    c1.metric("🚨 Total Anomalies", total_anomalies)
    c2.metric("🌤️ Avg AQI", avg_aqi)
    c3.metric("🔥 Peak AQI", peak_aqi)

    st.markdown("---")
    st.markdown("**City Selection**")
    cities = df["City"].unique().tolist()
    selected_cities = st.multiselect(
        "Select Cities to Compare",
        cities,
        # default=cities[:3]
    )
    filtered_df = df[df["City"].isin(selected_cities)]

with top_right:
    st.subheader("📈 AQI vs Hybrid Forecast Trend")
    fig_trend = px.line(
        filtered_df,
        x="DateTime",
        y=["AQI", "Hybrid_AQI"],
        color="City",
        labels={"value": "AQI", "variable": "Type"},
        title="Actual vs Predicted AQI (Hybrid Model)"
    )
    st.plotly_chart(fig_trend, use_container_width=True)

# -----------------------------------------------------------
# BOTTOM ROW — Heatmap + Real-Time Updates
# -----------------------------------------------------------
bottom_left, bottom_right = st.columns(2)

with bottom_left:
    st.subheader("🗺️ City-wise AQI Heatmap")

latest_time = df["DateTime"].max()
latest_df = df[df["DateTime"] == latest_time]

# ✅ Filter heatmap data based on selected cities
latest_df = latest_df[latest_df["City"].isin(selected_cities)]

fig_map = px.density_mapbox(
    latest_df,
    lat="Latitude",
    lon="Longitude",
    z="AQI",
    radius=25,
    hover_name="City",
    mapbox_style="carto-positron",
    color_continuous_scale="Turbo",
    title=f"AQI Distribution — {latest_time.date()} ({', '.join(selected_cities)})",
    zoom=4.3,
)

st.plotly_chart(fig_map, use_container_width=True)

with bottom_right:
    st.subheader("⏱️ Real-time AQI Update Simulation")

    update_interval = st.slider("Update Interval (sec)", 2, 10, 5)
    placeholder = st.empty()

    for _ in range(5):  # simulate 5 updates
        live_df = df.sample(80)
        fig_live = px.scatter(
            live_df,
            x="DateTime",
            y="AQI",
            color="City",
            title="Live AQI Stream (Simulated)",
        )
        placeholder.plotly_chart(fig_live, use_container_width=True)
        time.sleep(update_interval)

    st.success("✅ Live Simulation Completed")

    st.markdown("---")
    st.subheader("⚠️ Detected Anomalies (Top 10)")
    anomalies = df[df["Anomaly"] == 1][["DateTime", "City", "AQI", "Hybrid_AQI"]]
    st.dataframe(anomalies.head(10), height=250)
