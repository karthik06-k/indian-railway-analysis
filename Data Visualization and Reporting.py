import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# 1. LOAD AND PREPARE DATA
# ==============================================================================
file_path = r"C:\Users\konat\OneDrive\Documents\Railway_info.csv"
df = pd.read_csv(file_path)

# Text and Day Standardization
text_cols = ["Train_Name", "Source_Station_Name", "Destination_Station_Name"]
for col in text_cols:
    df[col] = df[col].astype(str).str.strip().str.upper()

df["days"] = df["days"].astype(str).str.strip().str.rstrip("d").str.capitalize()
df.dropna(subset=["Train_No", "Source_Station_Name", "Destination_Station_Name"], inplace=True)

# Define chronological day ordering
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
df["days"] = pd.Categorical(df["days"], categories=day_order, ordered=True)

# ==============================================================================
# 2. CREATE MULTI-PANEL VISUALIZATION SUITE (MATPLOTLIB & SEABORN)
# ==============================================================================
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# Chart 1: Horizontal Bar Chart - Top 10 Busiest Originating Stations
top_sources = (
    df.groupby("Source_Station_Name")["Train_No"]
    .nunique()
    .sort_values(ascending=False)
    .head(10)
    .reset_index(name="Total_Trains")
)

sns.barplot(
    data=top_sources,
    x="Total_Trains",
    y="Source_Station_Name",
    hue="Source_Station_Name",
    palette="Blues_r",
    legend=False,
    ax=axes[0, 0],
)
axes[0, 0].set_title("Top 10 Originating Stations by Train Count", fontsize=13, weight="bold")
axes[0, 0].set_xlabel("Number of Unique Trains")
axes[0, 0].set_ylabel("Source Station")

for p in axes[0, 0].patches:
    width = p.get_width()
    axes[0, 0].annotate(
        f"{int(width)}",
        (width, p.get_y() + p.get_height() / 2.0),
        ha="left",
        va="center",
        xytext=(5, 0),
        textcoords="offset points",
        fontsize=9,
        weight="bold",
    )

# Chart 2: Line Chart - Day-Wise Journey Volume Trend
daily_volume = df["days"].value_counts().reindex(day_order).reset_index()
daily_volume.columns = ["Day", "Volume"]

sns.lineplot(
    data=daily_volume,
    x="Day",
    y="Volume",
    marker="o",
    markersize=9,
    color="#1f77b4",
    linewidth=2.5,
    ax=axes[0, 1],
)
axes[0, 1].set_title("Weekly Train Operation Volume Trend", fontsize=13, weight="bold")
axes[0, 1].set_xlabel("Day of the Week")
axes[0, 1].set_ylabel("Total Scheduled Departures")
axes[0, 1].tick_params(axis="x", rotation=30)
axes[0, 1].set_ylim(bottom=0, top=daily_volume["Volume"].max() * 1.15)

# Chart 3: Horizontal Bar Chart - Top 8 High-Traffic Corridors
df["Route"] = df["Source_Station_Name"] + " ➔ " + df["Destination_Station_Name"]
top_routes = df["Route"].value_counts().head(8).reset_index()
top_routes.columns = ["Route", "Departures"]

sns.barplot(
    data=top_routes,
    x="Departures",
    y="Route",
    hue="Route",
    palette="viridis",
    legend=False,
    ax=axes[1, 0],
)
axes[1, 0].set_title("Top 8 High-Density Travel Corridors", fontsize=13, weight="bold")
axes[1, 0].set_xlabel("Weekly Operating Frequency")
axes[1, 0].set_ylabel("Route")

# Chart 4: Heatmap - Operational Load of Top 5 Hubs vs. Day of Week
top_5_stations = top_sources["Source_Station_Name"].head(5)
matrix_data = pd.crosstab(
    df[df["Source_Station_Name"].isin(top_5_stations)]["Source_Station_Name"],
    df["days"],
)[day_order]

sns.heatmap(
    matrix_data,
    annot=True,
    fmt="d",
    cmap="YlGnBu",
    cbar=True,
    cbar_kws={"label": "Daily Departures"},
    ax=axes[1, 1],
    linewidths=0.5,
)
axes[1, 1].set_title("Operational Load: Top 5 Hubs vs. Day of Week", fontsize=13, weight="bold")
axes[1, 1].set_xlabel("Day of the Week")
axes[1, 1].set_ylabel("Source Station")

plt.suptitle("Indian Railway Operational Analytics Dashboard", fontsize=16, weight="bold", y=0.98)
plt.tight_layout()
plt.savefig("railway_analysis_dashboard.png", dpi=300)
plt.show()