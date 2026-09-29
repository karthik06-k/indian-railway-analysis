import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ==========================================
# 1. LOAD AND PREPARE DATA
# ==========================================
# Update path to your exact file location
file_path = r"C:\Users\konat\OneDrive\Documents\Railway_info.csv"
df = pd.read_csv(file_path)

# Text and Day standardization
for col in ["Train_Name", "Source_Station_Name", "Destination_Station_Name"]:
  df[col] = df[col].astype(str).str.strip().str.upper()

df["days"] = df["days"].astype(str).str.strip().str.rstrip("d").str.capitalize()
df.dropna(
    subset=["Train_No", "Source_Station_Name", "Destination_Station_Name"],
    inplace=True,
)

# Define chronological day order
day_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
df["days"] = pd.Categorical(df["days"], categories=day_order, ordered=True)

# ==========================================
# Task 3.1: Pattern Analysis
# ==========================================

# 1. Weekly Journey Distribution
weekly_distribution = (
    df["days"].value_counts().reindex(day_order).reset_index()
)
weekly_distribution.columns = ["Day_of_Week", "Total_Train_Journeys"]

# 2. Route Patterns (Source -> Destination)
df["Route"] = df["Source_Station_Name"] + " -> " + df["Destination_Station_Name"]
top_routes = df["Route"].value_counts().head(10).reset_index()
top_routes.columns = ["Route", "Frequency"]

# Visualizations: Weekly Distribution & Top Routes
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.barplot(
    data=weekly_distribution,
    x="Day_of_Week",
    y="Total_Train_Journeys",
    hue="Day_of_Week",
    palette="Blues_r",
    legend=False,
    ax=axes[0],
)
axes[0].set_title(
    "Weekly Train Journey Distribution", fontsize=12, weight="bold"
)
axes[0].set_xlabel("Day of Week")
axes[0].set_ylabel("Total Journeys Scheduled")
axes[0].tick_params(axis="x", rotation=30)

sns.barplot(
    data=top_routes,
    x="Frequency",
    y="Route",
    hue="Route",
    palette="viridis",
    legend=False,
    ax=axes[1],
)
axes[1].set_title(
    "Top 10 High-Density Train Corridors", fontsize=12, weight="bold"
)
axes[1].set_xlabel("Frequency (Journeys / Week)")
axes[1].set_ylabel("Route (Source -> Destination)")

plt.tight_layout()
plt.show()

# ==========================================
# Task 3.2: Correlation & Station-Day Matrix
# ==========================================

# Cross-tabulate top 10 busiest origin stations against operating days
top_10_sources = df["Source_Station_Name"].value_counts().head(10).index
station_day_matrix = pd.crosstab(
    df[df["Source_Station_Name"].isin(top_10_sources)]["Source_Station_Name"],
    df["days"],
)[day_order]

# Heatmap visualization of operational load
plt.figure(figsize=(12, 6))
sns.heatmap(
    station_day_matrix,
    annot=True,
    fmt="d",
    cmap="YlGnBu",
    cbar_kws={"label": "Train Departures"},
)
plt.title(
    "Operational Density: Top 10 Hubs vs. Day of the Week",
    fontsize=13,
    weight="bold",
)
plt.xlabel("Day of Week")
plt.ylabel("Source Station")
plt.tight_layout()
plt.show()