import pandas as pd
import numpy as np

# Load and clean baseline data
file_path = r"C:\Users\konat\OneDrive\Documents\Railway_info.csv"
df = pd.read_csv(file_path)

# Standardize text and clean trailing 'd' from days
text_cols = ["Train_Name", "Source_Station_Name", "Destination_Station_Name"]
for col in text_cols:
    df[col] = df[col].astype(str).str.strip().str.upper()

df["days"] = df["days"].astype(str).str.strip().str.rstrip("d").str.capitalize()
df.dropna(subset=["Train_No", "Source_Station_Name", "Destination_Station_Name"], inplace=True)

# ==========================================
# Task 2.1: Data Filtering
# ==========================================

# 1. Filter trains operating on specific days (e.g., only on Saturdays)
saturday_trains = df[df["days"] == "Saturday"].copy()
print(f"Total trains operating on Saturday: {len(saturday_trains)}")

# 2. Extract and create a new DataFrame for trains starting from a specific station
specific_station = "CST-MUMBAI"
mumbai_origin_df = df[df["Source_Station_Name"] == specific_station].copy()
print(f"Total train schedules originating from {specific_station}: {len(mumbai_origin_df)}")
print(mumbai_origin_df.head())

# ==========================================
# Task 2.2: Grouping and Aggregation
# ==========================================

# 1. Count the number of unique trains originating from each source station
origin_train_counts = (
    df.groupby("Source_Station_Name")["Train_No"]
    .nunique()
    .reset_index(name="Total_Originating_Trains")
    .sort_values(by="Total_Originating_Trains", ascending=False)
)

# 2. Find the average number of trains per day for each source station
daily_station_counts = (
    df.groupby(["Source_Station_Name", "days"])["Train_No"]
    .count()
    .reset_index(name="Daily_Train_Count")
)

avg_trains_per_day = (
    daily_station_counts.groupby("Source_Station_Name")["Daily_Train_Count"]
    .mean()
    .round(2)
    .reset_index(name="Avg_Trains_Per_Day")
)

# Combine into a unified summary table
station_aggregation_summary = pd.merge(origin_train_counts, avg_trains_per_day, on="Source_Station_Name")
print("\n--- Station Aggregation Summary (Top 10) ---")
print(station_aggregation_summary.head(10))

# ==========================================
# Task 2.3: Data Enrichment
# ==========================================

# Add a category column: 'Weekend' (Saturday, Sunday) vs 'Weekday' (Monday-Friday)
weekend_days = ["Saturday", "Sunday"]
df["Day_Type"] = np.where(df["days"].isin(weekend_days), "Weekend", "Weekday")

# Verify enrichment distribution
print("\n--- Distribution of Day_Type ---")
print(df["Day_Type"].value_counts())

print("\n--- Sample of Enriched DataFrame ---")
print(df[["Train_No", "Train_Name", "Source_Station_Name", "days", "Day_Type"]].head(10))

# Save the enriched dataset
df.to_csv("Railway_info_enriched.csv", index=False)