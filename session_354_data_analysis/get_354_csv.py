import pandas as pd

# Read the CSV
df = pd.read_csv("/Users/gafurmammadov/Documents/Uchicago_classes/practicum/ems_lib/ems-main/ems/app_dev/training_data.csv")

# Keep only rows where session_id == 354
filtered = df[df["session_id"] == 354]

# Save to a new CSV
filtered.to_csv("354_data.csv", index=False)