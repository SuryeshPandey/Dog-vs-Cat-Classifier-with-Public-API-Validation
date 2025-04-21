import pandas as pd

# Load the metadata CSV file
metadata_path = "F:/suryesh project/vision_pipeline/metadata.csv"
meta_df = pd.read_csv(metadata_path)

# Drop rows where the 'class_name' is not 'dog' or 'cat'
meta_df = meta_df[meta_df['class_name'].isin(['dog', 'cat'])]

# Optionally, save the cleaned DataFrame back to a CSV file
meta_df.to_csv("F:/suryesh project/vision_pipeline/metadata_cleaned.csv", index=False)

# Print the first few rows to confirm
print(meta_df.head())
