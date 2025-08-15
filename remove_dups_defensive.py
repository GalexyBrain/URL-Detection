import pandas as pd

# File path
csv_path = "features_adversarial_defense_dataset.csv"

# Read the CSV
df = pd.read_csv(csv_path)

print(f"Original entries: {len(df)}")

# Remove duplicates
df = df.drop_duplicates()
print(f"After removing duplicates: {len(df)}")

# Overwrite the same file
df.to_csv(csv_path, index=False)
print(f"File overwritten: {csv_path}")

# Show stats for 'orig_label'
if 'orig_label' in df.columns:
    print("\nEntries per class in 'orig_label':")
    print(df['orig_label'].value_counts())
else:
    print("\nColumn 'orig_label' not found in the dataset!")
