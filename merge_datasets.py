import pandas as pd

# Define the names of your input files and the desired output file
file1 = 'url_dataset_1.csv'
file2 = 'url_dataset_2.csv'
output_file = 'merged_url_dataset.csv'

print("--- Starting CSV Merge Script ---")

try:
    # Read the two CSV files into pandas DataFrames
    print(f"Reading '{file1}'...")
    df1 = pd.read_csv(file1)
    print(f"Reading '{file2}'...")
    df2 = pd.read_csv(file2)
    print("✅ Files read successfully.")

    # Append the second DataFrame to the first one using pd.concat
    # ignore_index=True resets the index of the new DataFrame
    print("\nMerging the two files...")
    merged_df = pd.concat([df1, df2], ignore_index=True)
    print(f"✅ Merge complete. The new dataset has {len(merged_df)} total rows.")

    # Save the merged DataFrame to a new CSV file
    # index=False prevents pandas from writing the DataFrame index as a column
    print(f"\nSaving merged data to '{output_file}'...")
    merged_df.to_csv(output_file, index=False)
    print(f"🎉 Success! Merged file saved as '{output_file}'.")

except FileNotFoundError as e:
    print(f"\n❌ ERROR: File not found.")
    print(f"Please make sure the file '{e.filename}' is in the same directory as the script.")

except Exception as e:
    print(f"\n❌ An unexpected error occurred: {e}")