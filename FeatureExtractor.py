# feature_extraction_parallel.py

import pandas as pd
import numpy as np
from scipy.stats import entropy
import re
from urllib.parse import urlparse
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool, cpu_count
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- 1. Feature Extraction Functions ---
# These helper functions are used by the main processing function below.

def calculate_entropy(text):
    """Calculates the Shannon entropy of a string."""
    if not text:
        return 0
    prob = [float(text.count(c)) / len(text) for c in dict.fromkeys(list(text))]
    return entropy(prob, base=2)

def count_special_chars(text):
    """Counts non-alphanumeric characters, excluding common URL structural chars."""
    special_char_pattern = re.compile(r'[^a-zA-Z0-9\.\/\:\-\_\?\#\=]')
    return len(special_char_pattern.findall(text))

def is_ip_hostname(hostname):
    """Checks if the hostname is an IP address."""
    ip_pattern = re.compile(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$')
    return 1 if ip_pattern.match(hostname) else 0

def extract_features(df_chunk):
    """
    This function takes a chunk of the DataFrame and applies all feature
    extraction logic to it. This is the function that each core will run.
    """
    # Parse URL components first to avoid repetitive parsing
    parsed_urls = df_chunk['url'].apply(urlparse)
    
    # --- Lexical and Length-based Features ---
    df_chunk['url_length'] = df_chunk['url'].apply(len)
    df_chunk['hostname_length'] = parsed_urls.apply(lambda x: len(x.netloc))
    df_chunk['path_length'] = parsed_urls.apply(lambda x: len(x.path))
    df_chunk['query_length'] = parsed_urls.apply(lambda x: len(x.query))
    
    # --- Count-based Features ---
    df_chunk['digit_count'] = df_chunk['url'].apply(lambda x: len(re.findall(r'\d', x)))
    df_chunk['special_char_count'] = df_chunk['url'].apply(count_special_chars)
    df_chunk['num_tokens'] = df_chunk['url'].apply(lambda x: len(re.split(r'[\/\.\-\?\#\=]', x)))
    df_chunk['dot_count'] = df_chunk['url'].apply(lambda x: x.count('.'))
    df_chunk['semicolon_count'] = df_chunk['url'].apply(lambda x: x.count(';'))
    df_chunk['underscore_count'] = df_chunk['url'].apply(lambda x: x.count('_'))
    df_chunk['question_mark_count'] = df_chunk['url'].apply(lambda x: x.count('?'))
    df_chunk['hash_char_count'] = df_chunk['url'].apply(lambda x: x.count('#'))
    df_chunk['equal_count'] = df_chunk['url'].apply(lambda x: x.count('='))
    df_chunk['percent_char_count'] = df_chunk['url'].apply(lambda x: x.count('%'))
    df_chunk['ampersand_count'] = df_chunk['url'].apply(lambda x: x.count('&'))
    df_chunk['dash_count'] = df_chunk['url'].apply(lambda x: x.count('-'))
    df_chunk['at_char_count'] = df_chunk['url'].apply(lambda x: x.count('@'))
    df_chunk['tilde_char_count'] = df_chunk['url'].apply(lambda x: x.count('~'))
    df_chunk['double_slash_count'] = df_chunk['url'].apply(lambda x: x.count('//'))
    
    # --- Statistical and Ratio Features ---
    df_chunk['entropy'] = df_chunk['url'].apply(calculate_entropy)
    df_chunk['alphabet_count'] = df_chunk['url'].apply(lambda x: len(re.findall(r'[a-zA-Z]', x)))
    df_chunk['uppercase_count'] = df_chunk['url'].apply(lambda x: len(re.findall(r'[A-Z]', x)))
    df_chunk['lowercase_count'] = df_chunk['url'].apply(lambda x: len(re.findall(r'[a-z]', x)))
    
    # These ratios must be calculated after the counts are available
    df_chunk['digit_alphabet_ratio'] = df_chunk['digit_count'] / (df_chunk['alphabet_count'] + 1)
    df_chunk['special_char_alphabet_ratio'] = df_chunk['special_char_count'] / (df_chunk['alphabet_count'] + 1)
    df_chunk['uppercase_lowercase_ratio'] = df_chunk['uppercase_count'] / (df_chunk['lowercase_count'] + 1)
    df_chunk['domain_url_ratio'] = df_chunk['hostname_length'] / df_chunk['url_length']
    
    # --- Binary Features ---
    df_chunk['ip_as_hostname'] = parsed_urls.apply(lambda x: is_ip_hostname(x.netloc))
    df_chunk['exe_in_url'] = df_chunk['url'].apply(lambda x: 1 if '.exe' in x.lower() else 0)
    df_chunk['https_in_url'] = df_chunk['url'].apply(lambda x: 1 if 'https' in x else 0)
    df_chunk['ftp_used'] = df_chunk['url'].apply(lambda x: 1 if 'ftp://' in x.lower() else 0)
    df_chunk['js_used'] = df_chunk['url'].apply(lambda x: 1 if '.js' in x.lower() else 0)
    df_chunk['css_used'] = df_chunk['url'].apply(lambda x: 1 if '.css' in x.lower() else 0)
    
    return df_chunk

# --- Main Execution Block ---
if __name__ == '__main__':
    start_time = time.time()
    print("--- Starting Feature Extraction ---")

    # --- Load and Preprocess Data ---
    try:
        df = pd.read_csv('merged_url_dataset.csv')
        print(f"Successfully loaded merged_url_dataset.csv with {len(df)} rows.")
    except FileNotFoundError:
        print("Error: merged_url_dataset.csv not found.")
        print("Please make sure the dataset file is in the same directory.")
        exit()

    df.drop_duplicates(subset='url', inplace=True)
    df.dropna(inplace=True)
    print(f"Cleaned data, {len(df)} rows remaining for processing.")

    # --- Parallel Processing Setup ---
    # Use n-1 cores to leave one free for system processes
    num_cores = cpu_count() - 1 if cpu_count() > 1 else 1 
    print(f"\nUsing {num_cores} CPU cores for parallel processing...")

    # Split the DataFrame into chunks for each core
    df_chunks = np.array_split(df, num_cores)

    # Create a pool of worker processes
    with Pool(num_cores) as pool:
        # Map the extract_features function to each chunk and collect the results
        processed_chunks = pool.map(extract_features, df_chunks)

    # Concatenate the processed chunks back into a single DataFrame
    features_df = pd.concat(processed_chunks)
    print("Feature extraction complete.")
    
    # --- Label Encoding ---
    features_df['label'] = features_df['label'].apply(lambda x: 0 if x == 'benign' else 1)
    print("\nEncoded 'type' column into binary 'label' (0: benign, 1: malicious).")
    print(f"Class distribution:\n{features_df['label'].value_counts(normalize=True) * 100}")

    # --- Finalizing the DataFrame ---
    feature_columns = [
        'url_length', 'digit_count', 'special_char_count', 'hostname_length',
        'path_length', 'entropy', 'num_tokens', 'query_length', 'dot_count',
        'semicolon_count', 'underscore_count', 'question_mark_count', 'hash_char_count',
        'equal_count', 'percent_char_count', 'ampersand_count', 'dash_count',
        'at_char_count', 'tilde_char_count', 'double_slash_count',
        'digit_alphabet_ratio', 'special_char_alphabet_ratio',
        'uppercase_lowercase_ratio', 'domain_url_ratio', 'ip_as_hostname',
        'exe_in_url', 'https_in_url', 'ftp_used', 'js_used', 'css_used', 'label'
    ]
    
    # Ensure all columns exist before selection
    final_df = features_df[feature_columns]

    output_filename = 'features_extracted.csv'
    final_df.to_csv(output_filename, index=False)

    total_time = time.time() - start_time
    print(f"\n--- Feature Extraction Finished in {total_time:.2f} seconds ---")
    print(f"Processed data with features saved to '{output_filename}'.")
    print("\nFirst 5 rows of the new dataset:")
    print(final_df.head())

    # --- Generate and Save Plots ---
    print("\n--- Generating Plots ---")

    # Separate data into benign and malicious
    benign_urls = final_df[final_df['label'] == 0]
    malicious_urls = final_df[final_df['label'] == 1]

    # Plot 1: Frequency Distribution of URL Length for Benign URLs
    plt.figure(figsize=(10, 6))
    plt.hist(benign_urls['url_length'], bins=50, color='green', alpha=0.7)
    plt.title('Frequency Distribution of URL Length - Benign URLs')
    plt.xlabel('URL Length')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig('url_length_distribution_benign.png')
    print("Saved 'url_length_distribution_benign.png'")


    # Plot 2: Frequency Distribution of URL Length for Malicious URLs
    plt.figure(figsize=(10, 6))
    plt.hist(malicious_urls['url_length'], bins=50, color='red', alpha=0.7)
    plt.title('Frequency Distribution of URL Length - Malicious URLs')
    plt.xlabel('URL Length')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig('url_length_distribution_malicious.png')
    print("Saved 'url_length_distribution_malicious.png'")

    print("\n--- All tasks completed ---")