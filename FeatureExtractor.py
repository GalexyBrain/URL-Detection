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
    df_chunk['https_in_url'] = df_chunk['url'].apply(lambda x: 1 if 'https' in x else 0)  # scheme substring
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
        raise SystemExit(1)

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
    features_df = pd.concat(processed_chunks, ignore_index=True)
    print("Feature extraction complete.")
    
    # --- Label Encoding ---
    features_df['label'] = features_df['label'].apply(lambda x: 0 if x == 'benign' else 1)
    print("\nEncoded 'label' column into binary (0: benign, 1: malicious).")
    print(f"Class distribution (%):\n{(features_df['label'].value_counts(normalize=True) * 100).round(2)}")

    # --- Finalizing the DataFrame ---
    # Keep URL as the FIRST column, then features, then label LAST.
    feature_columns = [
        'url_length', 'digit_count', 'special_char_count', 'hostname_length',
        'path_length', 'entropy', 'num_tokens', 'query_length', 'dot_count',
        'semicolon_count', 'underscore_count', 'question_mark_count', 'hash_char_count',
        'equal_count', 'percent_char_count', 'ampersand_count', 'dash_count',
        'at_char_count', 'tilde_char_count', 'double_slash_count',
        'digit_alphabet_ratio', 'special_char_alphabet_ratio',
        'uppercase_lowercase_ratio', 'domain_url_ratio', 'ip_as_hostname',
        'exe_in_url', 'https_in_url', 'ftp_used', 'js_used', 'css_used'
    ]
    final_columns = ['url'] + feature_columns + ['label']
    
    # Ensure all columns exist before selection (this will raise if any missing — intentional)
    final_df = features_df[final_columns]

    output_filename = 'features_extracted.csv'
    final_df.to_csv(output_filename, index=False)

    total_time = time.time() - start_time
    print(f"\n--- Feature Extraction Finished in {total_time:.2f} seconds ---")
    print(f"Processed data with features saved to '{output_filename}'.")
    print("\nFirst 5 rows of the new dataset:")
    print(final_df.head())

    # --- Generating Beautiful, Comparable Plots ---
    print("\n--- Generating Plots ---")

    from scipy.stats import gaussian_kde
    import math

    # Always-White backgrounds (no surprises on export)
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 240,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.titlesize": 13,
        "figure.autolayout": True,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

    # Data splits for plotting
    benign_mask = final_df['label'] == 0
    mal_mask = final_df['label'] == 1

    # ----------------- Existing URL length comparisons -----------------
    benign_len = final_df.loc[benign_mask, 'url_length'].dropna().to_numpy()
    mal_len = final_df.loc[mal_mask, 'url_length'].dropna().to_numpy()

    def freedman_diaconis_bins(x):
        x = np.asarray(x); x = x[~np.isnan(x)]
        n = x.size
        if n < 2: return 30
        iqr = np.subtract(*np.percentile(x, [75, 25]))
        if iqr == 0:
            return min(100, max(30, int(np.sqrt(n))))
        h = 2 * iqr * (n ** (-1/3))
        if h <= 0: return min(100, max(30, int(np.sqrt(n))))
        bins = int(np.ceil((x.max() - x.min()) / h))
        return max(40, min(100, bins))  # keep things visually smooth

    def annotate_box(ax, data, label):
        med = np.median(data) if len(data) else float('nan')
        p10, p90 = np.percentile(data, [10, 90]) if len(data) else (float('nan'), float('nan'))
        ax.axvline(med, linestyle="--", linewidth=1, alpha=0.9, label=f"Median = {med:.0f}")
        txt = f"{label}\nN={len(data):,}\nP10={p10:.0f}, P90={p90:.0f}"
        ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left",
                fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    def nice_hist(ax, data, title, bins=None):
        if len(data) == 0:
            ax.set_title(title + " (no data)")
            return
        if bins is None: bins = freedman_diaconis_bins(data)
        ax.hist(data, bins=bins, density=True, alpha=0.95, edgecolor="none")
        xs = np.linspace(*np.percentile(data, [1, 99]), 400)
        try:
            kde = gaussian_kde(data)
            ax.plot(xs, kde(xs), linewidth=1.6)
        except Exception:
            pass
        ax.set_title(title)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.25, linestyle="--")

    # Shared x-range for URL length for fair compare
    if len(benign_len) and len(mal_len):
        lo = np.percentile(np.concatenate([benign_len, mal_len]), 1)
        hi = np.percentile(np.concatenate([benign_len, mal_len]), 99)
    else:
        lo, hi = 0, 1

    # 1) Side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    nice_hist(axes[0], benign_len, "Benign URLs — Length")
    annotate_box(axes[0], benign_len, "Benign")
    nice_hist(axes[1], mal_len, "Malicious URLs — Length")
    annotate_box(axes[1], mal_len, "Malicious")
    for ax in axes:
        ax.set_xlim(lo, hi)
        ax.legend(frameon=False, loc="upper right")
    fig.suptitle("URL Length Distributions by Class", y=1.02, fontsize=14, fontweight="bold")
    fig.savefig("url_length_distribution_compare.png", bbox_inches="tight")
    print("Saved 'url_length_distribution_compare.png'")

    # 2) Per-class polished exports
    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    nice_hist(ax, benign_len, "Frequency Distribution of URL Length — Benign")
    annotate_box(ax, benign_len, "Benign")
    ax.set_xlim(lo, hi)
    ax.legend(frameon=False, loc="upper right")
    plt.savefig("url_length_distribution_benign.png", bbox_inches="tight")
    print("Saved 'url_length_distribution_benign.png' (enhanced)")

    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    nice_hist(ax, mal_len, "Frequency Distribution of URL Length — Malicious")
    annotate_box(ax, mal_len, "Malicious")
    ax.set_xlim(lo, hi)
    ax.legend(frameon=False, loc="upper right")
    plt.savefig("url_length_distribution_malicious.png", bbox_inches="tight")
    print("Saved 'url_length_distribution_malicious.png' (enhanced)")

    # 3) CDF comparison (white bg, clean legend)
    plt.figure(figsize=(8, 5))
    for data, lbl in [(benign_len, "Benign"), (mal_len, "Malicious")]:
        vals = np.sort(data); y = np.linspace(0, 1, len(vals), endpoint=True)
        plt.step(vals, y, where="post", linewidth=1.6, label=lbl)
    plt.xlim(lo, hi)
    plt.xlabel("URL Length (characters)")
    plt.ylabel("Cumulative probability")
    plt.title("URL Length — CDF Comparison")
    plt.grid(True, alpha=0.25, linestyle="--")
    plt.legend(frameon=False)
    plt.savefig("url_length_cdf_compare.png", bbox_inches="tight")
    print("Saved 'url_length_cdf_compare.png'")

    # ----------------- NEW: One consolidated figure for ALL features -----------------
    print("Building consolidated feature grid...")

    # Treat every feature except 'url' and 'label' as plottable
    all_feats = [c for c in final_df.columns if c not in ('url', 'label')]
    n_feats = len(all_feats)
    n_cols = 4
    n_rows = math.ceil(n_feats / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.0, n_rows * 3.0))
    axes = np.array(axes).reshape(-1)  # flatten for easy indexing

    for i, feat in enumerate(all_feats):
        ax = axes[i]
        b = final_df.loc[benign_mask, feat].dropna().to_numpy()
        m = final_df.loc[mal_mask, feat].dropna().to_numpy()

        # Binary detection (strict 0/1 set)
        unique_vals = set(np.unique(final_df[feat].dropna().values))
        is_binary = unique_vals.issubset({0, 1})

        if is_binary:
            # plot per-class rate for '1'
            p_benign = float((b == 1).mean()) if len(b) else 0.0
            p_mal = float((m == 1).mean()) if len(m) else 0.0
            x = np.arange(2)
            ax.bar(x - 0.15, [1 - p_benign, p_benign], width=0.3, label="Benign")
            ax.bar(x + 0.15, [1 - p_mal, p_mal], width=0.3, label="Malicious")
            ax.set_xticks(x)
            ax.set_xticklabels(['0', '1'])
            ax.set_ylabel("Proportion")
            ax.set_title(feat)
            ax.grid(True, alpha=0.25, linestyle="--")
        else:
            # Overlay histograms (density) with KDE if possible
            # Robust x-limits by percentiles if data exists
            both = np.concatenate([b, m]) if len(b) and len(m) else np.array([])
            if both.size > 0:
                lo_f, hi_f = np.percentile(both, [1, 99])
            else:
                lo_f, hi_f = 0, 1

            if len(b):
                bins_b = freedman_diaconis_bins(b)
                ax.hist(b, bins=bins_b, density=True, alpha=0.55, edgecolor="none", label="Benign")
                try:
                    xs = np.linspace(lo_f, hi_f, 400)
                    kde_b = gaussian_kde(b)
                    ax.plot(xs, kde_b(xs), linewidth=1.2)
                except Exception:
                    pass

            if len(m):
                bins_m = freedman_diaconis_bins(m)
                ax.hist(m, bins=bins_m, density=True, alpha=0.55, edgecolor="none", label="Malicious")
                try:
                    xs = np.linspace(lo_f, hi_f, 400)
                    kde_m = gaussian_kde(m)
                    ax.plot(xs, kde_m(xs), linewidth=1.2)
                except Exception:
                    pass

            ax.set_xlim(lo_f, hi_f)
            ax.set_title(feat)
            ax.set_ylabel("Density")
            ax.grid(True, alpha=0.25, linestyle="--")

        if i % n_cols == 0:
            ax.set_xlabel("Value")
        ax.legend(frameon=False, fontsize=8)

    # Hide any unused axes
    for j in range(n_feats, len(axes)):
        axes[j].axis('off')

    fig.suptitle("All Feature Distributions (Benign vs Malicious)", y=0.995, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig("all_feature_distributions.png", bbox_inches="tight")
    print("Saved 'all_feature_distributions.png'")

    print("\n--- Plotting complete ---")
