import pandas as pd
import numpy as np

def combine_datasets(true_csv_path, fake_csv_path, output_path="./combined_news.csv"):
    
    print("Reading datasets...")
    true_df = pd.read_csv(true_csv_path)
    fake_df = pd.read_csv(fake_csv_path)
    
    print("Adding labels...")
    true_df['label'] = 1
    fake_df['label'] = 0
    
    print("Combining datasets...")
    combined_df = pd.concat([true_df, fake_df], ignore_index=True)
    
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    print("\nDataset Statistics:")
    print(f"Total articles: {len(combined_df)}")
    print(f"Real articles: {len(combined_df[combined_df['label'] == 1])}")
    print(f"Fake articles: {len(combined_df[combined_df['label'] == 0])}")
    print("\nMissing values:")
    print(combined_df.isnull().sum())
    
    print(f"\nSaving combined dataset to {output_path}")
    combined_df.to_csv(output_path, index=False)
    print("Done!")
    
    return combined_df

if __name__ == "__main__":
    true_csv = "true.csv"
    fake_csv = "fake.csv"
    
    combined_df = combine_datasets(true_csv, fake_csv)
    
    print("\nFirst few rows of the combined dataset:")
    print(combined_df.head())