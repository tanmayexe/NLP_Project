import pandas as pd
import os

print("Starting data preparation...")

# --- Check for required files ---
if not os.path.exists('True.csv') or not os.path.exists('Fake.csv'):
    print("\nERROR: Make sure both 'True.csv' and 'Fake.csv' are in this directory.")
    print("You can download them from: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset")
else:
    try:
        # --- Load the datasets ---
        print("Reading True.csv and Fake.csv...")
        df_true = pd.read_csv(r"C:\Users\tanma\Documents\project\True.csv")
        df_fake = pd.read_csv(r"C:\Users\tanma\Documents\project\Fake.csv")

        # --- Create the 'label' column ---
        # This is the most important step
        df_true['label'] = 'REAL'
        df_fake['label'] = 'FAKE'

        # --- Combine the dataframes ---
        print("Combining the datasets...")
        df_combined = pd.concat([df_true, df_fake], ignore_index=True)

        # --- Shuffle the dataset to mix REAL and FAKE news ---
        print("Shuffling data...")
        df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

        # --- Save the new, combined CSV ---
        df_combined.to_csv('news.csv', index=False)
        print("\nSUCCESS: 'news.csv' has been created successfully!")
        print("You can now run the Streamlit app.")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
