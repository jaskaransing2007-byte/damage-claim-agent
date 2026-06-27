"""
Damage Claim AI - Accuracy & Evaluation Metrics Suite
Compares the output predictions against ground truth to calculate
Accuracy, Precision, and Confusion matrix statistics.
"""

import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

print("📈 Initializing Damage Claim Evaluation Metrics Suite...\n")

try:
    # 1. Load your newly generated AI predictions
    predicted_df = pd.read_csv("output.csv")
    
    # 2. Load your local ground truth validation key 
    # (For a quick check, you can create a 'truth.csv' containing just user_id and actual_status)
    truth_df = pd.read_csv("dataset/claims.csv") 
    
    # Check if a known truth column exists for verification testing
    if 'claim_status' not in truth_df.columns:
        print("⚠️ 'claim_status' ground-truth labels not present in dataset/claims.csv.")
        print("To run local accuracy checks, append a column named 'claim_status' containing human-verified answers.")
        exit()

    # Align data on user_id to map them perfectly
    merged = pd.merge(truth_df[['user_id', 'claim_status']], predicted_df[['user_id', 'claim_status']], 
                      on='user_id', suffixes=('_true', '_pred'))

    y_true = merged['claim_status_true']
    y_pred = merged['claim_status_pred']

    # 3. Calculate Scores
    accuracy = accuracy_score(y_true, y_pred)
    print(f"==========================================")
    print(f"🎯 TOTAL PIPELINE ACCURACY: {accuracy * 100:.2f}%")
    print(f"==========================================\n")

    print("📋 Detailed Performance Classification Report:")
    print(classification_report(y_true, y_pred))
    
except FileNotFoundError:
    print("❌ Error: Could not locate 'output.csv'. Run python run_batch.py first to generate predictions!")
except Exception as e:
    print(f"❌ Metrics evaluation stopped: {str(e)}")