import os
import sqlite3
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

conn = sqlite3.connect("claims.db")
query = """
    SELECT user_claim, claim_object, severity, risk_flags, claim_status 
    FROM verified_claims
"""
df = pd.read_sql_query(query, conn)
conn.close()

if len(df) < 10:
    exit()

df['label'] = df['claim_status'].apply(lambda x: 1 if x == 'supported' else 0)

def combine_features(row):
    obj = str(row['claim_object']).strip()
    sev = str(row['severity']).strip()
    risk = str(row['risk_flags']).strip()
    claim_text = str(row['user_claim']).strip()
    return f"OBJECT: {obj} | SEVERITY: {sev} | RISK: {risk} | DIALOGUE: {claim_text}"

df['input_features'] = df.apply(combine_features, axis=1)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_dataset = Dataset.from_pandas(train_df[['input_features', 'label']])
test_dataset = Dataset.from_pandas(test_df[['input_features', 'label']])

model_name = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

def tokenize_function(examples):
    return tokenizer(examples["input_features"], padding="max_length", max_length=256, truncation=True)

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 4 if device == "cpu" else 8

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=3e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
)

trainer.train()
output_model_path = "./saved_validation_model"
model.save_pretrained(output_model_path)
tokenizer.save_pretrained(output_model_path)