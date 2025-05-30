import os
import torch
import json
import pandas as pd
from transformers import DebertaTokenizer, DebertaForSequenceClassification

# Configuration
IPC_dir = ""  # The directory where the trained model is saved
trained_model_dir = f"{IPC_dir}/model" 
input_json = "File_to_test.json"  # Path to the input JSON file for evaluation
column ="Watermarked_output" # Column name to evaluate
output_csv = "Detection_LLama.csv"  # Path to save the predictions CSV


batch_size = 32  # Process data in smaller batches to avoid memory issues
include_input = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained model and tokenizer
print("Loading trained model and tokenizer...")
tokenizer = DebertaTokenizer.from_pretrained(trained_model_dir)
model = DebertaForSequenceClassification.from_pretrained(trained_model_dir)
model.to(device)
model.eval()

# Function to encode the input data
def encode_data(data, tokenizer, max_length=400):
    encodings = tokenizer(data, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
    return encodings

# Load the input data
print("Loading input data...")
with open(input_json, 'r') as f:
    data = json.load(f)

# Extract texts for evaluation
texts = [item["input"] + item[column] if include_input else item[column] for item in data[000:1000]]

# Process data in batches
print("Running inference in batches...")
predictions = []
for start_idx in range(0, len(texts), batch_size):
    end_idx = min(start_idx + batch_size, len(texts))
    batch_texts = texts[start_idx:end_idx]

    # Encode the batch
    encodings = encode_data(batch_texts, tokenizer)
    encodings = {key: val.to(device) for key, val in encodings.items()}

    # Perform inference
    with torch.no_grad():
        outputs = model(**encodings)
        logits = outputs.logits
        batch_predictions = torch.argmax(logits, dim=-1).cpu().numpy()
        predictions.extend(batch_predictions)

# Save predictions in a CSV file (index and predicted_label only)
print("Saving predictions to CSV...")
output_data = [{"index": idx, "predicted_label": int(pred_label)} for idx, pred_label in enumerate(predictions)]
output_df = pd.DataFrame(output_data)
output_df.to_csv(output_csv, index=False)
print(f"Predictions saved to {output_csv}.")