import os
# Ensure the HF_HOME environment variable points to your desired cache location
os.environ["HF_TOKEN"] = "your_huggingface_token"
cache_dir = 'YOUR_CACHE_DIR'  # Replace with your desired cache directory
os.environ['HF_HOME'] = cache_dir

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import DebertaTokenizer, DebertaForSequenceClassification, TrainingArguments, Trainer, TrainerCallback, AlbertTokenizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import json
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# All Parameters
lrate = 1e-5
epoches_num = 10
batch_size = 32
base_model = "microsoft/deberta-base-mnli"
short_name = "Deberta"
# Clear cache
torch.cuda.empty_cache()

# Define the data loading function
def load_data(train_file, test_file):
    train_texts, train_labels, test_texts, test_labels = [], [], [], []

    # Load and process the training file
    with open(train_file, 'r') as f:
        data = json.load(f)
        for record in data:
            train_texts.append(record["Watermarked_output"])
            train_labels.append(1)  # 1 for Watermarked
            train_texts.append(record["Original_output"])
            train_labels.append(0)  # 0 for No_Watermark

    # Load and process the testing file
    with open(test_file, 'r') as f:
        data = json.load(f)
        for record in data[:1000]:
            test_texts.append(record["Watermarked_output"])
            test_labels.append(1)  # 1 for Watermarked
            test_texts.append(record["Original_output"])
            test_labels.append(0)  # 0 for No_Watermark

    return train_texts, train_labels, test_texts, test_labels

# Define file paths for the dataset
test_file = "/network/rit/lab/Lai_ReSecureAI/kiel/New_WM/Uniform_ST/Mistral/Test_Mistral_top_3_ST_threshold_0.8_Uniform_0_1000_1000.json"
train_file = "/network/rit/lab/Lai_ReSecureAI/kiel/New_WM/Uniform_ST/Mistral/TRAIN_Mistral_top_3_ST_threshold_0.8_Uniform_0_10000_10000.json"
# Extract file name and remove the .json extension
file_name = os.path.basename(train_file).replace(".json", "")
# Extract the desired portion of the file name
desired_portion = file_name.split("_", 1)[-1]  # This removes everything before and including the first underscore
# Construct the output directory name
output_dir = f'IPChecker_{desired_portion}_{short_name}'
print(output_dir)  # Output: IPChecker_for_LLaMA_top_2_threshold_0.85_10000
os.makedirs(output_dir, exist_ok=True)

# Load and prepare the data
train_texts, train_labels, test_texts, test_labels = load_data(train_file, test_file)

unique, counts = np.unique(train_labels, return_counts=True)
print("Train labels distribution:", dict(zip(unique, counts)))
unique, counts = np.unique(test_labels, return_counts=True)
print("Test labels distribution:", dict(zip(unique, counts)))

tokenizer = DebertaTokenizer.from_pretrained(base_model)
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=400, return_tensors="pt")
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=400, return_tensors="pt")

class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

train_dataset = CustomDataset(train_encodings, train_labels)
test_dataset = CustomDataset(test_encodings, test_labels)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

output_dim = 2  # Binary classification (0 or 1)
print("Output dimension:", output_dim)

model = DebertaForSequenceClassification.from_pretrained(base_model, num_labels=output_dim, ignore_mismatched_sizes=True).to(device)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average='weighted')
    accuracy = accuracy_score(labels, preds)
    return {'Accuracy': accuracy, 'F1': f1}

# Store loss for visualization
class LoggingCallback(TrainerCallback):                
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            output_log_file = os.path.join(args.output_dir, "train_results.json")
            with open(output_log_file, "a") as writer:
                writer.write(json.dumps(logs) + "\n")

loss_logger = LoggingCallback()

training_args = TrainingArguments(
    output_dir=output_dir,
    gradient_accumulation_steps=2,
    do_train=True,
    do_eval=True,
    learning_rate=lrate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epoches_num,
    weight_decay=0.01,
    evaluation_strategy="steps",
    logging_strategy="steps",
    logging_steps=250,
    save_steps=-1
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    callbacks=[loss_logger]
)

trainer.train()

model_path = os.path.join(output_dir, 'model')
trainer.save_model(model_path)
tokenizer.save_pretrained(model_path)

# Evaluate using the trainer
print("Starting evaluation...")
eval_results = trainer.predict(test_dataset)
predictions = eval_results.predictions
true_labels = eval_results.label_ids
eval_metrics = compute_metrics(eval_results)
print("Evaluation completed.")

# Save evaluation results
metrics_df = pd.DataFrame(eval_metrics.items(), columns=['Metric', 'Value'])
metrics_csv_path = os.path.join(output_dir, 'test_metrics.csv')
metrics_df.to_csv(metrics_csv_path, index=False)
print(f'Saved metrics to {metrics_csv_path}.')

# Generate confusion matrix and classification report using decoded labels
conf_matrix = confusion_matrix(true_labels, predictions.argmax(-1))
report = classification_report(true_labels, predictions.argmax(-1))

print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(report)

# Save classification report
report_txt_path = os.path.join(output_dir, 'classification_report.txt')
with open(report_txt_path, 'w') as f:
    f.write(report)
print(f'Saved classification report to {report_txt_path}.')

# Save confusion matrix
conf_matrix_txt_path = os.path.join(output_dir, 'confusion_matrix.txt')
with open(conf_matrix_txt_path, 'w') as f:
    for row in conf_matrix:
        f.write(' '.join(map(str, row)) + '\n')
print(f'Saved confusion matrix to {conf_matrix_txt_path}.')

# Plot the train and test loss
train_results = os.path.join(output_dir, 'train_results.json')
# Initialize lists to store the metrics
epochs = []
train_losses = []
eval_losses = []

# Read the JSON file and parse each line
with open(train_results, "r") as f:
    for line in f:
        data = json.loads(line)
        if 'epoch' in data:
            epoch = data['epoch']
            if 'loss' in data:
                train_losses.append(data['loss'])
                if epoch not in epochs:
                    epochs.append(epoch)  # Append epoch only when train loss is present
            if 'eval_loss' in data:
                eval_losses.append(data['eval_loss'])
                if epoch not in epochs:
                    epochs.append(epoch)  # Append epoch only when eval loss is present

# Ensure the lengths of epochs, train_losses, and eval_losses are consistent
min_length = min(len(epochs), len(train_losses), len(eval_losses))
epochs = epochs[:min_length]
train_losses = train_losses[:min_length]
eval_losses = eval_losses[:min_length]

# Plotting
plt.figure(figsize=(10, 5))

plt.plot(epochs, train_losses, label='Train Loss', color='blue')
plt.plot(epochs, eval_losses, label='Eval Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'Training and Evaluation Loss')
plt.legend()

plt.tight_layout()

# Save the plot in the output directory
plot_path = os.path.join(output_dir, 'training_evaluation_loss_plot.png')
plt.savefig(plot_path)
plt.close()

print(f"Plot saved in the current directory as 'training_evaluation_loss_plot.png'.")