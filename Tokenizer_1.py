import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, classification_report
from tqdm import tqdm
import pandas as pd
from sklearn.utils import resample

torch.manual_seed(42)
np.random.seed(42)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

model_name = "dccuchile/bert-base-spanish-wwm-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
beto_model = AutoModel.from_pretrained(model_name).to(device)

class AttentionBlock(nn.Module):
    def __init__(self, hidden_size, num_heads=4, dropout=0.1):
        super(AttentionBlock, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, dropout=dropout)
        self.layernorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        attn_output, _ = self.multihead_attn(x, x, x)
        x = self.layernorm(x + self.dropout(attn_output))
        return x

class SentimentDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = tokenizer(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        item = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }
        if self.labels is not None:
            item['labels'] = torch.tensor(float(self.labels[idx]), dtype=torch.float)
        return item

class BertMLP(nn.Module):
    def __init__(self, bert_model, projection_dim=256, dropout=0.1):
        super(BertMLP, self).__init__()
        self.bert = bert_model
        for param in self.bert.parameters():
            param.requires_grad = False
        self.attention_block = AttentionBlock(hidden_size=768, num_heads=4, dropout=dropout)
        self.projection = nn.Linear(768, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(projection_dim, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Sequential(
            nn.Linear(64, 32),
            nn.Linear(32, 1)
        )
        self.relu = nn.ReLU()
       
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = sequence_output.transpose(0, 1)
        sequence_output = self.attention_block(sequence_output)
        sequence_output = sequence_output.transpose(0, 1)
        cls_embedding = sequence_output[:, 0, :]
        proj_embedding = self.relu(self.projection(cls_embedding))
        x = self.dropout(proj_embedding)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        logits = self.fc3(x)
        return logits

def calculate_class_weights_full(labels):
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    weights = total_samples / (len(class_counts) * class_counts)
    return torch.FloatTensor(weights)

def prepare_data(combined_df, labels_series):
    sentences = [str(text) for text in combined_df['combined_text'].tolist()]
    labels = labels_series.tolist()
    X_train, X_val, y_train, y_val = train_test_split(
        sentences, labels, test_size=0.2, random_state=42, stratify=labels
    )
    train_dataset = SentimentDataset(X_train, y_train)
    val_dataset = SentimentDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    return train_loader, val_loader

def train_model_with_loss(model, train_loader, val_loader, loss_fn, class_weights=None, epochs=5, learning_rate=2e-5):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_val_f1 = 0
    if class_weights is not None:
        class_weights = class_weights.to(device)
    for epoch in range(1):
        model.train()
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs.squeeze(), labels)
            if class_weights is not None:
                loss = loss * class_weights[labels.long()]
                loss = loss.mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        model.eval()
        val_preds = []
        val_true = []
        val_probs = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask)
                probs = torch.sigmoid(outputs.squeeze()).cpu().numpy()
                predictions = (torch.sigmoid(outputs.squeeze()) > 0.5).int()
                val_preds.extend(predictions.cpu().numpy())
                val_true.extend(labels.cpu().numpy().astype(int))
                val_probs.extend(probs)
        precision, recall, f1, _ = precision_recall_fscore_support(val_true, val_preds, average='binary')
        auc = roc_auc_score(val_true, val_probs)
        accuracy = np.mean(np.array(val_preds) == np.array(val_true))
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(val_true, val_preds))
    return model

def main():
    file_path = "/home/iotlab/Desktop/CARD-MAIN/LLM/d/hello.xlsx"
    full_df = pd.read_excel(file_path)
    full_df['Labels_numeric'] = full_df['Labels'].apply(lambda x: 0 if x == "['normal']" else 1)
    labels = full_df['Labels_numeric']
    selected_columns = full_df[['PatientSex_DICOM', 'PatientBirth', 'Report']].astype(str)
    combined_df = pd.DataFrame({'combined_text': selected_columns.agg(' '.join, axis=1)})
    train_loader, val_loader = prepare_data(combined_df, labels)
    model = BertMLP(beto_model).to(device)
    train_model_with_loss(model, train_loader, val_loader, nn.BCEWithLogitsLoss(), epochs=5, learning_rate=2e-5)

if __name__ == "__main__":
    main()
