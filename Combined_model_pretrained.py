import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import numpy as np
import torchvision.models as models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model_name = "dccuchile/bert-base-spanish-wwm-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm_weights_path = "/home/iotlab/Desktop/CARD-MAIN/LLM/daas.pt"
cnn_weights_path = "/home/iotlab/Desktop/CARD-MAIN/LLM/DENSENET_OPTUNA.pth"
class DenseNetEncoder(nn.Module):
    def __init__(self):
        super(DenseNetEncoder, self).__init__()
        self.densenet = models.densenet121(pretrained=False)
        in_features = self.densenet.classifier.in_features
        self.classifier = nn.Sequential(
            nn.Dropout(0.1097452),
            nn.Linear(in_features, 64),
            nn.Linear(64, 32)
        )
    def forward(self, x):
        features = self.densenet.features(x)
        out = nn.functional.adaptive_avg_pool2d(features, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
    def load_pretrained_weights(self, weights_path):
        print("Loading CNN weights from:", weights_path)
        full_model = models.densenet121(pretrained=False)
        in_features = full_model.classifier.in_features
        full_model.classifier = nn.Sequential(
            nn.Dropout(0.1097452),
            nn.Linear(in_features, 64),
            nn.Linear(64, 32),
            nn.Linear(32, 1)
        )
        state_dict = torch.load(weights_path, map_location=device)
        full_model.load_state_dict(state_dict)
        densenet_dict = {}
        for k, v in full_model.features.state_dict().items():
            densenet_dict["features." + k] = v
        self.densenet.load_state_dict(densenet_dict)
        dropout_weight = full_model.classifier[0].state_dict()
        linear1_weight = full_model.classifier[1].state_dict()
        linear2_weight = full_model.classifier[2].state_dict()
        self.classifier[0].load_state_dict(dropout_weight)
        self.classifier[1].load_state_dict(linear1_weight)
        self.classifier[2].load_state_dict(linear2_weight)
        print("CNN encoder weights loaded successfully")
        return
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
class BETOEncoder(nn.Module):
    def __init__(self):
        super(BETOEncoder, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        for param in self.bert.parameters():
            param.requires_grad = False
        self.attention_block = AttentionBlock(hidden_size=768, num_heads=4, dropout=0.1)
        self.projection = nn.Linear(768, 256)
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(256, 256)
        for param in self.fc1.parameters():
            param.requires_grad = False
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 32)
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
        x = self.relu(self.fc3(x))  
        return x
    def load_pretrained_weights(self, weights_path):
        print("Loading LLM weights from:", weights_path)
        temp_bert = AutoModel.from_pretrained(model_name)
        temp_attention = AttentionBlock(hidden_size=768, num_heads=4, dropout=0.1)
        class TempBERTMLP(nn.Module):
            def __init__(self):
                super(TempBERTMLP, self).__init__()
                self.bert = temp_bert
                self.attention_block = temp_attention
                self.projection = nn.Linear(768, 256)
                self.dropout = nn.Dropout(0.1)
                self.fc1 = nn.Linear(256, 256)
                for param in self.fc1.parameters():
                    param.requires_grad = False
                self.fc2 = nn.Linear(256, 64)
                self.fc3 = nn.Sequential(
                    nn.Linear(64, 32),
                    nn.Linear(32, 1)
                )
                self.relu = nn.ReLU()
            def forward(self, x):
                pass
        temp_model = TempBERTMLP()
        state_dict = torch.load(weights_path, map_location=device)
        temp_model.load_state_dict(state_dict)
        self.bert.load_state_dict(temp_model.bert.state_dict())
        self.attention_block.load_state_dict(temp_model.attention_block.state_dict())
        self.projection.load_state_dict(temp_model.projection.state_dict())
        self.fc1.load_state_dict(temp_model.fc1.state_dict())
        self.fc2.load_state_dict(temp_model.fc2.state_dict())
        self.fc3.load_state_dict(temp_model.fc3[0].state_dict())
        for param in self.fc1.parameters():
            param.requires_grad = False
        print("LLM encoder weights loaded successfully")
        return
class CombinedModel(nn.Module):
    def __init__(self):
        super(CombinedModel, self).__init__()
        self.cnn_encoder = DenseNetEncoder()
        self.llm_encoder = BETOEncoder()
        self.combined_classifier = nn.Sequential(
            nn.Linear(64, 16),  
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 8),   
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(8, 4),    
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(4, 1)     
        )
    def forward(self, image, input_ids, attention_mask):
        cnn_embedding = self.cnn_encoder(image)
        llm_embedding = self.llm_encoder(input_ids, attention_mask)
        combined_embedding = torch.cat((cnn_embedding, llm_embedding), dim=1)
        output = self.combined_classifier(combined_embedding)
        return output
    def load_pretrained_weights(self):
        try:
            self.cnn_encoder.load_pretrained_weights(cnn_weights_path)
            self.llm_encoder.load_pretrained_weights(llm_weights_path)
            for param in self.llm_encoder.fc1.parameters():
                param.requires_grad = False
            print("Successfully loaded weights for both encoders")
            print("Verified fc1 layer is frozen (requires_grad=False)")
        except Exception as e:
            print(f"Error loading weights: {e}")
class CombinedDataset(Dataset):
    def __init__(self, image_dir, df, transform=None):
        self.image_dir = image_dir
        self.df = df
        self.transform = transform
        self.valid_indices = []
        for idx, row in df.iterrows():
            img_id = row['ImageID']
            img_path = os.path.join(image_dir, img_id)
            if os.path.isfile(img_path):
                self.valid_indices.append(idx)
        print(f"Found {len(self.valid_indices)} valid samples with both image and text data")
    def __len__(self):
        return len(self.valid_indices)
    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        row = self.df.iloc[actual_idx]
        img_path = os.path.join(self.image_dir, row['ImageID'])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        text = f"{row['PatientSex_DICOM']} {row['PatientBirth']} {row['Report']}"
        encoding = tokenizer(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        if row['Labels'] == "['normal']":
            label = 0
        else:
            label = 1
        return {
            'image': image,
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.float)
        }
def train_combined_model(model, train_loader, val_loader, epochs=5, lr=1e-4):
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    best_val_accuracy = 0.0
    for epoch in range(epochs):
        model.train()
        for param in model.llm_encoder.fc1.parameters():
            param.requires_grad = False
        train_loss = 0.0
        correct = 0
        total = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]")
        for batch in progress_bar:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            optimizer.zero_grad()
            outputs = model(images, input_ids, attention_mask)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            preds = (torch.sigmoid(outputs.squeeze()) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            progress_bar.set_postfix({
                'loss': f"{train_loss / (progress_bar.n + 1):.4f}", 
                'acc': f"{100.0 * correct / total:.2f}%"
            })
        train_loss = train_loss / len(train_loader)
        train_accuracy = 100.0 * correct / total
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Validation]"):
                images = batch['image'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(images, input_ids, attention_mask)
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item()
                preds = (torch.sigmoid(outputs.squeeze()) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_loss = val_loss / len(val_loader)
        val_accuracy = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{epochs}, "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), "best_combined_model.pt")
            print(f"New best model saved with validation accuracy: {val_accuracy:.2f}%")
    print(f"Training completed. Best validation accuracy: {best_val_accuracy:.2f}%")
    return model
def main():
    extract_dir = "/home/iotlab/Desktop/CARD-MAIN/LLM/images-224"
    file_path = "/home/iotlab/Desktop/CARD-MAIN/LLM/d/hello.xlsx"
    df = pd.read_excel(file_path)
    transform = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.RandomHorizontalFlip(),                     
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = CombinedDataset(extract_dir, df, transform=transform)
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    model = CombinedModel()
    model.load_pretrained_weights()
    for name, param in model.named_parameters():
        if 'fc1' in name:
            print(f"Parameter {name}: requires_grad = {param.requires_grad}")
    train_combined_model(model, train_loader, val_loader, epochs=5, lr=1e-4)
    print("Combined model training completed!")
if __name__ == "__main__":
    main()