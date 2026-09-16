import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from review_classifier import ReviewClassifier

#model parameters

p_batch_size = 32
p_lr = 5e-5
p_max_length = 256
p_epochs = 3
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
torch.backends.cudnn.benchmark = True

#loading data
class IMDBDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True, max_length=p_max_length)

def load_data():
    dataset = load_dataset('stanfordnlp/imdb')
    train_data = dataset['train']
    test_data = dataset['test']

    train_encodings = train_data.map(tokenize, batched=True)
    test_encodings = test_data.map(tokenize, batched=True)

    train_dataset = IMDBDataset(
        encodings={
            'input_ids': train_encodings['input_ids'],
            'attention_mask': train_encodings['attention_mask']
        },
        labels=train_encodings['label']
    )
    
    test_dataset = IMDBDataset(
        encodings={
            'input_ids': test_encodings['input_ids'],
            'attention_mask': test_encodings['attention_mask']
        },
        labels=test_encodings['label']
    )
    
    train_loader = DataLoader(train_dataset, batch_size=p_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=p_batch_size, shuffle=False)

    return train_loader, test_loader

#main cycle
if __name__ == '__main__':
    train_loader, test_loader = load_data()

    criterion = nn.CrossEntropyLoss()
    
    model = ReviewClassifier()
    optimizer = AdamW(model.parameters(), lr=p_lr)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    num_epochs = p_epochs
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
    
        for i, batch in enumerate(train_loader):
            if i % 100 == 0:
                print(f"Batch {i}/{len(train_loader)}")
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
    
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            total_loss += loss.item()
    
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
    
            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')
    
    os.makedirs('model', exist_ok=True)
    tokenizer.save_pretrained('model/')
    model.model.save_pretrained('model/')
    torch.save(model.classifier.state_dict(), 'model/classifier.pth')