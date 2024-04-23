import joblib
import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader

# Load labeled data
with open('mixed_labeled_data.txt', 'r') as file:
    lines = file.readlines()

# Extract features (words) and labels from the data
words = []
labels = []
for line in lines:
    label, word = line.strip().split()
    words.append(word)
    labels.append(int(label))

# Encode labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Save LabelEncoder
joblib.dump(label_encoder, 'label_encoder.pkl')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(words, labels, test_size=0.2, random_state=42)

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize input data
max_len = 128
X_train_tokens = tokenizer(X_train, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
X_test_tokens = tokenizer(X_test, padding=True, truncation=True, max_length=max_len, return_tensors='pt')

# Create PyTorch datasets
train_dataset = TensorDataset(X_train_tokens['input_ids'], X_train_tokens['attention_mask'], torch.tensor(y_train))
test_dataset = TensorDataset(X_test_tokens['input_ids'], X_test_tokens['attention_mask'], torch.tensor(y_test))

# Create data loaders
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Define BERT-based classifier
class BERTClassifier(nn.Module):
    def __init__(self, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs['pooler_output']
        output = self.dropout(pooled_output)
        output = self.fc(output)
        return output

# Initialize BERT classifier
num_classes = len(label_encoder.classes_)
classifier = BERTClassifier(num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(classifier.parameters(), lr=2e-5)  # Using AdamW optimizer for BERT

# Training loop
num_epochs = 5  # Increase the number of epochs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classifier.to(device)
classifier.train()

for epoch in range(num_epochs):
    total_loss = 0
    for inputs, attention_mask, labels in train_loader:
        inputs, attention_mask, labels = inputs.to(device), attention_mask.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = classifier(inputs, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Avg. Loss: {total_loss / len(train_loader)}")

# Evaluation
classifier.eval()
predictions = []
true_labels = []
with torch.no_grad():
    for inputs, attention_mask, labels in test_loader:
        inputs, attention_mask, labels = inputs.to(device), attention_mask.to(device), labels.to(device)
        outputs = classifier(inputs, attention_mask)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

# Calculate accuracy and other metrics
accuracy = accuracy_score(true_labels, predictions)
precision = precision_score(true_labels, predictions)
recall = recall_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions)

print("Test Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


# Save the trained model
torch.save(classifier.state_dict(), 'bert_model_updated3.pt')


