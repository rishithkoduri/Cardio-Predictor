import pandas as pd
import numpy as np
import torch     # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
from torch.utils.data import DataLoader, TensorDataset # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

print("Loading cardio_train.csv...")
df = pd.read_csv('cardio_train.csv', sep=';')

df.drop('id', axis=1, inplace=True)

df = df[(df['ap_hi'] > 60) & (df['ap_hi'] < 240)]
df = df[(df['ap_lo'] > 40) & (df['ap_lo'] < 160)]

X = df.drop("cardio", axis=1).values
y = df["cardio"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

class CardioTransformer(nn.Module):
    def __init__(self, num_features, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super(CardioTransformer, self).__init__()

        self.embedding = nn.Linear(1, d_model) 

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.Linear(num_features * d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        x = x.unsqueeze(-1)
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = x.view(x.size(0), -1)
        output = self.fc(x)
        return output
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_features = X.shape[1] 
model = CardioTransformer(num_features=num_features).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"Training on {len(X_train)} samples...")
epochs = 20 
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader):.4f}")

model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor.to(device))
    predicted = (test_outputs > 0.5).float()
    acc = accuracy_score(y_test, predicted.cpu().numpy())

print(f"✅ Final Accuracy: {acc:.2%}")

torch.save(model.state_dict(), 'cardio_transformer.pth')
joblib.dump(scaler, 'scaler_custom.pkl')
print("Model saved as 'cardio_transformer.pth'")