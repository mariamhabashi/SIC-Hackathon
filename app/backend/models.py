import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim=126, hidden_dim=256):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :]) 
        return x

class Head(nn.Module):
    def __init__(self, num_classes, hidden_dim=512):
        super(Head, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        return self.fc(x)