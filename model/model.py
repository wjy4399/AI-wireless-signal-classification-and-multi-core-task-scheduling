import torch
import torch.nn as nn
import torch.nn.functional as F

# class Model2(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.fc1 = nn.Linear(30720, 128)
#         self.fc2 = nn.Linear(128, 11)

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
    
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Input layer
        self.fc1 = nn.Linear(30720, 1024)  # Expanded first layer
        self.bn1 = nn.BatchNorm1d(1024)  # Batch normalization layer
        self.dropout1 = nn.Dropout(0.5)  # Dropout layer
        
        # Additional hidden layers
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(0.5)
        
        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.dropout4 = nn.Dropout(0.5)

        # Output layer
        self.fc5 = nn.Linear(128, 11)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.dropout4(x)
        
        x = self.fc5(x)  # No activation needed here, it will be used with CrossEntropyLoss
        return x