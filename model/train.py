import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# # 定义模型
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
# 定义模型
# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.fc1 = nn.Linear(30720, 128)
#         self.fc2 = nn.Linear(128, 11)

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

def preprocess(folder_path):
    labels = []
    data = []
    cnt = 0
    for filename in os.listdir(folder_path):
        cnt += 1
        if filename.endswith(".bin"):
            match = re.search(r'label_(\d+)_', filename)
            if match:
                label = int(match.group(1))
            else:
                continue
            with open(os.path.join(folder_path, filename), 'rb') as file:
                data_row_bin = file.read()
                labels.append(label)
                data_row_float16 = np.frombuffer(data_row_bin, dtype=np.float16)  # 原始数据是float16，直接把二进制bin读成float16的数组
                data_row_float16 = np.array(data_row_float16)
                data.append(data_row_float16)
    return data, labels


# 加载数据
folder_path = "../dataset/train_set_remake"
data, labels = preprocess(folder_path)

# 划分数据集
train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.2, random_state=42)


class ComplexDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {'data': torch.tensor(self.data[idx], dtype=torch.float32),
                  'label': torch.tensor(self.labels[idx], dtype=torch.long)}
        return sample


train_dataset = ComplexDataset(train_data, train_labels)
val_dataset = ComplexDataset(val_data, val_labels)


def collate_fn(batch):
    features = []
    labels = []
    for _, item in enumerate(batch):
        features.append(item['data'])
        labels.append(item['label'])
    return torch.stack(features, 0), torch.stack(labels, 0)


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# 定义模型
model = Model()

# 载入预训练模型
model_path = './model/pth/model_8500.pth'
if os.path.exists(model_path):
    model = torch.load(model_path)
    print("Loaded pretrained model from", model_path)
else:
    print("No pretrained model found at", model_path)
    
criterion = nn.CrossEntropyLoss()
# optimizer = optim.ASGD(model.parameters(), lr=0.001, weight_decay=1e-5)
# optimizer = optim.RAdam(model.parameters(), lr=0.001, weight_decay=1e-5)
optimizer = optim.NAdam(model.parameters(), lr=0.001, weight_decay=1e-5)

# 训练模型
num_epochs = 100000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)
accuracy_threshold = 0.80  # 设置保存模型的准确率阈值为90%

for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = torch.softmax(outputs, 1)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    # 验证集上的评估
    model.eval()
    with torch.no_grad():
        total_correct = 0
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == targets).sum().item()

    accuracy = total_correct / len(val_dataset)
    print(f'Epoch {epoch + 1}, Validation Accuracy: {accuracy}')
    # torch.save(model, f'./model/pth/model_{int(accuracy * 10000)}.pth')
        # 只有当准确率高于阈值时才保存整个模型
    if accuracy > accuracy_threshold:
        model_path = f'./model/pth/model_{int(accuracy * 10000)}.pth'
        torch.save(model, model_path)  # 保存整个模型
        print(f'---Complete model saved to {model_path}---')
