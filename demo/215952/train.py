import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter


# 定义模型
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(30720, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 11)  # 确保输出层有11个神经元，对应11个类别
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def preprocess(folder_path):
    labels = []
    data = []
    for filename in os.listdir(folder_path):
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


class ComplexDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.scaler = StandardScaler()
        self.data = self.scaler.fit_transform(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {'data': torch.tensor(self.data[idx], dtype=torch.float32),
                  'label': torch.tensor(self.labels[idx], dtype=torch.long)}
        return sample


def collate_fn(batch):
    features = []
    labels = []
    for _, item in enumerate(batch):
        features.append(item['data'])
        labels.append(item['label'])
    return torch.stack(features, 0), torch.stack(labels, 0)


if __name__ == "__main__":
    # 加载数据
    folder_path = r'D:\课外文件\比赛\智联杯\本地调试的数据集\1.AI场景分类训练集(带标签)\train_set_remake\train_set_remake'
    data, labels = preprocess(folder_path)

    # 统计每个类别的数量
    label_counts = Counter(labels)
    print(f"Label distribution: {label_counts}")

    # 计算类别权重
    total_samples = len(labels)
    class_weights = {label: total_samples / count for label, count in label_counts.items()}
    weights = [class_weights[label] for label in labels]

    # 划分数据集
    train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.2, random_state=42)
    train_dataset = ComplexDataset(train_data, train_labels)
    val_dataset = ComplexDataset(val_data, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)

    # 定义模型
    model = Model()
    class_weights_tensor = torch.tensor([class_weights[i] for i in range(11)], dtype=torch.float32).to(
        'cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

    # 使用混合精度训练
    scaler = torch.cuda.amp.GradScaler()

    # 训练模型
    num_epochs = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)

    best_accuracy = 0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        # 验证集上的评估
        model.eval()
        val_loss = 0.0
        total_correct = 0
        correct_per_class = [0] * 11
        samples_per_class = [0] * 11
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == targets).sum().item()
                for i in range(11):
                    correct_per_class[i] += ((predicted == targets) & (targets == i)).sum().item()
                    samples_per_class[i] += (targets == i).sum().item()

        accuracy = total_correct / len(val_dataset)
        print(
            f'Epoch {epoch + 1}, Train Loss: {train_loss / len(train_loader)}, Validation Loss: {val_loss / len(val_loader)}, Validation Accuracy: {accuracy}')
        print(
            f'Per-class Accuracy: {[correct_per_class[i] / samples_per_class[i] if samples_per_class[i] != 0 else 0 for i in range(11)]}')

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model, f'model_best.pth')

    print(f'Best Validation Accuracy: {best_accuracy}')
