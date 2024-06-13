import os
import re
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import wandb
import gc
from imblearn.over_sampling import SMOTE

record = not True
if record:
    # 初始化 W&B
    wandb.init()
    config = wandb.config
    # 假设联合参数的格式为 "dim_model-num_heads-num_layers-dropout"
    dim_model, num_heads, num_layers, dropout = map(float, config.combined_param.split('-'))
    # 确保转换为整数的参数为整数
    dim_model = int(dim_model)
    num_heads = int(num_heads)
    num_layers = int(num_layers)
    dropout = float(dropout)
    learning_rate = config.learning_rate
    weight_decay = config.weight_decay
    batch_size = config.batch_size
else:
    dim_model = int(64)
    num_heads = int(2)
    num_layers = int(1)
    dropout = float(0.1)
    learning_rate = 2e-5
    weight_decay = 1e-5
    batch_size = 64


num_workers = 8

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

def free_memory(*args):#解放内存
    for arg in args:
        del arg
    gc.collect()


# 数据预处理函数
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
                data_row_float16 = np.frombuffer(data_row_bin, dtype=np.float16)
                data_row_float32 = data_row_float16.astype(np.float32)  # 转换为 float32
                paired_data = data_row_float32.reshape(-1, 2)  # 两两配对
                labels.append(label)
                data.append(paired_data)

    return np.array(data), np.array(labels)

path1 = 'train_set_remake'
data1, labels1 = preprocess(path1)

print(f"Data shape: {data1.shape}")
print(f"Labels shape: {labels1.shape}")

# 数据分割
split = StratifiedShuffleSplit(n_splits=1, test_size=0.125, random_state=42)

for train_idx, val_idx in split.split(data1, labels1):
    X_train, X_val = data1[train_idx], data1[val_idx]
    y_train, y_val = labels1[train_idx], labels1[val_idx]

from collections import Counter

# 检查原始数据集、训练集和验证集中各类别的比例
def check_class_distribution(labels, dataset_name):
    counter = Counter(labels)
    total = len(labels)
    print(f"Class distribution in {dataset_name}:")
    for cls, count in counter.items():
        print(f"Class {cls}: {count} samples, proportion: {count / total:.2%}")

# 打印原始数据集、训练集和验证集的类别分布
check_class_distribution(labels1, "original dataset")
check_class_distribution(y_train, "training set")
check_class_distribution(y_val, "validation set")

# 数据标准化
scaler = StandardScaler()
X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
X_val_reshaped = X_val.reshape(X_val.shape[0], -1)

X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(X_train.shape)
X_val_scaled = scaler.transform(X_val_reshaped).reshape(X_val.shape)

# 过采样

desired_total_samples_per_class = 400  # 这里设定每个类别最终的样本数量
sampling_strategy_over = {label: desired_total_samples_per_class for label in np.unique(y_train) if np.sum(y_train == label) < desired_total_samples_per_class}

# 计算 sampling_strategy
sampling_strategy = {label: desired_total_samples_per_class for label in np.unique(y_train)}

sm = SMOTE(sampling_strategy=sampling_strategy_over, random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_reshaped, y_train)
X_train_res = X_train_res.reshape(-1, X_train.shape[1], X_train.shape[2])  # 恢复原始形状
counter_resampled = Counter(y_train_res)
print("Resampled training set label distribution:")
for label, count in counter_resampled.items():
    print(f"Label {label}: {count} samples")


'''X_train_res, y_train_res=X_train_scaled, y_train'''
X_val_res, y_val_res = X_val_scaled, y_val


# 创建自定义数据集
class ComplexDataset(Dataset):
    def __init__(self, data, labels, augment=False):
        self.data = data
        self.labels = labels
        self.augment = augment
        self.transform = transforms.Compose([
            transforms.RandomRotation(5),  # 旋转角度从 10° 降到 5°
            transforms.RandomHorizontalFlip(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_data = self.data[idx]
        sample_label = self.labels[idx]

        if self.augment:
            sample_data = sample_data.reshape(1, -1, 2)  # Reshape for augmentation (2 features)
            sample_data = torch.tensor(sample_data, dtype=torch.float32)
            sample_data = self.transform(sample_data)
            sample_data = sample_data.numpy().reshape(-1, 2)  # Reshape back to original shape

        return torch.tensor(sample_data, dtype=torch.float32), torch.tensor(sample_label, dtype=torch.long)

train_dataset = ComplexDataset(X_train_res, y_train_res, augment=False)
val_dataset = ComplexDataset(X_val_res, y_val_res, augment=False)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# 简化后的FT-Transformer模型
class SimplifiedFTTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, dim_model=dim_model, num_heads=num_heads, num_layers=num_layers, dropout=dropout):
        super(SimplifiedFTTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, dim_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model, nhead=num_heads, dropout=dropout, batch_first=True)  # 修改这里，添加 batch_first=True
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(dim_model, num_classes)

    def forward(self, x):
        batch_size, seq_len, feature_dim = x.shape
        x = x.view(batch_size, -1)  # 展平特征维度
        x = self.embedding(x)
        x = self.transformer_encoder(x.unsqueeze(1))
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x

# 初始化和训练FT-Transformer模型
input_dim = X_train_scaled.shape[1] * X_train_scaled.shape[2]  # 展平后的输入维度
num_classes = len(np.unique(y_train))
model = SimplifiedFTTransformer(input_dim=input_dim, num_classes=num_classes).to(device)

# 计算类别权重
class_weights = torch.tensor([1.0 / y_train[y_train == i].shape[0] for i in np.unique(y_train)], dtype=torch.float).to(device)

# 定义损失函数时使用类别权重
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

num_epochs = 1000

# 记录最佳模型信息的函数
def save_best_model(model, accuracy, batch_size, learning_rate, weight_decay, epoch, path='best_model_co.pth'):
    torch.save(model.state_dict(), path)
    with open('best_model_info_co.txt', 'w') as f:
        f.write(f'dim_model: {dim_model}\n')
        f.write(f'num_heads: {num_heads}\n')
        f.write(f'num_layers: {num_layers}\n')
        f.write(f'dropout: {dropout}\n')
        f.write(f'Accuracy: {accuracy}\n')
        f.write(f'Batch size: {batch_size}\n')
        f.write(f'Learning rate: {learning_rate}\n')
        f.write(f'Weight decay: {weight_decay}\n')
        f.write(f'Epoch: {epoch}\n')

# 读取当前最佳模型的准确率
def load_best_accuracy():
    if os.path.exists('best_model_info_co.txt'):
        with open('best_model_info_co.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith('Accuracy'):
                    return float(line.split(': ')[1])
    return 0.0

# 早停机制
class EarlyStopping:
    def __init__(self, patience=30, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, epoch, accuracy, train_loader, learning_rate, weight_decay):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.val_loss_min = val_loss
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and accuracy>0.85:
                self.early_stop = True
        else:
            self.best_score = score
            self.val_loss_min = val_loss
            self.counter = 0

best_accuracy = load_best_accuracy()
early_stopping = EarlyStopping(patience=15, verbose=True)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    all_targets = []
    all_preds = []
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, y_pred_ft = torch.max(outputs, 1)
        correct += (y_pred_ft == targets).sum().item()
        all_targets.extend(targets.cpu().numpy())
        all_preds.extend(y_pred_ft.cpu().numpy())

    avg_loss = total_loss / len(train_loader)
    train_accuracy = correct / len(train_dataset)
    train_f1 = f1_score(all_targets, all_preds, average='weighted')
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train F1 Score: {train_f1:.4f}')

    # 评估模型在验证集上的表现
    model.eval()
    correct = 0
    total = 0
    all_targets = []
    all_preds = []
    with torch.no_grad():
        total_loss = 0
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, y_pred_ft = torch.max(outputs, 1)
            correct += (y_pred_ft == targets).sum().item()
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(y_pred_ft.cpu().numpy())
        avg_val_loss = total_loss / len(val_loader)

    val_accuracy = correct / len(val_dataset)
    val_f1 = f1_score(all_targets, all_preds, average='weighted')
    print(f'Validation Accuracy after epoch {epoch+1}: {val_accuracy:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation F1 Score: {val_f1:.4f}')
    best_accuracy = load_best_accuracy()
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        save_best_model(model, val_accuracy, train_loader.batch_size, learning_rate, weight_decay, epoch+1)

    if record and (epoch+1)%5==0:
        # 记录到 W&B
        wandb.log({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "train_accuracy": train_accuracy,
            "train_f1": train_f1,
            "val_loss": avg_val_loss,
            "val_accuracy": val_accuracy,
            "val_f1": val_f1
        })
        # 保存模型权重到 W&B
        wandb.save('best_model_co.pth')

    # 调整学习率
    scheduler.step(avg_val_loss)

    # 使用早停机制
    early_stopping(avg_val_loss, model, epoch, val_accuracy, train_loader, learning_rate, weight_decay)

    if early_stopping.early_stop:
        print("Early stopping")
        break
    free_memory(inputs, targets, outputs)

# 打印预测结果的示例
print(f'Sample predictions: {y_pred_ft[:10]}')
