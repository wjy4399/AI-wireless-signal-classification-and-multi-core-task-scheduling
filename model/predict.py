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
import pandas as pd
import wandb
import joblib
flag=1
if os.path.exists('best_model_info_co1.txt'):
    with open('best_model_info_co.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('dim_model'):
                dim_model = int(line.split(': ')[1])
            if line.startswith('num_heads'):
                num_heads = int(line.split(': ')[1])
            if line.startswith('num_layers'):
                num_layers = int(line.split(': ')[1])
            if line.startswith('dropout'):
                dropout = float(line.split(': ')[1])
            if line.startswith('Learning rate'):
                learning_rate = float(line.split(': ')[1])
            if line.startswith('Weight decay'):
                weight_decay = float(line.split(': ')[1])
            if line.startswith('Batch size'):
                batch_size = int(line.split(': ')[1])
        flag=0
if flag:
    # 确保转换为整数的参数为整数
    dim_model = int(64)
    num_heads = int(2)
    num_layers = int(1)
    dropout = float(0.1)
    learning_rate=1e-5
    weight_decay=1e-5
    batch_size=64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 数据预处理函数
def preprocess(folder_path, num_files=1400):
    data = []
    for i in range(num_files):
        filename = f'{i}.bin'
        with open(os.path.join(folder_path, filename), 'rb') as file:
            data_row_bin = file.read()
            data_row_float16 = np.frombuffer(data_row_bin, dtype=np.float16)
            data_row_float32 = data_row_float16.astype(np.float32)  # 转换为 float32
            paired_data = data_row_float32.reshape(15360, 2)  # 确保形状为 15360*2
            data.append(paired_data)
    return np.array(data)

# 加载测试集数据
test_data_path = 'test_set'  # 替换为实际的测试集路径
data_test = preprocess(test_data_path)

# 加载标准化器
scaler = joblib.load('scaler.pkl')
data_reshaped = data_test.reshape(data_test.shape[0], -1)  # 将数据展平以进行标准化
data_scaled = scaler.transform(data_reshaped).reshape(data_test.shape)  # 进行标准化并恢复形状
# 创建自定义数据集
class ComplexDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_data = self.data[idx]
        return torch.tensor(sample_data, dtype=torch.float32)

test_dataset = ComplexDataset(data_scaled)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义简化后的FT-Transformer模型
class SimplifiedFTTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, dim_model=dim_model, num_heads=num_heads, num_layers=num_layers, dropout=dropout):
        super(SimplifiedFTTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, dim_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model, nhead=num_heads, dropout=dropout)
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

# 加载最佳模型
input_dim = data_scaled.shape[1] * data_scaled.shape[2]  # 展平后的输入维度
num_classes = 11  # 假设类别数为11
model = SimplifiedFTTransformer(input_dim=input_dim, num_classes=num_classes).to(device)
model.load_state_dict(torch.load('best_model_co.pth'))

# 预测函数
def predict(model, data_loader, device):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for inputs in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
    return np.array(all_preds)

# 进行预测
predictions = predict(model, test_loader, device)

# 保存预测结果到CSV文件
output_path = 'result.csv'
df = pd.DataFrame(predictions, columns=['predictions'])
df.to_csv(output_path, index=False, header=False)
print(f'Predictions saved to {output_path}')
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
import pandas as pd
import wandb
import joblib
flag=1
if os.path.exists('best_model_info_co1.txt'):
    with open('best_model_info_co.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('dim_model'):
                dim_model = int(line.split(': ')[1])
            if line.startswith('num_heads'):
                num_heads = int(line.split(': ')[1])
            if line.startswith('num_layers'):
                num_layers = int(line.split(': ')[1])
            if line.startswith('dropout'):
                dropout = float(line.split(': ')[1])
            if line.startswith('Learning rate'):
                learning_rate = float(line.split(': ')[1])
            if line.startswith('Weight decay'):
                weight_decay = float(line.split(': ')[1])
            if line.startswith('Batch size'):
                batch_size = int(line.split(': ')[1])
        flag=0
if flag:
    # 确保转换为整数的参数为整数
    dim_model = int(64)
    num_heads = int(2)
    num_layers = int(1)
    dropout = float(0.1)
    learning_rate=1e-5
    weight_decay=1e-5
    batch_size=64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 数据预处理函数
def preprocess(folder_path, num_files=1400):
    data = []
    for i in range(num_files):
        filename = f'{i}.bin'
        with open(os.path.join(folder_path, filename), 'rb') as file:
            data_row_bin = file.read()
            data_row_float16 = np.frombuffer(data_row_bin, dtype=np.float16)
            data_row_float32 = data_row_float16.astype(np.float32)  # 转换为 float32
            paired_data = data_row_float32.reshape(15360, 2)  # 确保形状为 15360*2
            data.append(paired_data)
    return np.array(data)

# 加载测试集数据
test_data_path = 'test_set'  # 替换为实际的测试集路径
data_test = preprocess(test_data_path)

# 加载标准化器
scaler = joblib.load('scaler.pkl')
data_reshaped = data_test.reshape(data_test.shape[0], -1)  # 将数据展平以进行标准化
data_scaled = scaler.transform(data_reshaped).reshape(data_test.shape)  # 进行标准化并恢复形状
# 创建自定义数据集
class ComplexDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_data = self.data[idx]
        return torch.tensor(sample_data, dtype=torch.float32)

test_dataset = ComplexDataset(data_scaled)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义简化后的FT-Transformer模型
class SimplifiedFTTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, dim_model=dim_model, num_heads=num_heads, num_layers=num_layers, dropout=dropout):
        super(SimplifiedFTTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, dim_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model, nhead=num_heads, dropout=dropout)
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

# 加载最佳模型
input_dim = data_scaled.shape[1] * data_scaled.shape[2]  # 展平后的输入维度
num_classes = 11  # 假设类别数为11
model = SimplifiedFTTransformer(input_dim=input_dim, num_classes=num_classes).to(device)
model.load_state_dict(torch.load('best_model_co.pth'))

# 预测函数
def predict(model, data_loader, device):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for inputs in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
    return np.array(all_preds)

# 进行预测
predictions = predict(model, test_loader, device)

# 保存预测结果到CSV文件
output_path = 'result.csv'
df = pd.DataFrame(predictions, columns=['predictions'])
df.to_csv(output_path, index=False, header=False)
print(f'Predictions saved to {output_path}')
