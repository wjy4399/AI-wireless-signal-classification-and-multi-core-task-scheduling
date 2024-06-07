import torch
import pandas as pd
import os
import re
import numpy as np
from torch.utils.data import DataLoader, Dataset
from train import Model


def preprocess(folder_path, target_size=30720):
    data = []
    file_list = os.listdir(folder_path)

    # 对文件名进行排序，假设文件名从0到1399
    file_list.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))

    for filename in file_list:
        if filename.endswith(".bin"):
            with open(os.path.join(folder_path, filename), 'rb') as file:
                data_row_bin = file.read()
                data_row_float16 = np.frombuffer(data_row_bin, dtype=np.float16)
                if data_row_float16.size < target_size:
                    padding = np.zeros(target_size - data_row_float16.size, dtype=np.float16)
                    data_row_float16 = np.concatenate((data_row_float16, padding), axis=0)
                data_row_float16 = data_row_float16[:target_size]
                data.append(data_row_float16)
    return np.array(data), file_list  # 返回numpy数组和文件名列表


folder_path = r'D:\课外文件\比赛\智联杯\本地调试的数据集\2.AI场景分类测试集(无标签)\test_set'
data, file_list = preprocess(folder_path)


class ComplexDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {'data': torch.tensor(self.data[idx], dtype=torch.float32)}
        return sample


# 创建测试数据集实例
test_dataset = ComplexDataset(data)


def collate_fn(batch):
    features = []
    for item in batch:
        features.append(item['data'])
    return torch.stack(features, 0)


# 构建test_loader
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

# 加载模型
model_path = r'D:\课外文件\比赛\智联杯\example(python)\model\model_best.pth'
model = torch.load(model_path, map_location='cpu')
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 假设test_loader用于加载无标签的测试数据
predictions = []

with torch.no_grad():
    for inputs in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)  # 获取最大概率的类别索引作为预测结果
        predictions.extend(preds.cpu().numpy())  # 将预测结果从GPU转移到CPU，并添加到列表中

# 将预测结果和文件名按顺序保存到CSV文件，去除表头
df_predictions = pd.DataFrame({'Filename': file_list, 'Prediction': predictions})
csv_output_path = os.path.join(os.path.dirname(folder_path), 'result.csv')
df_predictions.to_csv(csv_output_path, index=False, header=False)  # index=False避免将索引写入CSV文件, header=False避免写入表头

print(f'Predictions have been saved to {csv_output_path}')
