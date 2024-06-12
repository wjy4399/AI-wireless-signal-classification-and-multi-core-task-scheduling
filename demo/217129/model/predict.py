import datetime
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from model import Model 

def preprocess_test(folder_path):
    """
    从指定文件夹读取无标签的.bin文件，将每个文件的内容读取为float16格式的numpy数组。

    Args:
    folder_path (str): 包含无标签数据集的文件夹路径。

    Returns:
    numpy.ndarray: 预处理后的数据。
    """
    data = []
    cnt = 0
    for filename in os.listdir(folder_path):
        cnt += 1
        if filename.endswith(".bin"):
            # print(f'Processing {cnt}th file: {filename}')
            with open(os.path.join(folder_path, filename), 'rb') as file:
                data_row_bin = file.read()
                data_row_float16 = np.frombuffer(data_row_bin, dtype=np.float16)  # 原始数据是float16，直接把二进制bin读成float16的数组
                data_row_float16 = np.array(data_row_float16)
                data.append(data_row_float16)

    # 将数据列表转换为numpy数组
    if data:
        all_data = np.vstack(data)  # 堆叠成一个大的numpy数组
    else:
        all_data = np.array([])  # 如果没有数据文件，则返回空数组

    return all_data

folder_path = "../dataset/test_set/"
data = preprocess_test(folder_path)


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
    for _, item in enumerate(batch):
        features.append(item['data'])
    return torch.stack(features, 0)


# 构建test_loader
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# 加载模型
model_path = './model/model_1322.pth'  # 替换为你的模型路径
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

df_predictions = pd.DataFrame({'Prediction': predictions})

# 将预测结果保存到CSV文件，提交时注意去除表头
current_datetime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
csv_output_path = f'./result-{current_datetime}.csv'
df_predictions.to_csv(csv_output_path, index=False,header=False)  # index=False避免将索引写入CSV文件

print(f'Predictions have been saved to {csv_output_path}')
