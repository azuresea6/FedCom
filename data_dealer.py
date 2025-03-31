import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import torch
from torch.utils.data import Dataset, DataLoader
import os
import random

# 创建文件名列表作为跟踪函数的输入 ['path/name1','path/name2']
tau_max = 8
tau_min = 4
stren = True
inf = False
train_size = 0.9
net_batch_size = 5
stren_length = 60

def data_tracker(target_path):
    new_df = pd.DataFrame()
    name_list = ['Xiong_data.xlsx', 'Guo_data.xlsx', 'Dong_data.xlsx','Xu_data.xlsx']
    for i, name in enumerate(name_list):
        try:
            excel_file_path = target_path + f'\{name}'
            df = pd.read_excel(excel_file_path, sheet_name='Sheet1')[f'worker_{i}']
            new_df[f'worker_{i}'] = df.values.flatten()
        except Exception as e:
            print(f"Error reading {excel_file_path}: {e}")
    csv_file_path = target_path
    new_df.to_csv(csv_file_path + '/output.csv', index=False)

def mean_std_tracker(target_path, user_index):
    name_list = ['Xiong_data.xlsx', 'Guo_data.xlsx', 'Dong_data.xlsx','Xu_data.xlsx']
    excel_file_path = target_path + f'/{name_list[user_index]}'
    df = pd.read_csv(excel_file_path, sheet_name='Sheet2', header=None)
    mean = df.loc[0, 'value'] 
    std = df.loc[1, 'value'] 
    return mean, std

def value_strenthen(output_path, if_str, length):
    df = pd.read_csv(output_path + '/output.csv', names=['workers_1', 'workers_2', 'workers_3', 'workers_4'], header=None)
    df.head(4)
    if not if_str:
        df.to_csv(output_path + '/mid_output.csv', index=False)
    else:
        mean = []
        std = []
        for k in range(4):
            Name_list = ['Xiong_data.xlsx', 'Guo_data.xlsx', 'Dong_data.xlsx','Xu_data.xlsx']
            new_df = pd.read_excel(output_path + f'/{Name_list[k]}', sheet_name='Sheet2', header=None)
            mean.append(new_df.iat[1,0])
            std.append(new_df.iat[1,1])

        for k in range(length):
            new_row = {}
            for name in ['workers_1', 'workers_2', 'workers_3', 'workers_4']:
                i = ['workers_1', 'workers_2', 'workers_3', 'workers_4'].index(name)
                rand = np.random.normal(loc=mean[i], scale=std[i])
                new_row[name] = rand if rand > 0 else -rand
            df.loc[len(df)] = new_row
        
        df.to_csv(output_path + '/mid_output.csv', index=False)

def value_correction(output_path):
    name_list = ['workers_1', 'workers_2', 'workers_3', 'workers_4']
    Name_list = ['Xiong_data.xlsx', 'Guo_data.xlsx', 'Dong_data.xlsx','Xu_data.xlsx']
    df = pd.read_csv(output_path + '/mid_output.csv', names=name_list, header=None)
    for i in range(4):
        new_df = pd.read_excel(output_path + f'/{Name_list[i]}', sheet_name='Sheet2', header=None)
        mean = new_df.iat[1,0]
        std = new_df.iat[1,1]
        df[name_list[i]] = pd.to_numeric(df[name_list[i]], errors='coerce').dropna()
        data = df[name_list[i]][1:]
        current_mean = data.mean()
        current_std = data.std()
        for j in range(len(data)):
            df.loc[j, name_list[i]] = ((df.loc[j, name_list[i]] - current_mean)/ current_std)*std + mean
    df.to_csv(output_path + '/mid_output.csv', index=False)

def data_preparation(output_path):
    df = pd.read_csv(output_path + '/mid_output.csv', names=['workers_1', 'workers_2', 'workers_3', 'workers_4'], header=None)
    name_list = ['workers_1', 'workers_2', 'workers_3', 'workers_4']
    
    for name in name_list:
        df[name] = pd.to_numeric(df[name], errors='coerce').dropna()
        data = df[name][2:]
        mean = data.mean()
        std = data.std()
        df[name] = df[name].fillna(value=np.random.normal(loc=mean, scale=std))
    
    num_col = df.select_dtypes(include=['number']).columns.tolist()
    cat_col = df.select_dtypes(include=['category', 'object']).columns.tolist()
    
    if cat_col:
        df = pd.get_dummies(df, columns=cat_col, dtype=np.int8)
    
    df.to_csv(output_path + '/mid_output.csv', index=False)

def find_loss(args, offset=0):
    value_list = np.array(args)
    index = tau_min
    min_loss = np.max(value_list) - np.min(value_list)
    current_path = os.getcwd()
    mean = []
    std = []
    for k in range(4):
        Name_list = ['Xiong_data.xlsx', 'Guo_data.xlsx', 'Dong_data.xlsx','Xu_data.xlsx']
        new_df = pd.read_excel(current_path + f'/{Name_list[k]}', sheet_name='Sheet2', header=None)
        mean.append(new_df.iat[1,0])
        std.append(new_df.iat[1,1])
    for i in range(tau_min, tau_max + 1):
        times = []
        for k in range(4):
            time = value_list[k]
            for j in range(tau_min, i):
                rand = np.random.normal(loc=mean[k], scale=std[k])
                time += rand if rand > 0 else -rand
            times.append(time)  # 加入偏移量
        loss = np.max(times) - np.min(times)
        if min_loss >= loss:
            min_loss = loss
            index = i
        rand = random.random()
        if rand <= offset:
            index = i
    return index

def tau_predicter(row, row_index):
    workers = [
        pd.to_numeric(row['workers_1'], errors='coerce'),
        pd.to_numeric(row['workers_2'], errors='coerce'),
        pd.to_numeric(row['workers_3'], errors='coerce'),
        pd.to_numeric(row['workers_4'], errors='coerce')
    ]
    offset = row_index * 0.009  # 偏移量可以根据需要调整
    return find_loss(workers, offset)

def target_append(output_path):
    df = pd.read_csv(output_path + '/mid_output.csv', names=['workers_1', 'workers_2', 'workers_3', 'workers_4'], header=None)
    df['tau'] = df.apply(lambda row: tau_predicter(row, row.name), axis=1)
    df.to_csv(output_path + '/final_output.csv', index=False)

def data_loading_sgd_lin(target_path):
    
    df = pd.read_csv(target_path + '/final_output.csv', names=['workers_1', 'workers_2', 'workers_3', 'workers_4', 'tau'], header=None)
    y_tar = 'tau'
    X = df.drop(columns=y_tar)
    X = X[3:]
    y = df[y_tar].values.reshape(-1, 1)
    y = y[3:]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_size, random_state=42)
    #ros = RandomOverSampler(random_state=42)
    #X_train_rs, y_train_rs = ros.fit_resample(X_train, y_train)
    
    return X_train, y_train, X_test, y_test

def update_training_set(X_train, y_train, new_X, new_y):
    X_train = np.vstack([X_train, new_X])
    y_train = np.vstack([y_train, new_y])
    X_train = X_train[1:]
    y_train = y_train[1:]
    return X_train, y_train

class CustomDataset(Dataset):
    def __init__(self, csv_file, is_train, train_size=0.8):
        df = pd.read_csv(csv_file, names=['workers_1', 'workers_2', 'workers_3', 'workers_4', 'tau'], header=None)
        self.data = df[2:]
        self.data.head(5)
        self.train = is_train

        if is_train:
            self.data = self.data.iloc[:int(len(self.data) * train_size) + 1]
        else:
            self.data = self.data.iloc[int(len(self.data) * train_size) + 1:]

    def add_data(self, new_data, new_label):
    # 确保 new_data 是一个 DataFrame
        print('Adding data...')
        if isinstance(new_data, pd.DataFrame):
        # 检查 new_label 的类型
            if isinstance(new_label, (list, pd.Series, torch.Tensor)):
            # 将 new_label 转换为 DataFrame
                new_label_df = pd.DataFrame(new_label, columns=['tau'])  # 假设 'tau' 是标签列的名称
            
            # 确保 new_data 和 new_label 的索引一致
                new_feature_df = new_data.reset_index(drop=True)
                new_label_df = new_label_df.reset_index(drop=True)
            
            # 合并新数据和标签
                new_data_df = pd.concat([new_feature_df, new_label_df], axis=1)

            # 将新数据添加到现有数据中
                self.data = pd.concat([self.data, new_data_df], ignore_index=True)
            else:
                raise ValueError("new_label should be a list, pandas Series, or Tensor.")
        else:
            raise ValueError("new_data should be a pandas DataFrame.")

        # 删除最先的一个特征数据
        if len(self.data) > 0:
            self.data = self.data.iloc[1:].reset_index(drop=True)  # 删除第一行并重置索引

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.data):
            print(f"Accessing index: {idx}, dataset size: {len(self.data)}")
            raise IndexError(f"Index {idx} out of bounds for dataset of size {len(self.data)}")
    
    # 使用 .iloc 明确按位置访问

        label = pd.to_numeric(self.data.iloc[idx, -1], errors='coerce') - 4
    
        features = pd.to_numeric(self.data.iloc[idx, :-1], errors='coerce').astype('float32')

        if pd.isna(label) or features.isna().any():
            raise ValueError(f"Invalid data at index {idx}: label={label}, features={features}")
        label = int(label)  # 这里仍然可以将 label 转换为 int
        return torch.tensor(features.values), torch.tensor(label)

def data_loading_net(target_path):
    batch_size = net_batch_size
    final_path = target_path + '/final_output.csv'
    train_dataset = CustomDataset(final_path, True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = CustomDataset(final_path, False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, test_dataloader, train_dataset
