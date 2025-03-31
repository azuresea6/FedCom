import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import threading
import time
import copy
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import basic_method as bm
from functools import partial

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

name_list = ['Xiong', 'Guo', 'Dong', 'Xu']
user_index = 0
total_epoch = 60

class Tau_model_nom:
    def __init__(self, method_name, X_train_rs, y_train_rs):
        self.method_name = method_name
        self.X = X_train_rs.reshape(-1,4)
        self.y = y_train_rs.reshape(-1,1).ravel()  # 转为一维数组，解决 DataConversionWarning 
        _, n = self.X.shape
        self.model = None
        self.v = np.zeros(n)
        self.m_t = np.zeros(n)
        self.v_t = np.zeros(n)
        self.theta = np.zeros(n)

    def train(self):
        if self.method_name == 'sgd':
            for _ in range(10):
                self.theta = bm.tochastic_gradient_descent((self.X, self.y, self.theta, 0.1))
        elif self.method_name == 'Syn_sgd':
            self.theta, _ = bm.syn_sgd(self.X, self.y, 5, 0.1, 10, 4, 10, self.theta)
        elif self.method_name == 'Local_syn_sgd':
            self.theta, _ = bm.local_syn_sgd(self.X, self.y, self.theta, 5, 4, 4, 10, 0.1, 5)
        elif self.method_name == 'Newton':
            self.theta = bm.N_sgd(self.X, self.y, 5, 10, self.theta)
        elif self.method_name == 'Mom':
            self.theta, self.v = bm.M_sgd(self.X, self.y, 5, 0.1, 10, 0.9, self.v, self.theta)
        elif self.method_name == 'A_sgd':
            self.theta, self.m_t, self.v_t = bm.A_sgd(self.X, self.y, 5, 0.1, 10, 0.9, 0.999, self.theta, self.m_t, self.v_t)
        elif self.method_name == 'Knn':
            self.model = bm.KNN(self.X, self.y, 3)
        elif self.method_name == 'Dt':
            self.model = bm.DT(self.X, self.y, 10)
        elif self.method_name == 'Rf':
            self.model = bm.Random_Forest(5, 4, self.X, self.y)
        elif self.method_name == 'Svm':
            self.model = bm.Support_Vector_Machine(5, self.X, self.y)
        elif self.method_name == 'Lr':
            self.model = bm.Linear_regression(self.X, self.y)
        elif self.method_name == 'Rr':
            self.model = partial(bm.Ridge_regression, self.X, self.y)

    def predict(self, x):
        x = np.array(x).reshape(-1, 4)  # 输入强制转为二维
        name = self.method_name
        if name in ['sgd', 'Syn_sgd', 'Local_syn_sgd', 'Newton', 'Mom', 'A_sgd']:
            return np.dot(self.theta, x.T)
        elif name in ['Knn', 'Dt', 'Rf', 'Svm', 'Lr']:
            return self.model.predict(x)
        elif name == 'Rr':
            return self.model(x)
        
    def update_training_set(self, new_X, new_y):
        new_X = np.array(new_X).reshape(-1, 4)  # 新数据转为二维
        self.y = self.y.reshape(-1,1)
        new_y = np.array(new_y).reshape(-1,1) # 转为一维数组
        self.X = np.vstack([self.X, new_X])
        self.y = np.vstack([self.y, new_y]).ravel()  # 确保 new_y 是二维数组
        self.X = self.X[-10:]  # 保留最近10个样本
        self.y = self.y[-10:]

class GlobalSync:
    def __init__(self, num_workers):
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.worker_updates = []
        self.aggregation_counter = 0
        self.num_workers = num_workers
        self.global_params = None
        self.num_communications = 0  
        self.worker_current_time = []
        
class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*13*13, 10)
        )
    
    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)
    
global_model = MNISTModel()
global_learning_rate = 0.01
global_modeloptimizer = optim.SGD(global_model.parameters(), lr=global_learning_rate)

class Worker(threading.Thread):
    def __init__(self, worker_id, sync, train_loader, tau, tau_model):
        super().__init__()
        self.worker_id = worker_id
        self.sync = sync
        self.train_loader = train_loader
        self.total_steps = 0
        self.model = MNISTModel()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.timestamps = defaultdict(dict)
        self.tau = tau  
        self.tau_model = tau_model
        self.tau_train_times = []
        self.comm_time = 0

    def run(self):
        iter_loader = iter(self.train_loader)
        for _ in range(total_epoch):
            self.model.load_state_dict(global_model.state_dict())
            for step in range(self.tau):
                try:
                    data, target = next(iter_loader)
                except StopIteration:
                    iter_loader = iter(self.train_loader)
                    data, target = next(iter_loader)
                
                compute_start = time.perf_counter()
                outputs = self.model(data)
                loss = nn.CrossEntropyLoss()(outputs, target)
                self.optimizer.zero_grad()
                loss.backward()
                current_gradients = {name: param.grad.clone() for name, param in self.model.named_parameters()}
                self.gradient = current_gradients
                self.optimizer.step()
                compute_end = time.perf_counter()
                self.timestamps[step]['compute'] = compute_end - compute_start
                
                if (step + 1) % self.tau == 0:
                    comm_start = time.perf_counter()
                    with self.sync.lock:
                        self.sync.worker_updates.append((self.worker_id, self.gradient, step, self.timestamps[step]['compute']))
                        self.sync.aggregation_counter += 1
                        
                        while (self.sync.aggregation_counter % self.sync.num_workers != 0 or
                               any(item[2] != step for item in self.sync.worker_updates[-self.sync.num_workers:])):
                            self.sync.condition.wait()
                        
                        latest_updates = [item[1] for item in self.sync.worker_updates[-self.sync.num_workers:]]
                        latest_times = [item[3] for item in self.sync.worker_updates[-self.sync.num_workers:]]
                        avg_grad = {}
                        for key in latest_updates[0].keys():
                            avg_grad[key] = sum(grad[key] for grad in latest_updates) / len(latest_updates)
                        
                        for name, param in global_model.named_parameters():
                            param.data -= global_learning_rate * avg_grad[name]
                        self.model.load_state_dict(global_model.state_dict())
                        self.sync.num_communications += 1  
                        
                        # 输入数据强制转为二维
                        latest_times_2d = np.array(latest_times) if np.array(latest_times).size == 4 else np.full((1,4),latest_times[0])
                        self.tau = int(self.tau_model.predict(latest_times_2d)[0])  # 修复 DeprecationWarning 
                        self.tau = max(1, self.tau)  # 防止 tau 为 0 或负数
                        self.tau_model.update_training_set(latest_times_2d, np.array([self.tau]))
                        tau_train_start = time.perf_counter()
                        self.tau_model.train()
                        tau_train_time = time.perf_counter() - tau_train_start
                        self.tau_train_times.append(tau_train_time)
                        self.sync.condition.notify_all()
                    
                    comm_end = time.perf_counter()
                    self.timestamps[step]['comm'] = comm_end - comm_start - tau_train_time

def local_with_4w():
    num_workers = 4
    batch_size = 64
    data_path = './data'
    
    transform = transforms.Compose([transforms.ToTensor()])
    full_dataset = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    splits = [len(full_dataset) // num_workers] * num_workers
    subsets = random_split(full_dataset, splits)
    
    sync = GlobalSync(num_workers)
    dummy_model = MNISTModel()
    sync.global_params = copy.deepcopy(dummy_model.state_dict())
    
    X_train = np.array([[0.7, 1, 1.6, 1.5], [0.8, 1.1, 1.5, 1.6], [0.9, 1.2, 1.4, 1.7]])
    y_train = np.array([5, 6, 7])  
    y_train = y_train.ravel()  # 转为一维数组，解决 DataConversionWarning 
    tau_model = Tau_model_nom('Knn', X_train, y_train)
    tau_model.train()
    tau = int(tau_model.predict(np.array([0.9,1,1.1,1.7]).reshape(-1,4))[0])  # 修复 DeprecationWarning 

    workers = []
    for i in range(num_workers):
        train_loader = DataLoader(subsets[i], batch_size=batch_size, shuffle=True)
        worker = Worker(i, sync, train_loader, tau, tau_model)
        workers.append(worker)
    
    start_time = time.perf_counter()
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    total_time = time.perf_counter() - start_time
    
    compute_time_local = []
    comm_time_local = []
    for wid, worker in enumerate(workers):
        print(f"\nWorker {wid} Local时间轴:")
        for step in range(total_epoch * 10):
            if step in worker.timestamps:
                compute = worker.timestamps[step].get('compute', 0)
                comm = worker.timestamps[step].get('comm', 0)
                compute_time_local.append(compute)
                comm_time_local.append(comm)
                print(f"Step {step:2d} | 计算耗时: {compute:.4f} | 通信耗时: {comm:.4f}")
    
    plt.figure(figsize=(12, 6))
    print("The following is simulate the process for 4 workers")
    plt.subplot(1, 2, 1)
    plt.plot(compute_time_local, label='Local SGD计算时间', linestyle='--')
    plt.axhline(np.mean(compute_time_local), color='red', linestyle=':', label='Local SGD均值')
    plt.title('计算时间对比')
    plt.xlabel('训练步数')
    plt.ylabel('时间（秒）')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    comm_steps = []
    comm_durations = []
    for wid, worker in enumerate(workers):
        for step in worker.timestamps:
            if 'comm' in worker.timestamps[step]:
                comm_steps.append(step)
                comm_durations.append(worker.timestamps[step]['comm'])
    plt.scatter(comm_steps, comm_durations, c='red', label='通信事件')
    plt.plot(comm_steps, comm_durations, linestyle='--', alpha=0.5)
    plt.title('通信时间分布（tau={}）'.format(tau))
    plt.xlabel('触发通信的训练步数')
    plt.ylabel('通信耗时（秒）')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

    return np.mean(compute_time_local), np.std(compute_time_local)

def local_simulation():
    num_workers = 1
    batch_size = 64
    data_path = './data'
    
    transform = transforms.Compose([transforms.ToTensor()])
    full_dataset = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    splits = [len(full_dataset) // num_workers] * num_workers
    subsets = random_split(full_dataset, splits)
    
    sync = GlobalSync(num_workers)
    dummy_model = MNISTModel()
    sync.global_params = copy.deepcopy(dummy_model.state_dict())
    
    X_train = np.array([[0,1,2,3], [1,2,4,7], [2,3,6,9], [3,4,7,8]])  # 初始训练数据为二维
    y_train = np.array([[5], [5], [5], [5]])
    y_train = y_train.ravel()  # 转为一维数组，解决 DataConversionWarning 
    tau_model = Tau_model_nom('Knn', X_train, y_train)
    tau_model.train()
    tau = int(tau_model.predict(np.array([1,3,2,4]).reshape(-1,4))[0])  # 修复 DeprecationWarning 

    workers = []
    for i in range(num_workers):
        train_loader = DataLoader(subsets[i], batch_size=batch_size, shuffle=True)
        worker = Worker(i, sync, train_loader, tau, tau_model)
        workers.append(worker)
    
    start_time = time.perf_counter()
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    total_time = time.perf_counter() - start_time
    
    compute_time_local = []
    comm_time_local = []
    for wid, worker in enumerate(workers):
        print(f"\nWorker {wid} Local时间轴:")
        for step in range(total_epoch * 10):
            if step in worker.timestamps:
                compute = worker.timestamps[step].get('compute', 0)
                comm = worker.timestamps[step].get('comm', 0)
                compute_time_local.append(compute)
                comm_time_local.append(comm)
                print(f"Step {step:2d} | 计算耗时: {compute:.4f} | 通信耗时: {comm:.4f}")
    
    plt.figure(figsize=(12, 6))
    print("The following is simulate the process on the local computer")
    plt.subplot(1, 2, 1)
    plt.plot(compute_time_local, label='Local SGD计算时间', linestyle='--')
    plt.axhline(np.mean(compute_time_local), color='red', linestyle=':', label='Local SGD均值')
    plt.title('计算时间对比')
    plt.xlabel('训练步数')
    plt.ylabel('时间（秒）')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    comm_steps = []
    comm_durations = []
    for wid, worker in enumerate(workers):
        for step in worker.timestamps:
            if 'comm' in worker.timestamps[step]:
                comm_steps.append(step)
                comm_durations.append(worker.timestamps[step]['comm'])
    plt.scatter(comm_steps, comm_durations, c='red', label='通信事件')
    plt.plot(comm_steps, comm_durations, linestyle='--', alpha=0.5)
    plt.title('通信时间分布（tau={}）'.format(tau))
    plt.xlabel('触发通信的训练步数')
    plt.ylabel('通信耗时（秒）')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

    return compute_time_local, comm_durations


def main():
    ctl_mean, ctl_std = local_with_4w()
    ctl_list, cmd_list = local_simulation()
    file_name = f'{name_list[user_index]}_data.xlsx'
    data = {"mean": [ctl_mean], 'std': [ctl_std]}
    df = pd.DataFrame(data)
    data_2 = {f"worker_{user_index}":ctl_list}
    df_2 = pd.DataFrame(data_2)
    with pd.ExcelWriter(file_name, mode='w') as writer:
        df.to_excel(writer, sheet_name='Sheet2', index=False)
        df_2.to_excel(writer, sheet_name='Sheet1', index=False)

if __name__ == "__main__":
    main()