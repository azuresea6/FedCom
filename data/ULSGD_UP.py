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
from sklearn.cluster import KMeans, OPTICS
from sklearn.svm import SVR
import data_dealer as dd
import os

tau_max = 10
tau_min = 1
batch_size = 24
name_list = ['Xiong', 'Guo', 'Dong', 'Xu']
user_index = 0
total_epoch = 2

def Time_simulation(i):
    current_path = os.getcwd()
    Name_list = ['Xiong_data.xlsx', 'Guo_data.xlsx', 'Dong_data.xlsx','Xu_data.xlsx']
    new_df = pd.read_excel(current_path + f'/{Name_list[i]}', sheet_name='Sheet2', header=None)
    mean = new_df.iat[1,0]
    std = new_df.iat[1,1]
    rand = np.random.normal(loc=mean, scale=std)
    rand = rand if rand > 0 else  -rand
    time.sleep(rand)

class EnhancedCluster:
    """支持多维输入的自适应聚类算法"""
    def __init__(self):
        self.model = OPTICS(min_samples=2, cluster_method='xi')
        
    def fit_predict(self, X):
        self.model.fit(X)
        labels = self.model.labels_
        return labels + 1 if -1 not in labels else labels  # 处理噪声样本
    
class CommunicationTracker_2:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.comm_time = 0.0
        self.data_transferred = 0  # bytes
        self.comm_events = 0
        
    def record_communication(self, comm_time, param_size):
        self.comm_time += comm_time
        self.data_transferred += param_size * 4  # float32=4bytes
        self.comm_events += 1

class Tau_model_nom:
    def __init__(self, method_name, X_train_rs, y_train_rs):
        self.method_name = method_name
        self.X = np.array(X_train_rs).reshape(-1, 4) if not isinstance(X_train_rs, np.ndarray) else X_train_rs
        self.y = np.array(y_train_rs).flatten()
        m, n = self.X.shape
        self.model = None
        self.cluster_model = None
        self.v = np.zeros(n)
        self.m_t = np.zeros(n)
        self.v_t = np.zeros(n)
        self.theta = np.zeros(n)
        self.tau_history = []
        current_path = os.getcwd()
        train_dataloader, test_dataloader, train_dataset = dd.data_loading_net(current_path)
        self.train_dataloader = train_dataloader
        self.train_dataset = train_dataset

    def update_train_dataset(self, new_X, new_y):
        X = np.array(new_X).reshape(-1, 4)  # 将 new_X 转换为 2D 数组，并重塑为 (-1, 4)
        X = pd.DataFrame(X, columns=['workers_1', 'workers_2', 'workers_3', 'workers_4'])  # 添加列名

        y = np.array(new_y).flatten()  # 将 new_y 转换为 1D 数组
        y = pd.Series(y)  # 转换为 Series

# 添加数据到训练数据集中
        self.train_dataset.add_data(X, y)   
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    def train(self):
        if self.method_name == 'sgd':
            for _ in range(10):
                self.theta = bm.stochastic_gradient_descent((self.X, self.y, self.theta, 0.1))
        elif self.method_name == 'Syn_sgd':
            self.theta, _ = bm.syn_sgd(self.X, self.y, 5, 0.1, 10, 4, 10, self.theta)
        elif self.method_name == 'Local_syn_sgd':
            self.theta, _ = bm.local_syn_sgd(self.X, self.y, self.theta, 5, 4, 4, 10, 0.1, 5)
        elif self.method_name == 'Newton':
            self.theta = bm.Newton_sgd(self.X, self.y, 5, 10, self.theta)
        elif self.method_name == 'Mom':
            self.theta, self.v = bm.Momentum_sgd(self.X, self.y, 5, 0.1, 10, 0.01, self.v, self.theta)
        elif self.method_name == 'A_sgd':
            self.theta, self.m_t, self.v_t = bm.Adam_sgd(self.X, self.y, 5, 0.1, 5, 0.1, 0.1, self.theta, self.m_t, self.v_t)
        elif self.method_name == 'Knn':
            self.model = bm.KNN(self.X, self.y, tau_max - tau_min)
        elif self.method_name == 'SVM':
            self.model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
            self.model.fit(self.X, self.y)
        elif self.method_name == 'Cluster':
            self.cluster_model = EnhancedCluster()
            self.cluster_model.fit_predict(self.X)
        elif self.method_name == 'EnhancedCluster':
            self.cluster_model = EnhancedCluster()
            self.cluster_model.fit_predict(self.X.reshape(-1, 1))
        elif self.method_name == 'Net':
            self.model = bm.Net(tau_max + 1 - tau_min)
            self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            bm.train_net(self.model, self.train_dataloader, 0.1, 5)

    def predict(self, x):
        x = np.array(x).flatten()
        try:
            print(f"Method: {self.method_name}, Input: {x}")
            if self.method_name == 'Knn':
                return int(self.model.predict([x])[0])
            elif self.method_name == 'SVM':
                return max(1, int(self.model.predict([x])[0]))
            elif self.method_name in ['Cluster', 'EnhancedCluster']:
                return int(self.cluster_model.fit_predict([x])[0])
            elif self.method_name == 'Net':
                data = [x] * 24 
                data_np = np.array(data)
                data_tensor = torch.tensor(data_np, dtype=torch.float32)
                output = self.model(data_tensor)
                _, predicted = torch.max(output.data, 1)
                print("Type of predicted[0]:", type(predicted[0]))
                print("Value of predicted[0]:", predicted[0])
                label_array = torch.linspace(tau_min, tau_max, steps=tau_max + 1 - tau_min)
                return predicted[0]
            else:
                print(x.shape)
                return max(1, int(np.dot(self.theta, x)))
        except Exception as e:
            print(f"Prediction error: {e}, Type of error: {type(e)}")
            return 1

    def update_training_set(self, new_X, new_y):
        if self.method_name == 'Net':
            self.update_train_dataset(new_X, new_y)
            return 
        new_X = np.array(new_X).reshape(-1, 4)
        new_y = np.array(new_y).flatten()
        self.X = np.vstack([self.X, new_X])
        self.y = np.concatenate([self.y, new_y])
        if len(self.X) > 50:
            self.X = self.X[-50:]
            self.y = self.y[-50:]

    def track_tau(self):
        """可视化tau调整历史"""
        plt.figure(figsize=(10,4))
        plt.plot(self.tau_history, marker='o', linestyle='--')
        plt.title(f"Tau Adjustment History ({self.method_name})")
        plt.xlabel("Communication Round")
        plt.ylabel("Tau Value")
        plt.grid(True)
        plt.show()

class GlobalSync:
    def __init__(self, num_workers, comt):
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.worker_updates = []
        self.aggregation_counter = 0
        self.num_workers = num_workers
        self.global_params = None
        self.num_communications = 0
        self.comt = comt
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
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)
    
    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)
    
    def record_metrics(self, phase, loss, acc):
        self.train_metrics[f'{phase}_loss'].append(loss)
        self.train_metrics[f'{phase}_acc'].append(acc)
        self.val_metrics[f'{phase}_loss'].append(loss)
        self.val_metrics[f'{phase}_acc'].append(acc)
    
    
global_model = MNISTModel()
global_learning_rate = 0.01
global_modeloptimizer = optim.SGD(global_model.parameters(), lr=global_learning_rate)

class Worker(threading.Thread):
    def __init__(self, worker_id, sync, train_loader, tau, tau_model, test_loader):
        super().__init__()
        self.worker_id = worker_id
        self.sync = sync
        self.train_loader = train_loader
        self.model = MNISTModel()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.timestamps = defaultdict(dict)
        self.tau = tau
        self.tau_model = tau_model
        self.tau_train_times = []
        self.comm_time = 0
        self.test_loader = test_loader
        self.tau_history = []

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
                _, predicted = outputs.max(1)
                correct = predicted.eq(target).sum().item()
                total = target.size(0)
                self.optimizer.zero_grad()
                loss.backward()
                current_gradients = {name: param.grad.clone() for name, param in self.model.named_parameters()}
                self.gradient = current_gradients
                self.optimizer.step()
                Time_simulation(self.worker_id)
                compute_end = time.perf_counter()
                self.timestamps[step]['compute'] = compute_end - compute_start
                val_loss, val_acc = evaluate(self.model, self.test_loader)
                self.model.record_metrics('train', loss, correct/total)
                self.model.record_metrics('val', val_loss, val_acc)
                self.model.train_metrics['epoch_time'].append(compute_end - compute_start)
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
                        
                        latest_times_2d = np.array(latest_times).reshape(-1, 1)
                        predicted_tau = self.tau_model.predict(latest_times_2d)
                        self.tau = max(1, int(predicted_tau))
                        self.tau_history.append(self.tau)
                        self.tau_model.update_training_set(np.array(latest_times_2d).reshape(-1,4), np.array([self.tau]))
                        self.tau_model.train()
                        avg_grad = {}
                        for key in latest_updates[0].keys():
                            avg_grad[key] = sum(grad[key] for grad in latest_updates) / len(latest_updates)
                        for name, param in global_model.named_parameters():
                            param.data -= global_learning_rate * avg_grad[name]
                        self.model.load_state_dict(global_model.state_dict())
                        self.sync.num_communications += 1
                        self.sync.condition.notify_all()
                    
                    comm_end = time.perf_counter()
                    self.timestamps[step]['comm'] = comm_end - comm_start
                    total_params = sum(p.numel() for p in global_model.parameters())
                    self.sync.comt.record_communication(comm_end - comm_start, total_params)

def compare_methods():
    """不同方法的预测对比"""
    methods = ['Knn', 'SVM', 'EnhancedCluster']
    test_data = np.array([[0.2], [0.4], [0.6], [0.8]])
    results = {}

    for method in methods:
        X_train = np.array([[0.1], [0.3], [0.5], [0.7]])
        y_train = np.array([3, 5, 7, 5])
        
        model = Tau_model_nom(method, X_train, y_train)
        model.train()
        predictions = [model.predict(x) for x in test_data]
        results[method] = predictions
        model.track_tau()

    plt.figure(figsize=(12,6))
    for method, vals in results.items():
        plt.plot(vals, label=method, marker='o', linestyle='--')
    plt.title("Prediction Comparison Between Methods")
    plt.xlabel("Test Sample Index")
    plt.ylabel("Predicted Tau Value")
    plt.xticks(range(len(test_data)), [f"Sample {i+1}" for i in range(len(test_data))])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def local_with_4w(method='SVM'):
    num_workers = 4
    data_path = './data'
    
    transform = transforms.Compose([transforms.ToTensor()])
    full_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    subsets = random_split(train_dataset, [len(train_dataset)//num_workers]*num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    comm_tracker = CommunicationTracker_2()
    sync = GlobalSync(num_workers, comm_tracker)
    dummy_model = MNISTModel()
    sync.global_params = copy.deepcopy(dummy_model.state_dict())
    current_path = os.getcwd()
    X_train, y_train, x_test, y_test = dd.data_loading_sgd_lin(current_path)
    tau_model = Tau_model_nom(method, X_train.astype(float), y_train.astype(float))
    tau_model.train()
    tau = tau_model.predict(np.array(X_train.astype(float).iloc[0, :]).reshape(-1,4))
    
    workers = []
    for i in range(num_workers):
        train_loader = DataLoader(subsets[i], batch_size=batch_size, shuffle=True)
        worker = Worker(i, sync, train_loader, tau, tau_model,test_loader)
        workers.append(worker)
    
    start_time = time.perf_counter()
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    total_time = time.perf_counter() - start_time
    
    plt.figure(figsize=(15,5))
    plt.subplot(1, 2, 1)
    compute_times = []
    for worker in workers:
        compute_times.extend([ts['compute'] for ts in worker.timestamps.values()])
    plt.hist(compute_times, bins=20, alpha=0.7)
    plt.title("Computation Time Distribution")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency")
    
    plt.subplot(1, 2, 2)
    comm_steps = []
    for worker in workers:
        comm_steps.extend([step for step, ts in worker.timestamps.items() if 'comm' in ts])
    plt.hist(comm_steps, bins=20, color='orange', alpha=0.7)
    plt.title("Communication Event Distribution")
    plt.xlabel("Training Step")
    plt.ylabel("Frequency")
    
    plt.tight_layout()
    plt.show()
    
    for i, worker in enumerate(workers):
        plt.figure(figsize=(10,4))
        plt.plot(worker.tau_history, marker='o')
        plt.title(f"Worker {i} Tau Adjustment History")
        plt.xlabel("Communication Round")
        plt.ylabel("Tau Value")
        plt.grid(True)
        plt.show()
    
    for i in range(len(workers[0].model.train_metrics["train_acc"])):
        train_acc=np.array([w.model.train_metrics["train_acc"][i] for w in workers]).sum() / num_workers 
        train_loss = np.array([w.model.train_metrics["train_loss"][i].detach().numpy() for w in workers]).sum() / num_workers
        val_acc=np.array([w.model.val_metrics["val_acc"][i] for w in workers]).sum() / num_workers 
        val_loss=np.array([w.model.val_metrics["val_loss"][i] for w in workers]).sum() / num_workers 
        epoch_time = np.array([w.model.train_metrics["epoch_time"][i] for w in workers]).sum() / num_workers
        global_model.record_metrics('train', train_loss, train_acc)
        global_model.record_metrics('val', val_loss, val_acc)
        global_model.train_metrics['epoch_time'].append(epoch_time)

    return global_model, sync.comt

def evaluate(model, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    
    with torch.no_grad():
        for data, target in test_loader:
            outputs = model(data)
            loss = criterion(outputs, target)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(target).sum().item()
            total += target.size(0)
    
    return total_loss/len(test_loader), correct/total