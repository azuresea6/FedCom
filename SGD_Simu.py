import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import threading
import time
import copy
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.cluster import OPTICS
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from collections import defaultdict

# ====================== 全局配置 ======================
name_list = ['Xiong', 'Guo', 'Dong', 'Xu']
user_index = 0
total_epoch = 5  # 减少epoch数以便快速测试
DATA_RETENTION = 100
SMOOTHING_FACTOR = 0.3

# ====================== 异构环境模拟 ======================
class WorkerConfig:
    def __init__(self, worker_id):
        self.base_delay = random.uniform(0.1, 0.3)
        self.dynamic_factor = random.choice([0.8, 1.0, 1.2])
        self.batch_size = random.choice([64, 128])
        self.compute_intensity = random.randint(1, 2)
        print(f"Worker {worker_id} Config:")
        print(f"|- Batch: {self.batch_size}, Delay: {self.base_delay:.2f}s")

# ====================== 神经网络模型 ======================
class CommunicationTracker_1:
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


class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16*14*14, 10)
        )
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)

    def forward(self, x):
        return self.fc(self.conv(x))
    
    def record_metrics(self, phase, loss, acc):
        self.train_metrics[f'{phase}_loss'].append(loss)
        self.train_metrics[f'{phase}_acc'].append(acc)
        self.val_metrics[f'{phase}_loss'].append(loss)
        self.val_metrics[f'{phase}_acc'].append(acc)

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

# ====================== 增强型工作节点 ======================
class EnhancedWorker(threading.Thread):
    def __init__(self, worker_id, sync, train_data, test_data, config):
        super().__init__()
        self.worker_id = worker_id
        self.sync = sync
        self.config = config
        self.model = MNISTModel()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_data, batch_size=config.batch_size, shuffle=True)
        # 特征收集
        self.features = []
        self.labels = []
        self.step_times = []
        
    def _simulate_load(self):
        """模拟异构计算负载"""
        time.sleep(self.config.base_delay)
        if self.config.compute_intensity > 1:
            _ = np.random.rand(500,500) @ np.random.rand(500,500)

    def _collect_features(self, step_time):
        """收集四维特征：耗时/梯度/损失/参数变化"""
        grad_norms = [p.grad.norm().item() for p in self.model.parameters()]
        param_deltas = [
            (p.data - self.prev_params[name]).norm().item()
            for name, p in self.model.named_parameters()
        ]
        self.features.append([
            step_time,
            np.mean(grad_norms),
            self.current_loss.item(),
            np.mean(param_deltas)
        ])
        self.labels.append(self.sync.current_tau)

    def run(self):
        iter_loader = iter(self.train_loader)
        for epoch in range(total_epoch):
            self.prev_params = copy.deepcopy(self.model.state_dict())
            self.model.load_state_dict(self.sync.global_model.state_dict())
            
            for step in range(self.sync.tau):
                data, target = next(iter_loader)
                
                # 模拟负载
                self._simulate_load()
                
                # 正向传播
                start_time = time.perf_counter()
                outputs = self.model(data)
                self.current_loss = nn.CrossEntropyLoss()(outputs, target)
                _, predicted = outputs.max(1)
                correct = predicted.eq(target).sum().item()
                total = target.size(0)
                # 反向传播
                self.optimizer.zero_grad()
                self.current_loss.backward()
                self.optimizer.step()
                # 记录特征
                step_time = time.perf_counter() - start_time
                val_loss, val_acc = evaluate(self.model, self.test_loader)
                self.model.record_metrics('train', self.current_loss, correct/total)
                self.model.record_metrics('val', val_loss, val_acc)
                self.model.train_metrics['epoch_time'].append(step_time)
                with self.sync.lock:  # 线程安全收集
                    self._collect_features(step_time)
                    self.step_times.append(step_time)
                
                # 同步点
                if (step + 1) % self.sync.tau == 0:
                    self._sync_with_delay()

    def _sync_with_delay(self):
        """带网络延迟的同步"""
        with self.sync.lock:
            total_params = sum(p.numel() for p in self.model.parameters())
            comm_start = time.time()
            time.sleep(random.uniform(0.1, 0.3))  # 网络延迟
            self.sync.worker_data[self.worker_id] = {
                'features': self.features[-10:],  # 保留最近10个特征
                'labels': self.labels[-10:]
            }
            comm_time = time.time() - comm_start
            self.sync.comt.record_communication(comm_time, total_params)

# ====================== 全局同步与可视化 ======================
class GlobalSync:
    def __init__(self, num_workers, comt):
        self.lock = threading.Lock()
        self.global_model = MNISTModel()
        self.tau = 3  # 同步周期
        self.current_tau = self.tau
        self.worker_data = defaultdict(dict)
        self.scaler = StandardScaler()
        self.comt = comt
        
    def visualize_clusters(self):
        """聚合所有节点数据并可视化"""
        all_features = []
        all_labels = []
        
        # 线程安全数据聚合
        with self.lock:
            for wid in self.worker_data:
                if self.worker_data[wid]:
                    all_features.extend(self.worker_data[wid]['features'])
                    all_labels.extend(self.worker_data[wid]['labels'])
        
        if len(all_features) < 10:
            print("Not enough data for visualization")
            return
        
        # 数据标准化
        X = self.scaler.fit_transform(all_features)
        
        # t-SNE降维
        tsne = TSNE(n_components=2, perplexity=min(10, len(X)-1))
        X_2d = tsne.fit_transform(X)
        
        # 绘制聚类结果
        plt.figure(figsize=(10,6))
        scatter = plt.scatter(X_2d[:,0], X_2d[:,1], c=all_labels, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='Tau Value')
        plt.title("Dynamic Tau Clustering (t-SNE Projection)")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.grid(True)
        plt.show()

# ====================== 主执行流程 ======================
def local_simulation_4w():
    # 准备数据集
    transform = transforms.Compose([transforms.ToTensor()])
    full_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    num_workers = 4
    subsets = random_split(train_dataset, [len(train_dataset)//num_workers]*num_workers)
    # 初始化同步控制
    comm_tracker = CommunicationTracker_1()
    sync = GlobalSync(num_workers, comm_tracker)
    # 创建异构工作节点
    workers = []
    for i in range(num_workers):
        config = WorkerConfig(i)
        worker = EnhancedWorker(
            worker_id=i,
            sync=sync,
            train_data=subsets[i],
            test_data=test_dataset,
            config=config
        )
        workers.append(worker)
    
    # 启动训练
    for w in workers:
        w.start()
    
    
    # 最终可视化
    sync.visualize_clusters()
    print("Training completed")
    for i in range(len(workers[0].model.train_metrics["train_acc"])):
        train_acc=np.array([w.model.train_metrics["train_acc"][i] for w in workers]).sum() / num_workers 
        train_loss=np.array([w.model.train_metrics["train_loss"][i] for w in workers]).sum() / num_workers 
        val_acc=np.array([w.model.val_metrics["val_acc"][i] for w in workers]).sum() / num_workers 
        val_loss=np.array([w.model.val_metrics["val_loss"][i] for w in workers]).sum() / num_workers 
        epoch_time = np.array([w.model.train_metrics["epoch_time"][i] for w in workers]).sum() / num_workers
        sync.global_model.record_metrics('train', train_loss, train_acc)
        sync.global_model.record_metrics('val', val_loss, val_acc)
        sync.global_model.train_metrics['epoch_time'].append(epoch_time)
    return sync.global_model, sync.comt