import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, RandomSampler, Subset
import time
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from multiprocessing import Pool
from SGD_Simu import local_simulation_4w
import ULSGD_UP
import ULSGD

method_list = ['sgd','Syn_sgd','Local_syn_sgd','Mom','A_sgd','Knn','SVM','Cluster','EnhancedCluster', 'Net']
method_index = 0

# 实验配置
class Config:
    num_workers = 4
    batch_size = 64
    total_epochs = 5
    local_steps = 5
    data_path = './data'
    model_save_path = './model.pth'
    learning_rate = 0.01

# 增强型MNIST模型
class EnhancedMNISTModel(nn.Module):
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
        self.epoch_time = 0
        self.overall_time = 0
        self.step = 0
        
    def forward(self, x):
        return self.fc(self.conv(x))
    
    def record_metrics(self, phase, loss, acc):
        self.train_metrics[f'{phase}_loss'].append(loss)
        self.train_metrics[f'{phase}_acc'].append(acc)
        self.val_metrics[f'{phase}_loss'].append(loss)
        self.val_metrics[f'{phase}_acc'].append(acc)

# 通信统计模块
class CommunicationTracker:
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

# ================ 训练方法实现 ================
def vanilla_sgd(train_loader, test_loader):
    model = EnhancedMNISTModel()
    optimizer = optim.SGD(model.parameters(), lr=Config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    comm_tracker = CommunicationTracker()
    total_params = sum(p.numel() for p in model.parameters())
    model.step = Config.total_epochs
    for epoch in range(Config.total_epochs):
        epoch_start = time.time()
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for data, target in train_loader:
            outputs = model(data)
            loss = criterion(outputs, target)
            optimizer.zero_grad()
            loss.backward()
            
            # 模拟通信
            comm_start = time.time()
            comm_time = time.time() - comm_start
            comm_tracker.record_communication(comm_time, total_params)
            
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(target).sum().item()
            total += target.size(0)
        
        epoch_time = time.time() - epoch_start
        val_loss, val_acc = evaluate(model, test_loader)
        model.record_metrics('train', running_loss/len(train_loader), correct/total)
        model.record_metrics('val', val_loss, val_acc)
        model.train_metrics['epoch_time'].append(epoch_time)
        model.epoch_time += epoch_time
        model.overall_time += epoch_time
        
    return model, comm_tracker

def local_sgd(train_loader, test_loader):
    model = EnhancedMNISTModel()
    optimizer = optim.SGD(model.parameters(), lr=Config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    comm_tracker = CommunicationTracker()
    total_params = sum(p.numel() for p in model.parameters())
    local_params = copy.deepcopy(model.state_dict())
    model.step = Config.total_epochs*Config.local_steps
    for epoch in range(Config.total_epochs):
        epoch_start = time.time()
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        local_grads = None
        ct = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            model.load_state_dict(local_params)
            outputs = model(data)
            loss = criterion(outputs, target)
            optimizer.zero_grad()
            loss.backward()
            
            if local_grads is None:
                local_grads = {n: p.grad.clone() for n, p in model.named_parameters()}
            else:
                for n, p in model.named_parameters():
                    local_grads[n] += p.grad.clone()
            
            if (batch_idx + 1) % Config.local_steps == 0:
                comm_start = time.time()
                for n, p in model.named_parameters():
                    p.data -= Config.learning_rate * local_grads[n] / Config.local_steps
                comm_time = time.time() - comm_start
                ct = comm_time
                comm_tracker.record_communication(comm_time, total_params)
                local_params = copy.deepcopy(model.state_dict())
                local_grads = None
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(target).sum().item()
            total += target.size(0)
        
        epoch_time = time.time() - epoch_start
        model.epoch_time += epoch_time - ct
        model.overall_time += epoch_time
        val_loss, val_acc = evaluate(model, test_loader)
        model.record_metrics('train', running_loss/len(train_loader), correct/total)
        model.record_metrics('val', val_loss, val_acc)
        model.train_metrics['epoch_time'].append(epoch_time)
        
    return model, comm_tracker

def average_models(models):
    model_params = [model.state_dict() for model in models]
    averaged_params = {}
    for param_name in model_params[0]:
        params = torch.stack([model_params[i][param_name] for i in range(len(models))])
        averaged_params[param_name] = torch.mean(params, dim=0)
    return averaged_params

def duplicate_model(model, num_duplicates):
    return [copy.deepcopy(model) for _ in range(num_duplicates)]

def train_model(args):
    model, optimizer, criterion, sampler, dataset, batch_size = args
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    
    for data, target in train_loader:
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(target).sum().item()
        total += target.size(0)
    
    return model, total_loss, correct, total

def syn_sgd(train_dataset, test_dataset, batch_size, num_iterations):
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    global_model = EnhancedMNISTModel()
    comm_tracker = CommunicationTracker()
    total_params = sum(p.numel() for p in global_model.parameters())
    global_model.step = num_iterations
    for _ in range(num_iterations):
        epoch_start = time.time()
        models = duplicate_model(global_model, Config.num_workers)
        optimizers = [optim.SGD(model.parameters(), lr=Config.learning_rate) for model in models]
        samplers = [RandomSampler(train_dataset, replacement=True, num_samples=batch_size) for _ in range(Config.num_workers)]
        
        with Pool(processes=Config.num_workers) as pool:
            results = pool.map(train_model, [
                (models[i], optimizers[i], nn.CrossEntropyLoss(), samplers[i], train_dataset, batch_size)
                for i in range(Config.num_workers)
            ])
        
        comm_start = time.time()
        averaged_params = average_models([result[0] for result in results])
        global_model.load_state_dict(averaged_params)
        comm_time = time.time() - comm_start
        comm_tracker.record_communication(comm_time, total_params)
        
        epoch_time = time.time() - epoch_start
        global_model.epoch_time += epoch_time - comm_time
        global_model.overall_time += epoch_time
        total_loss = sum(result[1] for result in results) / Config.num_workers
        correct = sum(result[2] for result in results)
        total = sum(result[3] for result in results)
        val_loss, val_acc = evaluate(global_model, test_loader)
        
        global_model.record_metrics('train', total_loss/len(test_loader), correct/total)
        global_model.record_metrics('val', val_loss, val_acc)
        global_model.train_metrics['epoch_time'].append(epoch_time)
    
    return global_model, comm_tracker

def federated_training(train_dataset, test_dataset, batch_size, num_communications):
    # 修复数据集分割方式
    total_size = len(train_dataset)
    subset_size = total_size // Config.num_workers
    indices = list(range(total_size))
    np.random.shuffle(indices)
    train_subsets = [
        Subset(train_dataset, indices[i*subset_size : (i+1)*subset_size])
        for i in range(Config.num_workers)
    ]
    
    global_model = EnhancedMNISTModel()
    comm_tracker = CommunicationTracker()
    total_params = sum(p.numel() for p in global_model.parameters())
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    global_model.step = num_communications
    for comm in range(num_communications):
        epoch_start = time.time()
        models = duplicate_model(global_model, Config.num_workers)
        optimizers = [optim.SGD(model.parameters(), lr=Config.learning_rate) for model in models]
        samplers = [RandomSampler(subset, replacement=True, num_samples=batch_size) for subset in train_subsets]
        
        with Pool(processes=Config.num_workers) as pool:
            results = pool.map(train_model, [
                (models[i], optimizers[i], nn.CrossEntropyLoss(), samplers[i], train_subsets[i], batch_size)
                for i in range(Config.num_workers)
            ])
        
        comm_start = time.time()
        averaged_params = average_models([result[0] for result in results])
        global_model.load_state_dict(averaged_params)
        comm_time = time.time() - comm_start
        comm_tracker.record_communication(comm_time, total_params)
        
        epoch_time = time.time() - epoch_start
        global_model.epoch_time += epoch_time - comm_time
        global_model.overall_time += epoch_time
        total_loss = sum(result[1] for result in results) / Config.num_workers
        correct = sum(result[2] for result in results)
        total = sum(result[3] for result in results)
        val_loss, val_acc = evaluate(global_model, test_loader)
        
        global_model.record_metrics('train', total_loss/len(test_loader), correct/total)
        global_model.record_metrics('val', val_loss, val_acc)
        global_model.train_metrics['epoch_time'].append(epoch_time)
    
    return global_model, comm_tracker

# ================ 评估与可视化 ================
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

def enhanced_visualization(results):
    plt.figure(figsize=(18, 12))
    
    plt.subplot(2, 3, 1)
    time_data = [results[method]['val_acc'][-1] for method in results]
    plt.bar(results.keys(), time_data)
    plt.title('Validation Accuracy')
    plt.legend()
    
    plt.subplot(2, 3, 2)
    time_data = [results[method]['epoch_time'] for method in results]
    plt.bar(results.keys(), time_data)
    plt.title('Average Epoch Time (s)')
    
    plt.subplot(2, 3, 3)
    comm_time = [results[method]['comm_time']/results[method]['overall_step'] for method in results]
    plt.bar(results.keys(), comm_time)
    plt.title('Average Communication Time (s)')
    
    plt.subplot(2, 3, 4)
    data_transferred = [results[method]['data_transferred'] for method in results]
    plt.bar(results.keys(), data_transferred)
    plt.title('Data Transferred (MB)')
    
    plt.subplot(2, 3, 5)
    time_data = [results[method]['avg_step_time'] for method in results]
    plt.bar(results.keys(), time_data)
    plt.title('Average Step Time (s)')

    plt.subplot(2, 3, 6)
    time_data = [results[method]['overall_step'] for method in results]
    plt.bar(results.keys(), time_data)
    plt.title('Total steps')
    plt.xlabel('Time per Epoch (s)')
    plt.ylabel('Final Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('comparison.png')
    plt.show()

def generate_report(results):
    report = "=== Training Report ===\n"
    report += f"Total Epochs: {Config.total_epochs}\n"
    report += f"Batch Size: {Config.batch_size}\n"
    report += f"Learning Rate: {Config.learning_rate}\n\n"
    
    for method in results:
        report += f"Method: {method}\n"
        report += f"- Final Accuracy: {results[method]['val_acc'][-1]:.4f}\n"
        report += f"- Average Epoch Time: {results[method]['epoch_time']:.2f}s\n"
        report += f"- Total Comm Time: {results[method]['comm_time']:.2f}s\n"
        report += f"- Data Transferred: {results[method]['data_transferred']/1e6:.2f}MB\n"
        report += f"- Overall_step: {results[method]['overall_step']}\n"
        report += f"- Avg_step_time: {results[method]['avg_step_time']:.2f}s\n"
        report += f"- Avg_communication_time: {results[method]['comm_time']/results[method]['overall_step']:.2f}s"
    
    with open('training_report.txt', 'w') as f:
        f.write(report)
    print(report)

def main():
    transform = transforms.Compose([transforms.ToTensor()])
    full_dataset = datasets.MNIST(Config.data_path, train=True, download=True, transform=transform)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size)
    
    methods = {
        'Vanilla SGD': vanilla_sgd,
        'Local SGD': local_sgd,
        'Sync SGD': lambda tl, _: syn_sgd(train_dataset, test_dataset, Config.batch_size, Config.total_epochs),
        'Federated': lambda tl, _: federated_training(train_dataset, test_dataset, Config.batch_size, Config.total_epochs),
        #'local_simulation_4w': lambda tl, _: local_simulation_4w(),
        #'main_with_update': lambda tl, _:ULSGD_UP.local_with_4w(method_list[method_index]),
        #'main_without_update': lambda tl, _:ULSGD.local_with_4w(method_list[method_index]),
    }
    
    results = {}
    for name, func in methods.items():
        print(f"\nRunning {name}...")
        model, comm = func(train_loader, test_loader)
        results[name] = {
            'val_acc': model.val_metrics['val_acc'],
            'val_loss': model.val_metrics['val_loss'],
            'epoch_time': model.epoch_time / model.step,
            'comm_time': comm.comm_time,
            'data_transferred': comm.data_transferred,
            'overall_step': model.step,
            'avg_step_time': model.overall_time / model.step
        }
    
    max_epochs = Config.total_epochs
    for method in results:
        for key in ['val_acc', 'val_loss']:
            if len(results[method][key]) < max_epochs:
                results[method][key] += [0]*(max_epochs - len(results[method][key]))
    
    
    generate_report(results)
    enhanced_visualization(results)

if __name__ == '__main__':
    main()