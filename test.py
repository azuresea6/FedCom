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
import ULSGD_UP as UP
import ULSGD as U
from SGD_Simu import local_simulation_4w

def enhanced_visualization(results):
    plt.figure(figsize=(18, 12))
    
    plt.subplot(2, 3, 1)
    for method in results:
        plt.plot(results[method]['val_acc'], label=method)
    plt.title('Validation Accuracy')
    plt.legend()
    
    plt.subplot(2, 3, 2)
    for method in results:
        plt.plot(results[method]['val_loss'], label=method)
    plt.title('Validation Loss')
    plt.legend()
    
    plt.subplot(2, 3, 3)
    time_data = [np.mean(results[method]['epoch_time']) for method in results]
    plt.bar(results.keys(), time_data)
    plt.title('Average Epoch Time (s)')
    
    plt.subplot(2, 3, 4)
    comm_time = [results[method]['comm_time'] for method in results]
    plt.bar(results.keys(), comm_time)
    plt.title('Total Communication Time (s)')
    
    plt.subplot(2, 3, 5)
    data_transferred = [results[method]['data_transferred']/1e6 for method in results]
    plt.bar(results.keys(), data_transferred)
    plt.title('Data Transferred (MB)')
    
    plt.subplot(2, 3, 6)
    for method in results:
        plt.scatter(np.mean(results[method]['epoch_time']), 
                    results[method]['val_acc'][-1], label=method)
    plt.xlabel('Time per Epoch (s)')
    plt.ylabel('Final Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('comparison.png')
    plt.show()


class Config:
    num_workers = 4
    batch_size = 64
    total_epochs = 5
    local_steps = 5
    data_path = './data'
    model_save_path = './model.pth'
    learning_rate = 0.01

def generate_report(method_results):
    report = "=== Training Report ===\n"
    report += f"Total Epochs: {Config.total_epochs}\n"
    report += f"Batch Size: {Config.batch_size}\n"
    report += f"Learning Rate: {Config.learning_rate}\n\n"
    
    # 假设 method_name 是你要报告的方法名称
    method_name = "YourMethodName"  # 替换为实际的方法名称
    report += f"Method: {method_name}\n"
    report += f"- Final Accuracy: {method_results['val_acc'][-1]:.4f}\n"
    report += f"- Average computation Time: {np.mean(method_results['epoch_time']):.2f}s\n"
    report += f"- Total Comm Time: {method_results['comm_time']:.2f}s\n"
    report += f"- Data Transferred: {method_results['data_transferred']/1e6:.2f}MB\n"
    report += f"- Overall_step: {method_results['overall_step']}\n"
    report += f"- Avg_step_time: {method_results['avg_step_time']:.2f}s\n"
    report += f"- Avg_communication_time: {method_results['comm_time']/method_results['overall_step']:.2f}s"
    
    with open('training_report.txt', 'w') as f:
        f.write(report)
    print(report)

def main():
    current_path = os.getcwd()
    X_train, y_train, x_test, y_test = dd.data_loading_sgd_lin(current_path)
    model, comm = U.local_with_4w('Net')
    overall_step = model.over
    avg_step_time = np.sum(np.array(comm.overall_time).reshape(-1,1)) / (4*overall_step)

# 示例结果构造
    method_results = {
    'val_acc': model.val_metrics['val_acc'],
    'val_loss': model.val_metrics['val_loss'],
    'epoch_time': model.train_metrics['epoch_time'],
    'comm_time': comm.comm_time,
    'data_transferred': comm.data_transferred,
    'overall_step': overall_step,
    'avg_step_time': avg_step_time
}

# 调用生成报告函数
    generate_report(method_results)

if __name__ == '__main__':
    main()