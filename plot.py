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

def enhanced_visualization(results):
    plt.figure(figsize=(16, 10))
    
    plt.subplot(2, 3, 1)
    time_data = [results[method]['val_acc'] for method in results]
    plt.bar(results.keys(), time_data)
    plt.title('Validation Accuracy')
    plt.legend()
    
    plt.subplot(2, 3, 2)
    time_data = [results[method]['Avg_comp_time'] for method in results]
    plt.bar(results.keys(), time_data)
    plt.title('Average Epoch Time (s)')
    
    plt.subplot(2, 3, 3)
    comm_time = [results[method]['Avg_comm_time'] for method in results]
    plt.bar(results.keys(), comm_time)
    plt.title('Average Communication Time (s)')
    
    plt.subplot(2, 3, 4)
    data_transferred = [results[method]['data_transferred'] for method in results]
    plt.bar(results.keys(), data_transferred)
    plt.title('Data Transferred (MB)')
    
    plt.subplot(2, 3, 5)
    time_data = [results[method]['Avg_step_time'] for method in results]
    plt.bar(results.keys(), time_data)
    plt.title('Average Step Time (s)')
    plt.xlabel('Time per Epoch (s)')
    plt.ylabel('Final Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('comparison.png')
    plt.show()

method_list = ['sgd','Syn_sgd','Local_syn_sgd','Mom','A_sgd','Knn','SVM','Cluster','EnhancedCluster', 'Net']
val_acc = [ 0.515, 0.5943, 0.3453, 0.3362]
act = [0.737,0.141,0.218,0.244]
acm = [0.00, 0.26, 0.21, 0.13]
dt = [8.15, 3.623, 1.09, 1.09]
ast = [0.737, 0.401, 0.428, 0.245]

def append_result(index):
    results = {}
    methods = ['Vanilla SGD', 'Local SGD', 'Sync SGD', 'Federated']
    for i, name in enumerate(methods):
        results[name] = {
            'val_acc': val_acc[i],
            'Avg_comp_time': act[i],
            'Avg_comm_time': acm[i],
            'data_transferred': dt[i],
            'Avg_step_time': ast[i]
        }
    return results
    
def main():
    results = append_result(9)
    enhanced_visualization(results)


if __name__ == '__main__':
    main()