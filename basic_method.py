from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from multiprocessing import Pool
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ridge_regression

class Net(nn.Module):#简单的全连接神经网络
    def __init__(self, tau):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, tau)
        self.relu = nn.ReLU()

    def forward(self, x):
        num_samples = x.size(0)  # 获取样本数量
        valid_size = num_samples - (num_samples % 4)  # 计算可以被 4 整除的有效大小
        x = x[:valid_size]  # 丢弃多余的样本
        x = x.view(-1, 4)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def polynomial_loss(theta, X, y):#计算均方误差（MSE）损失。

    m = len(y)
    h = np.dot(X, theta)
    loss = np.sum((h - y) ** 2) / m
    return loss

def Base_sgd(args):#基础的随机梯度下降算法
    X, y, theta, learning_rate = args
    m = len(y)
    h = np.dot(X, theta)
    error = h - y
    gradient = np.dot(X.T, error) / m
    theta = theta - learning_rate * gradient
    return theta

def stochastic_gradient_descent(args):
    x, y, theta, learning_rate = args
    m = len(y)  # Number of training examples

    # 计算假设函数
    h = np.dot(x, theta)  # Hypothesis function

    # 计算误差并确保其形状为 (m, 1)
    error = h.reshape(-1, 1) - y.reshape(-1, 1)  # Error

    # 计算梯度
    gradient = np.dot(x.T, error) / m  # Calculate the gradient

    # 更新参数
    theta = theta - learning_rate * gradient.flatten()  # Update the parameters

    return theta

def syn_sgd(X, y, batch_size, learning_rate, num_iterations, num_workers, num_local_steps, theta):#同步 SGD，使用多进程并行计算梯度
    theta_history = [theta.copy()]
    for iter in range(num_iterations):
        with Pool(processes=num_workers) as pool:
            args_list = []
            for _ in range(num_workers):
                index = np.random.choice(len(X), batch_size // num_workers)
                X_batch = X[index, :]
                y_batch = y[index]
                args_list.append((X_batch, y_batch, theta, num_local_steps))
            grad_list = pool.map(stochastic_gradient_descent, args_list)
            theta = theta - learning_rate * np.mean(grad_list, axis=0)
            theta_history.append(theta.copy())
        if (iter + 1) % 10 == 0:
            loss = polynomial_loss(theta, X, y)
            print(f"Epoch {iter + 1}: Loss = {loss:.4f}")
    return theta, theta_history

def local_syn_sgd_helper(args):#
    X_batch, y_batch, theta, num_local_steps, tau, batch_size = args
    X = X_batch
    y = y_batch
    for _ in range(tau):
        theta = stochastic_gradient_descent((X_batch, y_batch, theta, num_local_steps))
    return theta

def local_syn_sgd(X, y, theta, num_local_steps, tau, num_workers, num_iterations, learning_rate, batch_size):
    m, n = X.shape
    theta = np.zeros(n)
    theta_history = [theta.copy()]
    for iter in range(num_iterations):
        with Pool(processes=num_workers) as pool:
            args_list = []
            for _ in range(num_workers):
                index = np.random.choice(len(X), batch_size // num_workers)
                X_batch = X[index, :]
                y_batch = y[index]
                args_list.append((X_batch, y_batch, theta, num_local_steps, tau, batch_size))
            grad_list = pool.map(local_syn_sgd_helper, args_list)
            theta = theta - learning_rate * np.mean(grad_list, axis=0)
            theta_history.append(theta.copy())
        if (iter + 1) % 10 == 0:
            loss = polynomial_loss(theta, X, y)
            print(f"Epoch {iter + 1}: Loss = {loss:.4f}")
    return theta, theta_history

def Momentum_sgd(X, y, batch_size, learning_rate, num_epochs, mom_rate, v, theta):#动量 SGD，引入动量加速收敛
    m, _ = X.shape
    theta_history = [theta.copy()]
    num_batches = m // batch_size
    for epoch in range(num_epochs):
        for batch in range(num_batches):
            start_index = batch * batch_size
            end_index = start_index + batch_size
            X_batch = X[start_index:end_index]
            y_batch = y[start_index:end_index]
            h_batch = np.dot(X_batch, theta)
            error_batch = h_batch - y_batch
            gradient = np.dot(X_batch.T, error_batch) / batch_size
            v = mom_rate * v + (1 - mom_rate) * gradient
            theta = theta - learning_rate * gradient
            theta_history.append(theta.copy())
            loss = polynomial_loss(theta, X, y)
    return theta, v

def Adam_sgd(X, y, batch_size, learning_rate, num_epochs, decay_rate_1, decay_rate_2, theta, m_t, v_t):#自适应 SGD，类似 Adam 优化器
    print(type(X), type(y), type(theta))
    m, _ = X.shape
    epsilon = 1e-8
    theta_history = [theta.copy()]
    t = 0
    num_batches = m // batch_size + (1 if m % batch_size != 0 else 0)
    for epoch in range(num_epochs):
        for batch in range(num_batches):
            start_index = batch * batch_size
            end_index = min(start_index + batch_size, m)
            X_batch = X[start_index:end_index]
            y_batch = y[start_index:end_index]
            h_batch = np.dot(X_batch, theta)
            error_batch = h_batch - y_batch
            gradient = np.dot(X_batch.T, error_batch) / (end_index - start_index)
            t += 1
            m_t = decay_rate_1 * m_t + (1 - decay_rate_1) * gradient
            v_t = decay_rate_2 * v_t + (1 - decay_rate_2) * gradient**2
            m_hat = m_t / (1 - decay_rate_1 ** t)
            v_hat = v_t / (1 - decay_rate_2 ** t)
            theta = theta - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
            theta_history.append(theta.copy())
            loss = polynomial_loss(theta, X, y)
    return theta, m_t, v_t

def Newton_sgd(X, y, batch_size, num_epochs, theta):#牛顿法 SGD，利用 Hessian 矩阵加速优化
    m, _ = X.shape
    print(X.shape)
    theta_history = [theta.copy()]
    num_batches = m // batch_size
    for epoch in range(num_epochs):
        for batch in range(num_batches):
            start_index = batch * batch_size
            end_index = start_index + batch_size
            X_batch = X[start_index:end_index]
            y_batch = y[start_index:end_index]
            h_batch = np.dot(X_batch, theta)
            error_batch = h_batch - y_batch
            gradient = np.dot(X_batch.T, error_batch) / batch_size
            H = np.dot(X_batch.T, X_batch) / batch_size
            theta = theta - np.linalg.inv(H + 1e-8 * np.eye(H.shape[0])).dot(gradient)
            theta_history.append(theta.copy())
            loss = polynomial_loss(theta, X, y)
    return theta, theta_history

def KNN(X, y, n):#K 最近邻分类器
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(X, y)
    return knn

def Decision_Tree(X_train, y_train, m_d):#决策树分类器
    model = DecisionTreeClassifier(max_depth=m_d)
    model.fit(X_train, y_train)
    return model

def Random_Forest(n_e, m_d, X_train, y_train):#随机森林分类器
    model = RandomForestClassifier(n_estimators=n_e, max_depth=m_d)
    model.fit(X_train, y_train)
    return model

def Support_Vector_Machine(C, X_train, y_train):#支持向量机分类器
    model = SVC(kernel='linear', C=C)
    model.fit(X_train, y_train)
    return model

def Linear_regression(X_train, y_train):#线性回归模型
    reg = LinearRegression().fit(X_train, y_train)
    return reg

def Ridge_regresssion(X_train, y_train, X_test):#岭回归模型
    coef, intercept = ridge_regression(X_train, y_train, alpha=1.0, return_intercept=True)
    y_pred = intercept + coef[0] * X_test
    return y_pred

def Network(tau):
    model = Net(tau)
    return model

def train_net(model, train_loader, learning_rate, num_epochs):#支持多 epoch 训练
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        for k, (datas, labels) in enumerate(train_loader):
            print(f"Batch {k}: Input size: {datas.size()}, Labels size: {labels.size()}")
            outputs = model(datas)
            labels = labels.long()
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model