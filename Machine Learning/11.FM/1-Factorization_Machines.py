"""
因子分解机 (Factorization Machines, FM)
因子分解机是由Steffen Rendle在2010年提出的一种机器学习算法。它的主要目标是解决在稀疏数据下特征之间的交互问题。

原理：FM 模型可以看作是线性模型和矩阵分解的结合。它能够有效地捕捉特征之间的二阶交互作用
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xlearn as xl

# 保存libsvm格式的数据集
data = """
0 1:0.5 2:1.3 3:0.7
1 1:1.2 2:0.1 3:2.4
0 1:0.8 2:0.9 3:0.5
1 1:0.4 2:1.5 3:0.9
"""
with open("data.libsvm", "w") as file:
    file.write(data)

# 加载数据
X, y = load_svmlight_file("data.libsvm")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 将训练数据保存到libsvm格式文件中
train_data = "train_data.libsvm"
test_data = "test_data.libsvm"
with open(train_data, "w") as f_train, open(test_data, "w") as f_test:
    for i in range(len(X_train)):
        f_train.write(f"{y_train[i]} " + " ".join([f"{j}:{X_train[i,j]}" for j in range(X_train.shape[1])]) + "\n")
    for i in range(len(X_test)):
        f_test.write(f"{y_test[i]} " + " ".join([f"{j}:{X_test[i,j]}" for j in range(X_test.shape[1])]) + "\n")

# 创建FM模型
fm_model = xl.create_fm()

# 设置训练参数
param = {'task': 'reg', 'lr': 0.2, 'k': 10, 'epoch': 10, 'lambda': 0.001, 'metric': 'rmse'}

# 训练FM模型
fm_model.fit(param, train_data, test_data)

# 预测
y_pred = fm_model.predict(test_data)

# 评估
mse = mean_squared_error(y_test, y_pred)
print(f"FM Model Mean Squared Error: {mse}")

# 清理临时文件
import os
os.remove(train_data)
os.remove(test_data)
