import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

# 生成模拟数据

"""
make_blobs：
make_blobs函数是用来生成聚类数据的工具函数，用于创建聚类数据集。在这个代码中，
make_blobs函数生成了一个包含300个样本的数据集，这些样本被分成4个簇（即4个中心点，
每个簇的标准差为0.60。random_state=0表示设置随机种子为0，以确保结果的可重复性。

函数返回两个值：
X：生成的样本数据，是一个二维数组，每一行代表一个样本，每一列代表一个特征。
_：生成的样本的真实标签，但在这个例子中，由于我们不需要真实标签，因此将其赋值给了下划线 _，表示忽略这个返回值。
因此，X是包含300个样本的二维数组，而_则是忽略的真实标签。
"""
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

print(X)

# 创建并拟合高斯混合模型
gmm = GaussianMixture(n_components=4)
gmm.fit(X)

# 标记每个样本所属的聚类
labels = gmm.predict(X)

# 绘制结果
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')
plt.show()
