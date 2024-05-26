import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

# 与逻辑回归LR的对比：GMM 在处理复杂的、非线性的数据集时表现良好，因为它具有更大的灵活性，能够适应各种不同形状的类别。
# 它假设不知道这个数据集有啥类别，而通过gmm就知道啥数据表现出的啥类别--无监督学习
# 加载鸢尾花数据集
iris = datasets.load_iris()

# X = iris.data 中的每一行数据都对应着 y = iris.target 中的一个类别标签
X = iris.data

"""
iris.target 是鸢尾花数据集中的目标变量，它记录了每个样本所属的类别（或类别标签）。
在进行高斯混合模型（GMM）的聚类时，通常不使用目标变量 y = iris.target，
因为 GMM 是一种无监督学习方法，它不需要任何关于类别标签的信息
"""

y = iris.target

print(f"尽管gmm进行无监督学习不需要任何标签类别信息，但是样本数据中原本的类别数据为：{y}")

# 使用 PCA 将数据降维到二维，便于可视化
pca = PCA(n_components=2)
# fit_transform函数用于对数据进行主成分分析，并返回转换后的数据集。
"""
具体而言，fit_transform函数执行了两个步骤：
拟合（Fit）： 将PCA模型适配到数据集上，计算出数据集的主成分分析结果。
转换（Transform）： 使用拟合好的PCA模型对数据进行变换，将数据映射到主成分空间中。

"""
X_pca = pca.fit_transform(X)  # 只是画图降低维度，但是gmm本身不降低维度

# 创建并拟合高斯混合模型
gmm = GaussianMixture(n_components=3, random_state=42)

# 原始数据的的表现
print(f"原始数据集：{X}")
print(f"原始数据的样本个数：{len(X)}")

# 找到具体函数并返回给gmm
gmm.fit(X)

# 获取每个样本的聚类标签
labels = gmm.predict(X)
print(f"gmm模型进行预测数据的结果：{labels}")
print(f"预测个数：{len(labels)}")

# 绘制聚类结果
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('GMM Clustering of Iris Dataset')
plt.colorbar(label='Cluster')
plt.show()
