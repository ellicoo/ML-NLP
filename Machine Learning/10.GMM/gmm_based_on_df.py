import pandas as pd
from sklearn.datasets import load_iris
from IPython.display import display

"""
# 与逻辑回归LR的对比：GMM 在处理复杂的、非线性的数据集时表现良好，因为它具有更大的灵活性，能够适应各种不同形状的类别。
# 它假设不知道这个数据集有啥类别，而通过gmm就知道啥数据表现出的啥类别--无监督学习

GMM（高斯混合模型）的默认分类是以整数形式给出的，每个类别都有一个整数编号。这些编号通常从0开始递增，
对应于模型中每个组件的索引。例如，如果你的GMM模型有3个组件，那么它会给出0、1、2三个整数作为类别标签。
在使用gmm.predict(X)方法时，它会根据模型对数据进行分类，并返回每个样本所属的类别标签，这些标签就是整数形式的类别编号
"""
# 加载鸢尾花数据集
iris = load_iris()

# 将数据集转换为DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
# 增加标签列，如果target列存在则被新数据覆盖
df['target'] = iris.target

# 显示DataFrame的前几行
display(df.head())

# 基于df数据进行模型训练
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 使用 PCA 将数据降维到二维，便于可视化
pca = PCA(n_components=2)
# X_pca = pca.fit_transform(df.iloc[:, :-1])  # 只选择特征列进行PCA降维
X_pca = pca.fit_transform(df.drop('target', axis=1))  # 选择特征列进行PCA降维

# 创建并拟合高斯混合模型
gmm = GaussianMixture(n_components=3, random_state=42)
# gmm.fit(df.iloc[:, :-1])  # 只选择特征列进行GMM拟合
gmm.fit(df.drop('target', axis=1))  # 选择特征列进行GMM拟合

# 获取每个样本的聚类标签
labels = gmm.predict(df.iloc[:, :-1])  # 只选择特征列进行预测

# 绘制聚类结果
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('GMM Clustering of Iris Dataset')
plt.colorbar(label='Cluster')
plt.show()
