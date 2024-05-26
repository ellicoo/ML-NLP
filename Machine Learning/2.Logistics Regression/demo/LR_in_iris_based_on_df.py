import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

"""
iris鸢尾花数据集合的介绍：
鸢尾花数据集（Iris dataset）是机器学习和统计学中最常用的示例数据集之一。
它是一个多变量数据集，由英国统计学家和生物学家 Ronald Fisher 
于 1936 年在他的论文 “The use of multiple measurements in taxonomic problems” 中首次引入。
这个数据集经常用于分类算法的演示和验证。以下是对鸢尾花数据集的详细介绍：

数据集结构
鸢尾花数据集包含 150 个样本，每个样本对应一朵鸢尾花。每个样本有 4 个特征和 1 个目标变量。
数据集中的鸢尾花属于三个不同的品种，每个品种有 50 个样本。

特征：
鸢尾花数据集中的特征都是数值型的，描述了鸢尾花的形态学特征：
    花萼长度（Sepal Length）：单位是厘米。
    花萼宽度（Sepal Width）：单位是厘米。
    花瓣长度（Petal Length）：单位是厘米。
    花瓣宽度（Petal Width）：单位是厘米。

目标变量：
    目标变量表示鸢尾花的品种，分别是：
        Setosa：山鸢尾
        Versicolor：变色鸢尾
        Virginica：维吉尼亚鸢尾
在数据集中，这些品种通常被编码为整数值：
        0 表示 Setosa
        1 表示 Versicolor
        2 表示 Virginica

"""

# 加载数据集
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(df.drop('target', axis=1), df['target'], test_size=0.2, random_state=42)

# 创建并训练逻辑回归模型
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = lr.predict(X_test)

"""
计算 Precision、Recall、Accuracy、F1 score：
precision = tp / (tp + fp)
recall = tp / (tp + fn)
accuracy = (tp + tn) / (tp + tn + fp + fn) 即正确分类的样本数量与总样本数量之比
f1_score = 2 * (precision * recall) / (precision + recall)
"""

# 计算准确率、精确率、召回率和 F1 分数
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
