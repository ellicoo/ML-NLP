from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score

# 加载数据集
iris = load_iris()
X = iris.data
y = iris.target

# 将数据集拆分为训练集和测试集
# random_state参数用于设置随机数种子，它控制了数据集的随机划分方式

"""
当 random_state 参数设置为一个固定的整数值时，每次运行代码时，
train_test_split 函数将以相同的方式随机划分数据集，这样就能够得到相同的训练集和测试集，保证了结果的可复现性。
如果不设置 random_state 参数，每次运行代码时，train_test_split 函数都会使用不同的随机数种子，从而得到不同的数据划分结果

"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""
 lbfgs 求解器在迭代次数达到限制时未收敛。这种情况下，可能需要增加最大迭代次数 max_iter
  或者对数据进行缩放。根据警告提示的链接查看详细信息，并尝试调整参数。
"""
# 创建并训练逻辑回归模型，增加最大迭代次数
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = lr.predict(X_test)

print(y_pred)

# 计算准确率、精确率、召回率和 F1 分数
"""
计算 Precision、Recall、Accuracy、F1 score：
precision = tp / (tp + fp)
recall = tp / (tp + fn)
accuracy = (tp + tn) / (tp + tn + fp + fn) 即正确分类的样本数量与总样本数量之比
f1_score = 2 * (precision * recall) / (precision + recall)
"""

"""
average='macro'通常用于处理多类别分类问题中各类别样本数量不平衡的情况，
它对每个类别的影响权重是相同的，不考虑各类别样本数量的差异。

举个例子，假设一个多类别分类问题有三个类别 A、B、C，
对应的精确率分别为 0.8、0.7、0.6，如果采用 average='macro'，
那么计算得到的平均精确率为 (0.8 + 0.7 + 0.6) / 3 = 0.7

但准确率没有这个参数

"""
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
