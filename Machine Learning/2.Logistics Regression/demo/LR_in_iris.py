from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score


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
X = iris.data
y = iris.target

# 将数据集拆分为训练集和测试集
# random_state参数用于设置随机数种子，它控制了数据集的随机划分方式，确定它就可以使得每次跑都得到一样的结果

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

"""
max_iter是逻辑回归模型的一个超参数，它指定了模型训练的最大迭代次数。在优化算法（如梯度下降）中，
模型会根据训练数据逐步调整参数，直到达到最优解或者迭代次数达到了设定的最大值。

如果模型在指定的迭代次数内没有收敛（即参数没有收敛到最优解），训练过程会提前结束，
并且会发出警告或者报错。这时，可以尝试增加 max_iter 参数的值，以允许模型更多的迭代次数，从而有更多的机会找到最优解。

需要注意的是，增加 max_iter 可能会增加模型训练的时间，特别是在数据量较大、特征较多的情况下。
因此，需要根据实际情况和计算资源来调整 max_iter 的值
"""
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
