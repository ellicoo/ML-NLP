import pandas as pd
from IPython.core.display_functions import display
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

"""
第一步： 数据处理，空值，异常值处理
"""

pd.set_option('display.max_columns', 500)  # 设置显示的最大列数
import zipfile

with zipfile.ZipFile('KaggleCredit2.csv.zip', 'r') as z:  ##读取zip里的文件
    f = z.open('KaggleCredit2.csv')
    data = pd.read_csv(f, index_col=0)  # 使用pandas读取csv文件，将第一列作为索引
data.head()  # 显示头部5行数据
print(f"数据源的行列元祖：{data.shape}")  # (112915, 11) 返回一个元祖，第一个位置值为行数，第二个位置值为了列数
print(f"该元祖类型：{type(data.shape)}")

#
# Pandas保持了Numpy对关键字axis的用法，用法在Numpy库的词汇表当中有过解释：
# 轴用来为超过一维的数组定义的属性，二维数据拥有两个轴：第0轴沿着行的垂直往下，第1轴沿着列的方向水平延伸。
# axis=0：沿着行方向操作（即操作每列）。
# axis=1：沿着列方向操作（即操作每行）。
# 所以，axis=1 并不是指行，而是表示沿着列方向进行操作。
#
# 为了更清晰地理解这个概念，以下是 axis 参数在不同上下文中的含义：
#
# 删除操作（drop 方法）：
#
# axis=0：删除行。
# axis=1：删除列。
# 求和操作（sum 方法）：
#
# axis=0：沿着水平方向求和，即对每列求和。
# axis=1：沿着竖直方向求和，即对每行求和。

# 对于 Series：
#
# axis 参数一般没有影响，因为 Series 是一维的，但在一些函数中仍可以使用 axis=0 表示对 Series 本身进行操作。


# 返回的是一个 pandas.Series 对象，一种类似于一维数组的对象，它能存储任何数据类型（整数、字符串、浮点数、Python 对象等
null_detail_Series = data.isnull().sum(axis=0)  # 统计每列数据的的空值数量，本例子中age列空值4267个，NumberOfDependents列空值4267个
print(f"尝试统计各个列的空值数量：{type(null_detail_Series)}")
# data.isnull()这一部分会生成一个与 data 大小相同的 DataFrame，其中每个元素表示原 DataFrame 中相应位置上的值是否为空（NaN）。如果是 NaN，则对应位置为 True，否则为 False


# SeriousDlqin2yrs                           0
# RevolvingUtilizationOfUnsecuredLines       0
# age                                     4267
# NumberOfTime30-59DaysPastDueNotWorse       0
# DebtRatio                                  0
# MonthlyIncome                              0
# NumberOfOpenCreditLinesAndLoans            0
# NumberOfTimes90DaysLate                    0
# NumberRealEstateLoansOrLines               0
# NumberOfTime60-89DaysPastDueNotWorse       0
# NumberOfDependents                      4267

# 处理空值数据--本例子采用去掉为空的数据
data.dropna(inplace=True)  ##去掉为空的数据
# data.shape  # (108648, 11)，112915-4267=108648

print(f"空值处理后的数据集合的行列元祖：{data.shape}")

# SeriousDlqin2yrs: Person experienced 90 days past due delinquency or worse
# y 是目标变量，包含 SeriousDlqin2yrs 列中的所有值。
# X 是特征变量，包含 data DataFrame 中除 SeriousDlqin2yrs 列外的所有列。
Y = data['SeriousDlqin2yrs']
X = data.drop('SeriousDlqin2yrs', axis=1)

print(f"标签结果列Y的均值：{Y.mean()}")
# y.mean() ##求取均值

"""
第二步：分割数据集合：训练集合&测试集合
"""

from sklearn import model_selection

x_tran, x_test, y_tran, y_test = model_selection.train_test_split(X, Y, test_size=0.2)
print(f"测试集合的特征列行列元祖：{x_test.shape}")

"""
第三步：选择逻辑回归模型LR，参数指定多分类
"""

from sklearn.linear_model import LogisticRegression

## https://blog.csdn.net/sun_shengyun/article/details/53811483
lr = LogisticRegression(multi_class='ovr', solver='sag', class_weight='balanced')

"""
第四步：将抽象模型训练出具体的参数，返回一个具体的函数lr(带有参数的)
"""
lr.fit(x_tran, y_tran)

# 训练集合上的分数，模型的 score 指的是模型在数据集合上的预测准确率
# score = lr.score(x_tran, y_tran)
#
# print(f"训练集合上的分数：{score}")  ## 最好的分数是1
#
# # 测试集合上的分数
# score = lr.score(x_test, y_test)
# print(f"测试集合上的分数：{score}")  ## 最好的分数是1

"""
第五步：在测试集合上进行预测
"""
# 在测试集上进行预测
y_pred = lr.predict(x_test)

"""
第六步：模型评估
计算准确率、精确率、召回率和 F1 -- Precision、Recall、Accuracy、F1 score：
precision = tp / (tp + fp)
recall = tp / (tp + fn) 
accuracy = (tp + tn) / (tp + tn + fp + fn) 
f1_score = 2 * (precision * recall) / (precision + recall)

# Precision精确率: 表示在所有预测分类为正例中，有多少是真正的正例。---预测对的/所有预测为正的(对预测结果而言)
# Recall 召回率: 所有正样本中，所有被预测为正的有多少。--预测对的/所有正样本中(对样本而言)
# Accuracy 准确率: 即正确分类的样本数量与总样本数量之比 （有些需求不需要，可选择性忽略）
# F1 score :F1分数是Precision精确率和Recall召回率的调和平均值，用于综合考虑分类器的性能,特别是在类别不平衡的情况下

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


"""
第七步：将训练好的模型进行存储，以便下次直接使用而不需要重复训练

"""

import joblib
# 保存模型
# joblib.dump(lr, 'logistic_regression_model.pkl')
# joblib.dump(lr, './model_save/logistic_regression_model.pkl') # 指定文件位置
