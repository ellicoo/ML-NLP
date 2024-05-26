import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# 加载数据
pd.set_option('display.max_columns', 500)  # 设置显示的最大列数
import zipfile

with zipfile.ZipFile('KaggleCredit2.csv.zip', 'r') as z:  ##读取zip里的文件
    f = z.open('KaggleCredit2.csv')
    data = pd.read_csv(f, index_col=0)  # 使用pandas读取csv文件，将第一列作为索引
data.head()  # 显示头部5行数据
print(f"数据源的行列元祖：{data.shape}")  # (112915, 11) 返回一个元祖，第一个位置值为行数，第二个位置值为了列数
print(f"该元祖类型：{type(data.shape)}")

# 数据处理
null_detail_Series = data.isnull().sum(axis=0)  # 统计每列数据的的空值数量，本例子中age列空值4267个，NumberOfDependents列空值4267个
print(f"尝试统计各个列的空值数量：{type(null_detail_Series)}")


# 处理空值数据--本例子采用去掉为空的数据
data.dropna(inplace=True)  ##去掉为空的数据
# data.shape  # (108648, 11)，112915-4267=108648

print(f"空值处理后的数据集合的行列元祖：{data.shape}")

# SeriousDlqin2yrs: Person experienced 90 days past due delinquency or worse
# y 是目标变量，包含 SeriousDlqin2yrs 列中的所有值。
# X 是特征变量，包含 data DataFrame 中除 SeriousDlqin2yrs 列外的所有列。
y = data['SeriousDlqin2yrs']
X = data.drop('SeriousDlqin2yrs', axis=1)


# 分裂训练集合测试集合
from sklearn.model_selection import train_test_split
# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



#直接使用已经存在的模型
# 加载模型
loaded_model = joblib.load('./model_save/logistic_regression_model.pkl')
y_pred = loaded_model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)