import pandas as pd
from IPython.core.display_functions import display

pd.set_option('display.max_columns', 500) # 设置显示的最大列数
import zipfile
with zipfile.ZipFile('KaggleCredit2.csv.zip', 'r') as z:   ##读取zip里的文件
    f = z.open('KaggleCredit2.csv')
    data = pd.read_csv(f, index_col=0)  # 使用pandas读取csv文件，将第一列作为索引
data.head() # 显示头部5行数据
print(data.shape) #(112915, 11) 返回一个元祖，第一个位置值为行数，第二个位置值为了列数


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


data.isnull().sum(axis=0) # 统计每列数据的的空值数量，本例子中age列空值4267个，NumberOfDependents列空值4267个

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
data.dropna(inplace=True)   ##去掉为空的数据
data.shape # (108648, 11)，112915-4267=108648

print(data.shape)

#SeriousDlqin2yrs: Person experienced 90 days past due delinquency or worse
# y 是目标变量，包含 SeriousDlqin2yrs 列中的所有值。
# X 是特征变量，包含 data DataFrame 中除 SeriousDlqin2yrs 列外的所有列。
y = data['SeriousDlqin2yrs']
X = data.drop('SeriousDlqin2yrs', axis=1)

