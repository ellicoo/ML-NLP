import pandas as pd
from IPython.core.display_functions import display

df = pd.DataFrame([[2, 2.5, 2, 3.5], [2, 2, 2, 2], [3, 3, 3, 3]], columns=["col1", "col2", "col3", "col4"])

print("原始df")
display(df)

df_mean=df.mean(axis=1) # 得到按水平方向计算的均值--其实是每列取一个值的操作
# axis=0 表示先进行第一列数据竖直方向的操作，感觉像是一行一行来，处理完处理第二列的竖直方向的操作，
print("均值处理后的df")
display(df_mean)

# 然而，如果我们调用 df.drop((name, axis=1),我们实际上删掉了一列，而不是一行

df_drop_col=df.drop("col4", axis=1) #按水平方向的操作，即是列操作
print("删除列的df")
display(df_drop_col)