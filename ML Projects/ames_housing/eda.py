import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv("data/train.csv", index_col='Id')
test = pd.read_csv("data/test.csv", index_col='Id')


y = train['SalePrice']
X_train = train.drop(columns=['SalePrice'])
train_numerical = train.select_dtypes(include="number")
cor_mat = train_numerical.corr()
# np.black_white_dots(cor_mat.tonumpy(), black = 1, red = -1, white = 0)
# 转换为 numpy array
cor_mat = cor_mat[['SalePrice']]
corr_array = cor_mat.to_numpy()

# 用 seaborn 的 heatmap 来画相关矩阵
# plt.figure(figsize=(12, 10))
# sns.heatmap(corr_array, cmap="coolwarm", center=0, vmin=-1, vmax=1)
# plt.show()

cor_mat = cor_mat.sort_values(by='SalePrice', ascending=False)
print(cor_mat)