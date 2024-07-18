
import joblib


def process_data(data):
    """
    原始数据处理
    input:
        data: 从测试 csv 文件读取的 DataFrame 数据
    output:
        X, labels: 经过预处理和特征选择后的特征数据、标签数据
    """
    import numpy as np
    import pandas as pd
    from sklearn.impute import KNNImputer
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    mask = data.isnull()

    # 填充缺失值
    data.fillna(data.mode().iloc[0], inplace=True)

    # class 使用 OneHotEncoder
    label = OneHotEncoder().fit_transform(
        data['class'].values.reshape(-1, 1)).toarray()
    # 其余使用 LabelEncoder 编码

    for col in data.columns:
        data[col] = LabelEncoder().fit_transform(data[col])

    # 将 class 重新加入
    data['class'] = label

    data.where(~mask, np.nan, inplace=True)

    # KNN Imputer
    data = pd.DataFrame(KNNImputer(
        n_neighbors=5).fit_transform(data), columns=data.columns)

    data.head()

    x = data.drop('class', axis=1)
    y = data['class']

    return x, y


# 编写符合测试需求的方法
# 加载模型,加载请注意 path 是相对路径, 与当前文件同级。
path = './results/ada_model_2024-07-18-01-53-49.pkl'
model = joblib.load(path)


def predict(X):
    """
    模型预测
    input:
        毒蘑菇测试数据
    output:
        是否有毒
    """
    X.drop(columns = ['veil-type'], inplace = True)
    X.drop(columns=['veil-color'], inplace=True)
    X.drop(columns=['gill-attachment'], inplace=True)
    X.drop(columns=['cap-color'], inplace=True)

    y_predict = model.predict(X)
    return y_predict


if __name__ == '__main__':
    import pandas as pd
    data = pd.read_csv('./datasets/mushroom.csv')
    X, y = process_data(data)
    y_predict = predict(X)

    from sklearn.metrics import f1_score
    print(f1_score(y, y_predict))
