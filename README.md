# 毒蘑菇分类：披着华丽外衣的“杀手”

## 1. 项目背景介绍

蘑菇（mushroom）是深受人们喜爱的一种美食，但蘑菇华丽的外衣下却可能藏着致命的危险。我国是世界上蘑菇种类最多的国家之一，与此同时，蘑菇中毒却是我国最严重的食品安全问题之一，据相关报道，2021 年，我国共开展研究蘑菇中毒事件 327 次，涉及 923 例患者，20 例死亡，总死亡率为 **2.17%**。对于非专业人士，无法从外观、形态、颜色等方面区分有毒蘑菇与可食用蘑菇，没有一个简单的标准能够将有毒蘑菇和可食用蘑菇区分开来。要了解蘑菇是否可食用，必须采集具有不同特征属性的蘑菇是否有毒进行分析。在本次比赛中，对蘑菇的 22 种特征属性进行分析，从而得到蘑菇可使用性模型，能更好的预测出蘑菇是否可食用。

下面将使用**支持向量机**对蘑菇进行分类。

- 数据集相关介绍

  - mushroom.csv：蘑菇的22种相关特征

- 数据集的整体特征

| 数据集名称 | 数据类型 | 特征数 | 实例数 | 值缺失 | 相关任务 |
| :--------- | :------- | :----- | :----- | :----- | :------- |
| 毒蘑菇分类 | 字符数据 | 22     | 8123   | 无     | 分类     |

- 属性描述
  **文件 mushroom.csv 包含 22 个字段，具体信息如下：**
  
  每一行代表蘑菇的一种属性特征，对应 mushroom.csv 中的一列，其中第一列为分类值edible（可食用）, poisonous（有毒）：

| No   | 属性                   | 数据类型 | 字段取值                                                                                              |
| :--- | :--------------------- | :------- | :---------------------------------------------------------------------------------------------------- |
| 1    | 蘑菇是否有毒           | char     | ''e'（可食用）'p'（有毒）                                                                             |
| 2    | 蘑菇帽子的形状         | char     | 'b'（凸）'c'（锥形）'x'（扁平）'f'（凹）'k'（钟形）'s'（沙漏形）                                      |
| 3    | 蘑菇帽子表面的质地     | char     | 'f'（纤维）'g'（沟槽）'y'（光滑）'s'（粘性）                                                          |
| 4    | 蘑菇帽子的颜色         | char     | 'n'（棕）'b'（黄）'c'（肉桂）'g'（绿）'r'（红）'p'（粉）'u'（紫）'e'（淡黄）'w'（白）'y'（黄）        |
| 5    | 蘑菇有无斑点           | char     | 't'（有斑点）'f'（无斑点）                                                                            |
| 6    | 蘑菇气味               | char     | 'a'（杏仁）'l'（淡漠）'c'（腥臭）'y'（肥皂）'f'（芳香）'m'（蘑菇）'n'（无味）'p'（辛辣）'s'（刺激性） |
| 7    | 蘑菇菌肉的附着方式     | char     | 'a'（附着）'d'（下降）'f'（自由）'n'（无）                                                            |
| 8    | 蘑菇菌褶的间距         | char     | 'c'（密集）'w'（稍密）'d'（稀疏）                                                                     |
| 9    | 蘑菇菌褶的大小         | char     | 'b'（较大）'n'（较小）                                                                                |
| 10   | 蘑菇菌褶的颜色         | char     | 'k'（黑）'n'（淡黄）'b'（巧克力）'h'（褐）'g'（绿）'r'（红）'o'（橙）'p'（粉）'u'（紫） 'e'（淡黄棕） |
| 11   | 蘑菇柄的形状           | char     | 't'（细长）'e'（扁平）                                                                                |
| 12   | 蘑菇柄根部的形状       | char     | 'b'（结球）'c'（锥形）'u'（块状）'e'（裸露）'z'（支离破碎）。                                         |
| 13   | 环上方蘑菇柄的表面类型 | char     | 'f'（纤维）'y'（光滑）'k'（有点粘）'s'（沟槽）                                                        |
| 14   | 环下方蘑菇柄的表面类型 | char     | 'f'（纤维）'y'（光滑）'k'（有点粘）'s'（沟槽）                                                        |
| 15   | 环上方蘑菇柄的颜色     | char     | 'n'（棕）'b'（黄）'c'（肉桂）'g'（绿）'o'（橙）'p'（粉）'e'（淡黄）'w'（白）'y'（黄）                 |
| 16   | 环下方蘑菇柄的颜色     | char     | 'n'（棕）'b'（黄）'c'（肉桂）'g'（绿）'o'（橙）'p'（粉）'e'（淡黄）'w'（白）'y'（黄）                 |
| 17   | 蘑菇的菌环类型         | char     | 'p'（部分）                                                                                           |
| 18   | 蘑菇的菌环颜色         | char     | 'n'（棕色）'o'（橙色）'w'（白色）'y'（黄色）                                                          |
| 19   | 蘑菇的孢子数量         | char     | 'n'（没有）'o'（一个） 't'（两个）                                                                    |
| 20   | 蘑菇的孢子印类型       | char     | 'c'（有环）'e'（大环）'f'（平坦）'l'（线性）'n'（无）'p'（鳞片）                                      |
| 21   | 蘑菇的孢子印颜色       | char     | 'k'（黑）'n'（淡黄）'b'（巧克力）'h'（褐）'r'（红）'o'（橙）'u'（紫）'w'（白）                        |
| 22   | 蘑菇的种群密度         | char     | 'a'（多）'c'（群）'n'（大）'s'（几个）'v'（遍布）'y'（单个）                                          |
| 23   | 蘑菇生长地             | char     | 'g'（草地）'l'（叶）                                                                                  |

在接下来的流程中，通过**数据集处理、数据集分析和可视化、模型构建、模型训练、模型评估**五个部分展开。

## 2. 数据集处理

数据集已经导入在 **datasets 文件夹**下，利用 **pandas** 来读取 csv 文件，并给数据的每一列标记上属性(标记为英文)，一共 23 列，其含义如模块一中属性表格所示，各字段的含义也在上述表格中。数据集为结构化数据，在此利用pandas工具展示其中 5 行数据：

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

mushrooms = pd.read_csv('./datasets/mushroom.csv')
mushrooms.columns = ['class', 'cap-shape', 'cap-surface', 'cap-color', 'ruises', 'odor', 'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
                     'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']
mushrooms
```

运行上述代码可以看到，一共有 23 列，第一列代表蘑菇的类别，即本次项目探究的是可食用还是有毒的蘑菇，其余 22 列为每种蘑菇的特征，依次为：蘑菇帽子的形状，蘑菇帽子表面的质地，蘑菇帽子的颜色，蘑菇有无斑点，蘑菇气味，蘑菇菌肉的附着方式，蘑菇菌褶的间距，蘑菇菌褶的大小，蘑菇菌褶的颜色，蘑菇柄的形状，蘑菇柄根部的形状，环上方蘑菇柄的表面类型，环下方蘑菇柄的表面类型，环上方蘑菇柄的颜色，环下方蘑菇柄的颜色，蘑菇的菌环类型，蘑菇的菌环颜色，蘑菇的孢子数量，蘑菇的孢子印类型，蘑菇的孢子印颜色，蘑菇的种群密度，蘑菇生长地。每一种属性对应的取值在上述表格中。这些特征与蘑菇的毒性有哪些联系呢，我们接着往下看。

## 3. 数据集分析和可视化

首先通过 Pandas 工具查看原始数据的大小：

```python
print(mushrooms.shape)
```

该数据集共有 25986 行 23 列，其中属性值已经在之前分析过。我们要对蘑菇特征进行分析之前先要检查数据集中有没有缺失值，数据的完整性对后序的分析很重要。

```python
mushrooms.isnull().sum()
```

可以看到数据具有大量缺失值，需要处理缺失值后才可以进行下一步的分析。我们要对蘑菇是否有毒进行判断，先看看数据集样本中可食用蘑菇的数量和有毒蘑菇的数量：

```python
mushrooms['class'].value_counts()
```

在所有样本数据中，e (可食用蘑菇)有 14354 个样本，p (有毒蘑菇)有 11632 个样本，二者比例相差不大，相近的比例有利于后续的分析。接下来对样本中的一些信息进行可视化，首先对菌盖的颜色进行可视化：

```python
# 中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
cap_colors = mushrooms['cap-color'].value_counts()
m_height = cap_colors.values.tolist()
cap_colors.axes
cap_color_labels = cap_colors.axes[0].tolist()


def label_num(rects, fontsize=14):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, 1*height, '%d' % int(height),
                ha='center', va='bottom', fontsize=fontsize)


color_number = np.arange(10)
width = 0.7
# 将蘑菇的颜色与柱状图颜色画成一样
colors = ['#DEB887', '#778899', '#DC143C', '#FFFF99',
          '#f8f8ff', '#F0DC82', '#FF69B4', '#D22D1E', '#C000C5', 'g']

fig, ax = plt.subplots(figsize=(10, 7), dpi=300)

cap_colors_bars = ax.bar(color_number, m_height, width, color=colors)

ax.set_xlabel("菌盖颜色", fontsize=15)
ax.set_ylabel('数量', fontsize=15)
ax.set_title('蘑菇菌盖颜色', fontsize=18)
ax.set_xticks(color_number)
ax.set_xticklabels(('棕色', '灰色', '红色', '黄色', '白色', '深黄色', '粉色', '肉桂色', '紫色', '绿色'),
                   fontsize=13)

label_num(cap_colors_bars)
plt.show()
```

由上面运行结果图像可以看到，菌盖颜色中棕色与灰色两种颜色数量最多，二者加起来占了总体样本的一半左右。在日常生活中，棕色和灰色的蘑菇确实更常见一些，在上面图像中，红色，黄色，白色三种颜色也占了很大比例。在生活中，很多人仅凭经验会认为颜色鲜艳的蘑菇往往有毒，而像棕色，灰色这种蘑菇往往没有毒，事实会是这样吗？我们继续将数据集中各个颜色中有毒蘑菇和可食用蘑菇做一个可视化。

```python
# 创建两个列表，分别为各颜色有毒蘑菇的数量和个颜色食用菌的数量
poisonous_cc = []
edible_cc = []

for capColor in cap_color_labels:
    size = len(mushrooms[mushrooms['cap-color'] == capColor].index)  # 各颜色蘑菇总数
    edibles = len(mushrooms[(mushrooms['cap-color'] == capColor)
                  & (mushrooms['class'] == 'e')].index)  # 各颜色食用菌的数量
    edible_cc.append(edibles)
    poisonous_cc.append(size-edibles)


width = 0.4
fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
edible_bars = ax.bar(color_number, edible_cc, width,
                     color='#FFB90F')  # 画食用菌的bars

poison_bars = ax.bar(color_number+width, poisonous_cc, width, color='#4A708B')

ax.set_xlabel("菌盖颜色", fontsize=15)
ax.set_ylabel('数量', fontsize=15)
ax.set_title('不同菌盖颜色下蘑菇的毒性分布', fontsize=18)
ax.set_xticks(color_number + width / 2)
ax.set_xticklabels(('棕色', '灰色', '红色', '黄色', '白色', '深黄色', '粉色', '肉桂色', '紫色', '绿色'),
                   fontsize=13)
ax.legend((edible_bars, poison_bars), ('可食用', '有毒性'), fontsize=17)
label_num(edible_bars, 10)
label_num(poison_bars, 10)
plt.show()
```

由运行结果可知，一些颜色鲜艳的蘑菇，如红色，黄色，粉色等有毒性的数量比可食用的数量多，但棕色，灰色这些较常见的颜色中可食用蘑菇和有毒蘑菇的占比相差不大。由此可见，仅仅由蘑菇颜色来判断蘑菇是否有毒性是不科学的，需要综合考虑多个特征才能判断蘑菇是否可食用。食物的气味往往会是一个重要特征，我们来观察蘑菇的气味对蘑菇毒性的影响，和上面的方法一样，绘制出每种气味中蘑菇有毒与无毒的数量。

```python
odors = mushrooms['odor'].value_counts()
m_height = odors.values.tolist()
odors.axes
odors_labels = odors.axes[0].tolist()


def label_num(rects, fontsize=14):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, 1*height, '%d' % int(height),
                ha='center', va='bottom', fontsize=fontsize)


odor_number = np.arange(9)
width = 0.7

poisonous_cc = []
edible_cc = []

for odor in odors_labels:
    size = len(mushrooms[mushrooms['odor'] == odor].index)  # 各颜色蘑菇总数
    edibles = len(mushrooms[(mushrooms['odor'] == odor) & (
        mushrooms['class'] == 'e')].index)  # 各颜色食用菌的数量
    edible_cc.append(edibles)
    poisonous_cc.append(size-edibles)  # 总减食用得到有毒的数量


width = 0.4
fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
edible_bars = ax.bar(odor_number, edible_cc, width,
                     color='#FFB90F')  # 画食用菌的bars

poison_bars = ax.bar(odor_number+width, poisonous_cc, width, color='#4A708B')
plt.rcParams['font.sans-serif'] = ['SimHei']
ax.set_xlabel("蘑菇气味", fontsize=15)
ax.set_ylabel('数量', fontsize=15)
ax.set_title('不同菌盖气味下蘑菇的毒性分布', fontsize=18)
ax.set_xticks(color_number + width / 2)
# ax.set_xticklabels(('无味', '臭位', '腥味', '香辣味', '杏仁味', '茴香位', '香辣味', '烟熏味', '霉味'),
#                    fontsize=13)
ax.legend((edible_bars, poison_bars), ('可食用', '有毒性'), fontsize=17)
label_num(edible_bars, 10)
label_num(poison_bars, 10)
plt.show()
```

通过做出的蘑菇气味对蘑菇毒性影响的柱状图可以看出，绝大多数可食用的蘑菇都是无味，杏仁味和茴香味的，其他的一些味道均为毒蘑菇可见气味是区分蘑菇毒性的一个重要特征。

```python
mushrooms_encoded = pd.get_dummies(mushrooms)

corr = mushrooms_encoded.corr()

sns.set(rc={'figure.figsize': (10, 8)})

sns.heatmap(corr, cmap="Blues")
```

热力图中的颜色可以表示特征之间的相关性。颜色越浅，相关性越强，颜色越深，相关性越弱。可以使用热力图来确定哪些特征之间具有强烈的正相关性或负相关性。由于 mushroom 数据集中特征数有 22 个，绘制出的热力图比较密集，只能在总体上观察各个特征之间的相关性。总体上分析可以看到odor（气味）特征与其他特征之间的相关性较弱，这表明**气味可以是区分蘑菇是否有毒的重要特征**，这也与上面绘制出的柱状图相吻合。dor（气味）和gill-size（菌褶大小）是与其他特征相关性较高的特征，这表明它们可能对分类结果产生较大的影响。

## 4. 模型构建与训练

### 4.1  SVM模型的构建与训练

我们在构建模型之前先要对数据集进行处理，由于 mushroom 数据集所有的取值均为 char 类型的字符，我们要分析特征之间的关系需要将 char 类型映射到数字上。我们使用了Scikit-learn 中的 LabelEncoder 类对数据集的文本类型特征进行编码。编码后将文本类型的特征转换为数字类型，以便于机器学习算法的处理。LabelEncoder 类将每个文本类型特征的取值映射到一个唯一的整数值。例如某个特征有三种取值 A、B 和 C，LabelEncoder 将 A 映射到整数 0，B 映射到整数 1，C 映射到整数 2。

```python
# 利用众数填充缺失值
mushrooms = mushrooms.fillna(mushrooms.mode().iloc[0])
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()
onehotencoder = OneHotEncoder()

for col in mushrooms.columns:
    mushrooms[col] = labelencoder.fit_transform(mushrooms[col])
X = mushrooms.drop('class', axis=1)
y = mushrooms['class']
X = onehotencoder.fit_transform(X).toarray()
X
```

处理过后数据集中取值全转换成了 int 类型数据，如上图运行结果所示。数据集中 y 为标签，也就是蘑菇有毒和可食用这两种，X 为特征
我们已经将数据集处理完毕，所有的数据都已经转化成了int类型，接下来可以搭建**支持向量机 (SVM) 模型**。SVM（Support Vector Machine）是一种常用的监督学习算法，广泛应用于模式识别、分类和回归分析等领域。在本问题中，由 22 个特征向量来推断蘑菇是有毒还是无毒，本质上是一个二分类，SVM 算法非常适合分类问题。在训练之前，我们要划分数据集，将训练集和测试集合按照 8:2 的比例划分，接下来便可以**创建 SVM 模型进行训练**，将训练好的模型保存在 model 文件夹下。

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import joblib

# 将数据集拆分为训练集和测试集，按照8:2划分

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print(X_train.shape)

# 创建 SVM 模型并进行训练

svm = SVC()
svm.fit(X_train, y_train)

# 保存模型

joblib.dump(svm, './results/mushroom_svm_model.pkl')
```

## 5. 模型评估

当我们训练好了 SVM 模型后，模型的性能一般可以从 4 个方面进行评估：

1. **准确率（Accuracy）**：准确率是指分类器正确分类的样本数占总样本数的比例。准确率越高，表示分类器的性能越好。

2. **精确率（Precision）和召回率（Recall）**：精确率是指分类器预测为正类别的样本中有多少是真正的正类别样本。召回率是指真正的正类别样本中有多少被分类器正确地预测为正类别。精确率和召回率通常是成对使用的，它们的值越高，表示分类器的性能越好。

3. **混淆矩阵**：将分类器的预测结果与实际结果进行对比，从而计算出四种不同的分类情况：真正例（True Positive，TP）、假正例（False Positive，FP）、真反例（True Negative，TN）和假反例（False Negative，FN）。在使用 SVM 进行分类时，混淆矩阵可以用于评估模型的性能，从而判断模型的分类能力和错误类型。

4. **ROC 曲线和 AUC 值**：ROC 曲线是一种用于评估二元分类器性能的图形化工具。ROC 曲线的横轴是假正例率（False Positive Rate，FPR），纵轴是真正例率（True Positive Rate，TPR），它们的计算公式分别为 FPR=FP/(FP+TN) 和 TPR=TP/(TP+FN)，其中 TP、FP、TN 和 FN 分别表示真正例、假正例、真反例和假反例的数量。AUC 值是 ROC 曲线下方的面积，它的取值范围为 0.5 到 1，AUC 值越大，表示分类器的性能越好。

下面将从这四个方面对我们训练好的模型进行评估：

```python
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# 预测并评估模型
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)
print("Confusion matrix:\n", confusion)
y_pred
```

基于上述代码的运行结果，我们可以根据数据对模型进行评估：

1.**准确率**：经过训练后的模型在测试集上的准确率达到了 **0.746**， 可以尝试调整参数或更换模型以达到更高的准确率；

2.**精确率和召回率**：运行结果中精确率和召回率分别是 **0.759** 和 **0.634**，F1 分数是精确率和召回率的调和平均数，它综合了两个指标的性能，可以看到 F1 分数稳定在了 **0.691** 左右。

3.**混淆矩阵**：

| 真实类别预测类别 | 正类别 | 负类别 |
| :--------------- | :----- | :----- |
| 正类别           | 2405   | 468    |
| 负类别           | 850    | 1475   |

值得注意的是，FN 的值还是相对较大，也就是说**错误地将有毒菇样本分类为食用菇**，这是一个**很严重的问题**，当我们的分类出现将毒蘑菇分类成可食用蘑菇的时候可能会导致食用有毒蘑菇而中毒，因此需要对模型加以调整；

## 6. 评分

**注意：**

通过对以上步骤流程的了解，相信大家对该任务有了深刻的认识，但是模型比较简单，
准确率也不高，大家可以试着写自己的模型，并将其调到最佳状态。

1. 你可以在我们准好的接口中实现模型（若使用可以修改函数接口），也可以自己实现深度学习模型，写好代码后可以在 Py 文件中使用 CPU 进行模型训练。
2. 在训练模型等过程中如果需要**保存数据、模型**等请写到 **results** 文件夹，如果采用 [离线任务](https://momodel.cn/docs/#/zh-cn/%E5%9C%A8GPU%E6%88%96CPU%E8%B5%84%E6%BA%90%E4%B8%8A%E8%AE%AD%E7%BB%83%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E6%A8%A1%E5%9E%8B) 请务必将模型保存在 **results** 文件夹下。
3. 训练出自己最好的模型后，先按照下列 cell 操作方式实现 NoteBook 加载模型测试；请测试通过再进行【系统测试】。
4. 请将你的数据预处理函数进行修改为`process_data()`的格式，能够对后台的测试数据进行处理以便对你的模型进行测试打分。
5. 测试数据的格式与你所使用的数据相同，为csv文件。
6. 在修改 `process_data()` 函数时请务必注意，请保持测试数据集顺序与原始顺序一致，避免对测试数据的排序结果和原排序结果不同导致测试结果的偏差。
7. 请填写你的模型路径及名称并补充 `predict()` 函数以实现预测。
8. 点击左侧栏提交结果后点击生成文件则需勾选 `process_data()` 函数和 `predict()` 函数的 cell。
9. 请导入必要的包和第三方库 (包括此文件中曾经导入过的)。
10. 请加载你认为训练最佳的模型，即请按要求填写模型路径。
11. 测试提交时服务端会调用 `process_data()` 函数和 `predict()` 函数，请不要修改该函数的输入输出及其数据类型。

```python
========================================  **模型预测代码答题区域**  ===========================================  
在下方的代码块中编写 **数据处理 process_data()** 和 **模型预测 predict()** 部分的代码，请勿在别的位置作答
def process_data(data):
    """
    原始数据处理
    input:
        data: 从测试 csv 文件读取的 DataFrame 数据
    output:
        X, labels: 经过预处理和特征选择后的特征数据、标签数据
    """
    new_features, label = None, None
    # -------------------------- 实现数据处理和特征选择部分代码 ----------------------------

    # ------------------------------------------------------------------------
    # 返回筛选后的数据
    return X, labels
# -------------------------- 请加载最满意的模型 ---------------------------
# 加载模型(请加载你认为的最佳模型)
# 加载模型,加载请注意 model_path 是相对路径, 与当前文件同级。
# 如果你的模型是在 results 文件夹下的 my_model.m 模型，则 model_path = './results/my_model.m'
path = None

# 加载模型
model = None


def predict(X):
    """
    模型预测
    :param  X : 特征数据，是 process_data 函数的返回值之一。
    :return y_predict : 预测结果是标签值。
    """

    # -------------------------- 实现模型预测部分的代码 ---------------------------
    # 获取输入的类别
    y_predict = None

    # -------------------------------------------------------------------------

    # 返回类别
    return y_predict
```

========================================  **测试提交函数示例**  ===========================================

```python
def process_data(data):
    """
    原始数据处理
    input:
        data: 从测试 csv 文件读取的 DataFrame 数据
    output:
        X, labels: 经过预处理和特征选择后的特征数据、标签数据
    """
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.preprocessing import LabelEncoder

    # 利用众数填充缺失值
    data = data.fillna(data.mode().iloc[0])

    # 对数据进行编码和独热编码
    labelencoder = LabelEncoder()
    onehotencoder = OneHotEncoder()
    for col in data.columns:
        data[col] = labelencoder.fit_transform(data[col])

    # 获取标签
    labels = data['class']

    # 获取特征数据
    eval_x = data.drop(columns='class')

    X = onehotencoder.fit_transform(eval_x).toarray()

    return X, labels
# 编写符合测试需求的方法
import joblib

# 加载模型,加载请注意 path 是相对路径, 与当前文件同级。
path = './results/mushroom_svm_model.pkl'
model = joblib.load(path)


def predict(X):
    """
    模型预测
    input:
        毒蘑菇测试数据
    output:
        是否有毒
    """

    y_predict = model.predict(X)
    return y_predict
```
