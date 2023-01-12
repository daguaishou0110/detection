import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegressionCV

data=pd.read_csv(r"C:\data\autism_screening.csv",encoding="gbk")
#print(data.head())
#print(data.shape)
#print(data.info())
#筛选特征
data.drop(["ethnicity","contry_of_res","used_app_before","relation","age_desc","result"],inplace=True,axis=1)
#去除重复值
data.drop_duplicates(inplace=True)
#删除重复值之后恢复索引
data.index=range(data.shape[0])
#print(data.info())
#age缺少两个值
data=data.dropna()
data.index=range(data.shape[0])
#print(data.info())
#用描述法统计来观察数据有无异常值
#print(data.describe())
#有age为“383”的数据将其删除
data=data[data["age"]!=383]
data.index=range(data.shape[0])
#print(data.info())
#将object类型数据转化为int类型
(data["gender"]=="f").astype("int")
lables1=data["gender"].unique().tolist()
data["gender"]=data["gender"].apply(lambda x:lables1.index(x))
lables2=data["jundice"].unique().tolist()
data["jundice"]=data["jundice"].apply(lambda x:lables2.index(x))
lables3=data["austim"].unique().tolist()
data["austim"]=data["austim"].apply(lambda x:lables3.index(x))
lables4=data["Class/ASD"].unique().tolist()
data["Class/ASD"]=data["Class/ASD"].apply(lambda x:lables4.index(x))
#print(data)
#print(data.info())

#划分测试集和训练集
#分离特征值和标签
x=data.iloc[:,data.columns != "Class/ASD"]
#print(x)
y=data.iloc[:,data.columns == "Class/ASD"]
#print(y)
#X=pd.DataFrame(x)
#Y=pd.DataFrame(y)
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.3,random_state=55)

#将划分后的测试集训练集索引排序正常
# for i in [X_train,X_test,Y_train,Y_test]:
#     i.index=range(i.shape[0])
#print(X_train)
#print(Y_train)
print((X_test))
#print(Y_test)


# 标准化数据，保证每个维度的特征数据方差为1，均值为0。使得预测结果不会被某些维度过大的特征值而主导
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)
# 创建模型：逻辑回归
lr = LogisticRegression()
#训练模型
lr.fit(X_train,Y_train.values.ravel())
print("得出来的权重：",lr.coef_)
# 预测类别
Y_pre=lr.predict(X_test)
print("预测的类别：",Y_pre)
#获取概率值
print(lr.predict_proba(np.array(X_test)))

print('训练集评分:', lr.score(X_train,Y_train))
print('测试集评分:', lr.score(X_test,Y_test))
print(classification_report(Y_test,Y_pre))

#指定随机梯度下降优化算法
LR = LogisticRegression(solver='saga')
LR.fit(X_train, Y_train.values.ravel())
print("训练集准确率: ", LR.score(X_train, Y_train))
print("测试集准确率: ", LR.score(X_test, Y_test))

#进行10折交叉验证选择合适的参数C
Cs = 10**np.linspace(-10, 10, 400)
lr_cv = LogisticRegressionCV(Cs=Cs, cv=10, penalty='l2', solver='saga',  max_iter=10000, scoring='accuracy')
lr_cv.fit(X_train, Y_train.values.ravel())
print(lr_cv.C_)

LR = LogisticRegression(solver='saga', penalty='l2', C=0.37491674)
LR.fit(X_train, Y_train.values.ravel())
print("训练集准确率: ", LR.score(X_train, Y_train))
print("测试集准确率: ", LR.score(X_test, Y_test))

