import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics, tree
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score,StratifiedShuffleSplit
import graphviz
from sklearn.metrics import classification_report

data=pd.read_csv(r"C:\data\autism_screening.csv")
#print(data.head())
#print(data.shape)
#print(data.info())
#筛选特征
data.drop(["ethnicity","contry_of_res","used_app_before","age_desc","relation","age_desc","result"],inplace=True,axis=1)
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


#分离特征值和标签
x=data.iloc[:,data.columns != "Class/ASD"]
#print(x)
y=data.iloc[:,data.columns == "Class/ASD"]
#print(y)
Xtrain,Xtest,Ytrain,Ytest=train_test_split(x,y,test_size=0.3,random_state=55)
#print(Xtrain)
# #将划分后的测试集训练集索引排序正常
# for i in [Xtrain,Xtest,Ytrain,Ytest]:
#     i.index=range(i.shape[0])
#print(Xtrain)
#print(Ytrain)
print(Xtest)

# #采用交叉验证来选取criterion参数
# cv = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=42)
# clf = tree.DecisionTreeClassifier(criterion='entropy', random_state=42)
# score = cross_val_score(clf, x, y, cv=cv).mean()
# print("参数为entropy时:",score)
# clf = tree.DecisionTreeClassifier(criterion='gini', random_state=42)
# score = cross_val_score(clf, x, y, cv=cv).mean()
# print("参数为gini时:",score)

#调整参数，寻找最佳深度
train=[]
test=[]
score=[]
for i in range(10):
    clf=DecisionTreeClassifier(criterion="entropy"
                              ,random_state=55
                              ,max_depth=i+1
                              )
    clf=clf.fit(Xtrain,Ytrain)
    score_tr = clf.score(Xtrain, Ytrain)
    score_te=clf.score(Xtest,Ytest)
    train.append(score_tr)
    test.append(score_te)
plt.plot(range(1,11),train,color="red",label="train")
plt.plot(range(1,11),test,color="blue",label="test")
plt.legend()
plt.show()

#建立模型
clf=DecisionTreeClassifier(criterion="gini"
                              ,random_state=42
                             # ,splitter="random"
                              #,max_depth=6
                             # ,min_samples_leaf=2
                              #,min_samples_split=3
                              )
clf=clf.fit(Xtrain,Ytrain)



#画出决策树
# feature_name=["A1_Score","A2_Score","A3_Score","A4_Score","A5_Score","A6_Score",
#               "A7_Score","A8_Score","A9_Score","A10_Score","age","gender","jundice","austim"]
# mytree=tree.export_graphviz(clf,
#                             feature_names=feature_name,
#                             class_names=["未患自闭症","患有自闭症"],
#                             filled=True,
#                             rounded=True)
#获取生成的决策树代码
# graph=graphviz.Source(mytree)
# print(graph)
print(clf.feature_importances_)
y_pred = clf.predict(Xtest)
print('训练集评分:', clf.score(Xtrain,Ytrain))
print('测试集评分:', clf.score(Xtest,Ytest))
# print("查准率：", metrics.precision_score(Ytest,y_pred))
# print('召回率:',metrics.recall_score(Ytest,y_pred))
# print('f1分数:', metrics.f1_score(Ytest,y_pred))
print(y_pred)
print(classification_report(Ytest,y_pred))


