#导入SMAC包
import skopt
from skopt import forest_minimize
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import sklearn.model_selection as sk_model_selection
from sklearn import metrics
#读取文件
df1=pd.read_csv(r'ant-1.3.csv',header=0)
df2=pd.read_csv(r'ant-1.4.csv',header=0)
#调参实验
x_train=df1.iloc[:,0:19].values
y_train=df1.iloc[:,20].values
x_test=df2.iloc[:,0:19].values
y_true=df2.iloc[:,20].values
#定义model
clf=tree.DecisionTreeClassifier(random_state=0)
#参数空间
from skopt.space import Real,Integer,Categorical
space=[ Categorical(('sqrt','log2'),name='max_features'),
            Categorical(('gini','entropy'),name='criterion'),
            Real(0.1,1,name='min_samples_leaf'),
            Real(0.1,1,name='min_samples_split')]
#目标函数
from skopt.utils import use_named_args
@use_named_args(space)
def DT(**params):
    clf.set_params(**params)
    auc=sk_model_selection.cross_val_score(clf,t_train,y_train,scoring='roc_auc',cv=10)
    return -auc.mean()
import time
time_start = time.process_time()
result=forest_minimize(DT,space,n_calls=200,random_state=0,base_estimator='RF')
time_end = time.process_time()
time_sum = time_end - time_start
#参数带回实验进行预测
model=tree.DecisionTreeClassifier(random_state=0,min_samples_split=result.x[3],
                                      min_samples_leaf=result.x[2],max_features=result.x[0],criterion=result.x[1])
model.fit(t_train,y_train)
y_pred=model.predict(t_test)

probas=model.predict_proba(t_test)
fpr,tpr,thresholds=roc_curve(y_true,probas[:,1])
auc=metrics.auc(fpr,tpr)
