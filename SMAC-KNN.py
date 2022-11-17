#导入SMAC包
import skopt
from skopt import forest_minimize
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import sklearn.model_selection as sk_model_selection
from sklearn import metrics
import time
#读取文件
df1=pd.read_csv(r'ant-1.3.csv',header=0)
df2=pd.read_csv(r'ant-1.4.csv',header=0)
#调参实验
x_train=df1.iloc[:,0:19].values
y_train=df1.iloc[:,20].values
x_test=df2.iloc[:,0:19].values
y_true=df2.iloc[:,20].values

#定义model
model=KNeighborsClassifier()
#参数空间
from skopt.space import Real,Integer,Categorical
space=[Integer(1,50,name="n_neighbors"),
        Categorical(('uniform','distance'),transform='label',name="weights")]
#目标函数
from skopt.utils import use_named_args
@use_named_args(space)
def predictK(**params):
    model.set_params(**params)
    roc_auc=sk_model_selection.cross_val_score(model,t_train,y_train,scoring='roc_auc',cv=10)
    return -roc_auc.mean()
#参数优化
time_start = time.process_time()
result=forest_minimize(predictK,space,n_calls=100,random_state=0,base_estimator='RF')
time_end = time.process_time()
time_sum = time_end - time_start
#参数带回实验进行预测
model=KNeighborsClassifier(n_neighbors=result.x[0],weights=result.x[1])
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
probas=model.predict_proba(x_test)
fpr,tpr,thresholds=roc_curve(y_true,probas[:,1])
auc=metrics.auc(fpr,tpr)