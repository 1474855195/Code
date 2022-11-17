#导入SMAC包
import skopt
from skopt import forest_minimize
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
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
model=MLPClassifier()
#参数空间
from skopt.space import Real,Integer,Categorical
space=[ Categorical(('identity','logistic','tanh','relu'),name='activation'),
        Real(0.0001,1,name="alpha"),
        Integer(100,500,name='max_iter')]
#目标函数
from skopt.utils import use_named_args
@use_named_args(space)
def MLP(**params):
    model.set_params(**params)
    auc=sk_model_selection.cross_val_score(model,x_train,y_train,scoring='roc_auc',cv=5)
    return -auc.mean()
time_start = time.process_time()
result=forest_minimize(MLP,space,n_calls=100,random_state=0,base_estimator='RF')
time_end = time.process_time()
time_sum = time_end - time_start
#参数带回实验进行预测
model=MLPClassifier(activation=result.x[0],alpha=result.x[1],max_iter=result.x[2])
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
#评分
probas=model.predict_proba(x_test)
fpr,tpr,thresholds=roc_curve(y_true,probas[:,1])
auc=metrics.auc(fpr,tpr)