from sklearn import metrics
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import sklearn.model_selection as sk_model_selection
from hyperopt import fmin,tpe,hp,partial,anneal,rand
from sklearn import metrics
import time
from hyperopt.early_stop import no_progress_loss
#读取文件
df1=pd.read_csv(r'ant-1.3.csv',header=0)
df2=pd.read_csv(r'ant-1.4.csv',header=0)
#调参实验
x_train=df1.iloc[:,0:19].values
y_train=df1.iloc[:,20].values
x_test=df2.iloc[:,0:19].values
y_true=df2.iloc[:,20].values

def KNN(args):
    hKN=KNeighborsClassifier(n_neighbors=int(args['n_neighbors']),weights=args['weights'])
    auc=sk_model_selection.cross_val_score(hKN,t_train,y_train,scoring='roc_auc',cv=10)
    return -auc.mean()
#参数空间
space={'n_neighbors':hp.quniform('n_neighbors',1,50,2),
      'weights':hp.choice('weights',['uniform','distance'])}
#代理函数
algo=partial(tpe.suggest)
time_start = time.process_time()
best = fmin(KNN,space=space,algo = algo,max_evals=200)
time_end = time.process_time()
time_sum = time_end - time_start
#设置离散变量
if best['weights']==0:
    best['weights']='uniform'
else:
    best['weights']='distance'
#参数带回实验进行预测
model=KNeighborsClassifier(n_neighbors=int(best['n_neighbors']),weights=best['weights'])
model.fit(t_train,y_train)
y_pred=model.predict(t_test)
#评分
probas=model.predict_proba(x_test)
fpr,tpr,thresholds=roc_curve(y_true,probas[:,1])
auc=metrics.auc(fpr,tpr)
#计算模型稳定性
model_d=KNeighborsClassifier()
model_d.fit(x_train,y_train)
from mlxtend.evaluate import bias_variance_decomp
MSE_Default,Bias_Default,var_Default=bias_variance_decomp(model_d,x_train,y_train,x_test,y_true,
                                  loss='mse',num_rounds=100,random_seed=1)
MSE_TPE,Bias_TPE,var_TPE=bias_variance_decomp(model,x_train,y_train,x_test,y_true,
                                  loss='mse',num_rounds=100,random_seed=1)