import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import sklearn.model_selection as sk_model_selection
from hyperopt import fmin,tpe,hp,partial,anneal,rand
from sklearn import metrics
from hyperopt.early_stop import no_progress_loss
import time
#读取文件
df1=pd.read_csv(r'ant-1.3.csv',header=0)
df2=pd.read_csv(r'ant-1.4.csv',header=0)
#调参实验
x_train=df1.iloc[:,0:19].values
y_train=df1.iloc[:,20].values
x_test=df2.iloc[:,0:19].values
y_true=df2.iloc[:,20].values
#参数空间
space={'alpha': hp.uniform('alpha',0.0001,1),
        'activation': hp.choice('activation',['identity','logistic','tanh']),
        'max_iter': hp.choice('max_iter',[*range(10,500,20)])}
def MLP(args):
    ML= MLPClassifier(activation=args['activation'],alpha=float(args['alpha']),max_iter=int(args['max_iter']))
    roc_auc=sk_model_selection.cross_val_score(ML,x_train,y_train,scoring='roc_auc',cv=5)
    return -roc_auc.mean()
#代理函数
algo=partial(rand.suggest)

time_start = time.process_time()
best = fmin(MLP,space=space,algo = algo,max_evals=200,early_stop_fn = no_progress_loss(50))
time_end = time.process_time()  
time_sum = time_end - time_start
#设置离散变量
if best['activation']==0:
        best['activation']='identity'
if best['activation']==1:
        best['activation']='logistic'
if best['activation']==2:
        best['activation']='tanh'
#参数带回实验进行预测
model=MLPClassifier(activation=best['activation'],alpha=float(best['alpha']),max_iter=best['max_iter']*20+10)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
#评分

probas=model.predict_proba(x_test)
fpr,tpr,thresholds=roc_curve(y_true,probas[:,1])
auc=metrics.auc(fpr,tpr)
