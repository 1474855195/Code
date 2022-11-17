import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn import tree
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
#训练集测试集
x_train=df1.iloc[:,0:19].values
y_train=df1.iloc[:,20].values
x_test=df2.iloc[:,0:19].values
y_true=df2.iloc[:,20].values
#参数空间
space={'min_samples_split':hp.uniform('min_samples_split',0.1,1.0),
       'min_samples_leaf':hp.uniform('min_samples_leaf',0.1,1.0),
       'max_features':hp.choice('max_features',['sqrt','log2']),
      'criterion':hp.choice('criterion',['gini','entropy'])}
#代理函数
algo=partial(anneal.suggest)
def DT(args):
    clf=tree.DecisionTreeClassifier(random_state=0,
                                    min_samples_split=float(args['min_samples_split']),
                                     min_samples_leaf=float(args['min_samples_leaf']),
                                     max_features=args['max_features'],
                                     criterion=args['criterion'])
    auc=sk_model_selection.cross_val_score(clf,x_train,y_train,scoring='roc_auc',cv=10)
    return -auc.mean()
time_start = time.process_time()
best=fmin(DT,space=space,algo=algo,max_evals=200)
time_end = time.process_time()
time_sum = time_end - time_start
#设置离散变量
if best['max_features']==0:
    best['max_features']='sqrt'
else:
    best['max_features']='log2'
if best['criterion']==0:
    best['criterion']='gini'
else:
        best['criterion']='entropy'
#参数带回实验进行预测
model=tree.DecisionTreeClassifier(random_state=0,min_samples_split=float(best['min_samples_split']),
                                 min_samples_leaf=float(best['min_samples_leaf']),
                                 max_features=best['max_features'],
                                 criterion=best['criterion'])
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
probas=model.predict_proba(x_test)
fpr,tpr,thresholds=roc_curve(y_true,probas[:,1])
auc=metrics.auc(fpr,tpr)

