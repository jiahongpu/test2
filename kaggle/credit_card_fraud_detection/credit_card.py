import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score, GridSearchCV
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import time
import matplotlib.patches as mpatches
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from imblearn.under_sampling import NearMiss
import collections
from collections import Counter
from imblearn.pipeline import make_pipeline as imblanced_make_pipeline

df = pd.read_csv(r'/Users/jiahongpu/Desktop/kaggle/credit_card/creditcard.csv')
# df_fraud = df[df['Class']==1].reset_index()
# df_other = df[df['Class']==0].reset_index()

# 样本类型分类数量
# sns.countplot('Class',data=df)
# plt.show()

rs = RobustScaler()

df['scaled_amount'] = rs.fit_transform(df['Amount'].values.reshape(-1, 1))
df['scaled_time'] = rs.fit_transform(df['Time'].values.reshape(-1, 1))
df.drop(['Time', 'Amount'], axis=1, inplace=True)
X = df.drop('Class', axis=1)
y = df['Class']

sss = StratifiedShuffleSplit(n_splits=5, random_state=None)
# for train_index, test_index in sss.split(X, y):
#     # print('train_index:' , train_index,'test_index:', test_index)
#     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#     y_train, y_test = y.iloc[train_index], y.iloc[test_index]
#
# # label distribution
# train_unique_label, train_counts_label = np.unique(y_train, return_counts=True)
# test_unique_label, test_counts_label = np.unique(y_test, return_counts=True)
# print(test_unique_label)
# print(train_counts_label/len(y_train))
# print(test_counts_label/len(y_test))

df = df.sample(frac=1, random_state=42)
df_fraud = df[df['Class'] == 1]
non_fraud = df[df['Class'] == 0][:492]
new_distributed_df = pd.concat([non_fraud, df_fraud])
new_df = new_distributed_df.sample(frac=1, random_state=42)

# correlation heatmap
# df_corr = df.corr()
# f, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 20))
# sns.heatmap(df_corr, cmap='coolwarm_r',ax=ax1)
# sub_corr = new_df.corr()
# print(sub_corr['Class'].sort_values())
# sns.heatmap(sub_corr,cmap='coolwarm_r',ax=ax2)
# plt.show()

# V17, V14, V12 and V10
# f, axes = plt.subplots(ncols=4, figsize=(20, 4))
# sns.boxplot('Class', 'V17', data=new_df, ax=axes[0])
# sns.boxplot('Class', 'V14', data=new_df, ax=axes[1])
# sns.boxplot('Class', 'V12', data=new_df, ax=axes[2])
# sns.boxplot('Class', 'V10', data=new_df, ax=axes[3])
# plt.show()

# V14 removing outliers
v14_fraud = new_df['V14'].loc[new_df['Class'] == 1].values
v14_25, v14_75 = np.percentile(v14_fraud, 25), np.percentile(v14_fraud, 75)
# print('Quartile 25:{}|Quartile 75:{}'.format(v14_25, v14_75))
v14_iqr = v14_75 - v14_25
# print('iqr:{}'.format(v14_iqr))
v14_cut_off = v14_iqr * 1.5
v14_lower, v14_upper = v14_25 - v14_cut_off, v14_75 + v14_cut_off
# print('lower:{}|upper:{}'.format(v14_lower, v14_upper))
new_df.drop(new_df[(new_df['V14'] < v14_lower) | (new_df['V14'] > v14_upper)].index, inplace=True)

# v12 removing outliers
v12_fraud = new_df['V12'].loc[new_df['Class'] == 1].values
v12_25, v12_75 = np.percentile(v12_fraud, 25), np.percentile(v12_fraud, 75)
v12_iqr = v12_75 - v12_25
v12_cut_off = v12_iqr * 1.5
v12_lower, v12_upper = v12_25 - v12_cut_off, v12_75 + v12_cut_off
new_df.drop(new_df[(new_df['V12'] < v12_lower) | (new_df['V12'] > v12_upper)].index, inplace=True)

# V10 removing outliers
v10_fraud = new_df['V10'].loc[new_df['Class'] == 1].values
v10_25, v10_75 = np.percentile(v10_fraud, 25), np.percentile(v10_fraud, 75)
v10_iqr = v10_75 - v10_25
v10_cut_off = v10_iqr * 1.5
v10_lower, v10_upper = v10_25 - v10_cut_off, v10_75 + v10_cut_off
new_df.drop(new_df[(new_df['V10'] < v10_lower) | (new_df['V10'] > v10_upper)].index, inplace=True)

# f, axes = plt.subplots(1, 3, figsize=(12, 24))
# sns.boxplot(x='Class',y='V14',data=new_df,ax=axes[0])
# sns.boxplot(x='Class',y='V12',data=new_df,ax=axes[1])
# sns.boxplot(x='Class',y='V10',data=new_df,ax=axes[2])
# plt.show()

X = new_df.drop('Class', axis=1)
y = new_df['Class']

#  T-SNE implementation
t0 = time.time()
X_decomposition_sne = TSNE(2, random_state=42).fit_transform(X.values)
t1 = time.time()
print('TSNE took {}'.format(round(t1 - t0, 4)))

# PCA implementation
t2 = time.time()
X_decomposition_pca = PCA(2, random_state=42).fit_transform(X.values)
t3 = time.time()
print('PCA took {}'.format(round(t3 - t2, 4)))

# truncated SVD implementation
t4 = time.time()
X_decomposition_t_svd = TruncatedSVD(2, random_state=42).fit_transform(X.values)
t5 = time.time()
print('TruncatedSVD took {}'.format(round(t5 - t4, 4)))

# f,axes = plt.subplots(1,3,figsize=(24,6))
# blue_patch = mpatches.Patch(color='#0A0AFF', label='non-fraud')
# red_patch = mpatches.Patch(color='#AF0000', label='fraud')
# #T-SNE scatter map
# axes[0].scatter(x=X_decomposition_sne[:,0],y=X_decomposition_sne[:,1],c=(y==1),cmap='coolwarm',label='fraud')
# axes[0].scatter(x=X_decomposition_sne[:,0],y=X_decomposition_sne[:,1],c=(y==0),cmap='coolwarm',label='non-fraud')
# axes[0].legend(handles=[blue_patch,red_patch])
# # sns.scatterplot(x=X_decomposition_sne[:,0],y=X_decomposition_sne[:,1],hue=(y==0),ax=axes[0])
# # sns.scatterplot(x=X_decomposition_sne[:,0],y=X_decomposition_sne[:,1],hue=(y==1),ax=axes[0])
#
# #PCA scatter map
# axes[1].scatter(x=X_decomposition_pca[:,0],y=X_decomposition_pca[:,1],c=(y==0),cmap='coolwarm',label='non-fraud')
# axes[1].scatter(x=X_decomposition_pca[:,0],y=X_decomposition_pca[:,1],c=(y==1),cmap='coolwarm',label='fraud')
# axes[1].legend(handles=[blue_patch,red_patch])
#
# #truncated-svd
# axes[2].scatter(x=X_decomposition_t_svd[:,0],y=X_decomposition_t_svd[:,1],c=(y==0),cmap='coolwarm',label='non-fraud')
# axes[2].scatter(x=X_decomposition_t_svd[:,0],y=X_decomposition_t_svd[:,1],c=(y==1),cmap='coolwarm',label='fraud')
# axes[2].legend(handles=[blue_patch,red_patch])
# plt.show()

X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.2,random_state=42)
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values
# classfiers = {
#     'LR':LogisticRegression(random_state=42),
#     'SVC':SVC(random_state=42,),
#     'DT':DecisionTreeClassifier(random_state=42),
#     'RF':RandomForestClassifier(random_state=42)
# }

# for key,classfier in classfiers.items():
#     classfier.fit(X_train,y_train)
#     training_score = cross_val_score(classfier,X_train,y_train,cv=5)
#     print('Classfier: ',classfier.__class__.__name__,'Has a training score of',round(training_score.mean(),4)*100,'accuracy score')

# LogisticRegression gridsearch
log_reg_para = {'C':[0.001,0.01,0.1,1,10,100,1000]}
grid_log_reg = GridSearchCV(LogisticRegression(random_state=42,max_iter=1000,n_jobs=-1),log_reg_para,n_jobs=-1)
grid_log_reg.fit(X_train,y_train)
log_reg =grid_log_reg.best_estimator_

# SVC GridSearch
svc_grid_para = {'C':[0.5,0.6,0.7,0.9,1],'kernel':['rbf','poly','linear','sigmoid']}
grid_svc = GridSearchCV(SVC(random_state=42),svc_grid_para,n_jobs=-1)
grid_svc.fit(X_train,y_train)
svc = grid_svc.best_estimator_

#DT GridSearch
dt_grid_para = {"criterion": ["gini", "entropy"], "max_depth": list(range(2,4,1)),
              "min_samples_leaf": list(range(5,7,1))}
grid_dt = GridSearchCV(DecisionTreeClassifier(random_state=42),dt_grid_para,n_jobs=-1)
grid_dt.fit(X_train,y_train)
dt = grid_dt.best_estimator_

#RF GridSearch
rf_grid_para = {"criterion": ["gini", "entropy"], "max_depth": list(range(2,4,1)),
              "min_samples_leaf": list(range(5,7,1))}
grid_rf = GridSearchCV(RandomForestClassifier(random_state=42),rf_grid_para,n_jobs=-1)
grid_rf.fit(X_train,y_train)
rf = grid_rf.best_estimator_

cross_val_score
log_reg_score = cross_val_score(log_reg,X_train,y_train,cv=5)
svc_score = cross_val_score(svc,X_train,y_train,cv=5)
dt_score = cross_val_score(dt,X_train,y_train,cv=5)
rf_score = cross_val_score(rf,X_train,y_train,cv=5)
print('lr:{}'.format(round(log_reg_score.mean()*100,2)))
print('svc:{}'.format(round(svc_score.mean()*100,2)))
print('dt:{}'.format(round(dt_score.mean()*100,2)))
print('rf:{}'.format(round(rf_score.mean()*100,2)))

# under sampling during cross validating
undersample_X = df.drop('Class', axis=1)
undersample_y = df['Class']
for train_index, test_index in sss.split(undersample_X, undersample_y):
    # print('Train:{}'.format(train_index),'Test:{}'.format(test_index))
    undersample_Xtrain, undersample_ytrain = undersample_X.iloc[train_index], undersample_y.iloc[train_index]
    undersample_Xtest, undersample_ytest = undersample_X.iloc[test_index], undersample_y.iloc[test_index]
# print(undersample_Xtrain.index)
undersample_Xtrain = undersample_Xtrain.values
undersample_ytrain = undersample_ytrain.values
undersample_Xtest = undersample_Xtest.values
undersample_ytest = undersample_ytest.values
# nm = NearMiss()
# X_nm,y_nm = nm.fit_sample(undersample_X.values,undersample_y.values)

# undersample_accuracy = []
# undersample_precision = []
# undersample_recall = []
# undersample_f1 = []
# undersample_auc = []
#
# for train, test in sss.split(undersample_Xtrain,undersample_ytrain):
#     undersample_pipeline = imblanced_make_pipeline(NearMiss(sampling_strategy='majority'),log_reg)
#     undersample_model = undersample_pipeline.fit(undersample_Xtrain[train],undersample_ytrain[train])
