
# coding: utf-8

# In[363]:


import tensorflow as tf
import pandas as pd


# In[2]:


from sklearn import svm


# In[492]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, precision_score
from sklearn.metrics import precision_recall_curve


# In[145]:


from sklearn.model_selection import cross_val_score


# In[366]:


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse = False)


# In[23]:


data = pd.read_csv('zoo.data', header=None)


# In[376]:


print(data)


# In[236]:


data1 = data.copy()


# In[478]:


print(data.head())


# In[500]:


print(data.describe())


# In[408]:


X = data.iloc[:,1:-1]
y = data.iloc[:,-1]
X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=1, stratify=y)


# In[468]:


clf_linear= svm.SVC(kernel='linear', C=1)
clf_linear.fit(X_train,y_train)
clf_rbf = svm.SVC(kernel='rbf',C=1)
clf_rbf.fit(X_train,y_train)
clf_poly = svm.SVC(kernel='poly', C=1)
clf_poly.fit(X_train,y_train)
clf_sigmoid = svm.SVC(kernel='sigmoid', C=2.5)
clf_sigmoid.fit(X_train, y_train)


# In[469]:


print(clf_linear.score(X_test,y_test))


# In[470]:


print(clf_rbf.score(X_test, y_test))


# In[471]:


print(clf_poly.score(X_test, y_test))


# In[472]:


print(clf_sigmoid.score(X_test, y_test))


# In[473]:


print(clf_sigmoid.score(X_train, y_train))


# In[474]:


print(cross_val_score(clf_linear, X, y, cv=5))


# In[501]:


print(cross_val_score(clf_poly, X, y, cv=5))


# In[502]:


print(cross_val_score(clf_rbf, X, y, cv=5))


# In[503]:


print(cross_val_score(clf_sigmoid, X, y, cv=5))


# In[495]:


y_pred_lin = clf_linear.predict(X_test)
y_pred_poly = clf_poly.predict(X_test)
y_pred_rbf = clf_rbf.predict(X_test)
y_pred_sigmoid = clf_sigmoid.predict(X_test)


# In[496]:


precision_score(y_test, y_pred,labels=[1,2,3,4,5,6,7], average='micro')


# In[498]:


p_score_lin=precision_score(y_test, y_pred_lin,labels=[1,2,3,4,5,6,7], average='micro')
p_score_poly=precision_score(y_test, y_pred_poly,labels=[1,2,3,4,5,6,7], average='micro')
p_score_rbf=precision_score(y_test, y_pred_rbf,labels=[1,2,3,4,5,6,7], average='micro')
p_score_sigmoid=precision_score(y_test, y_pred_sigmoid,labels=[1,2,3,4,5,6,7], average='micro')


# In[499]:


print(p_score_lin)
print(p_score_poly)
print(p_score_rbf)
print(p_score_sigmoid)

