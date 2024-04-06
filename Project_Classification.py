#!/usr/bin/env python
# coding: utf-8

# Objective:Developed a machine learning model to predict customer churn prediction using historical data.

# # Libraries

# In[3]:


from warnings import filterwarnings
filterwarnings('ignore')

import os
os.chdir("E:/ds/Class_material/ML/")

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split,RandomizedSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,f1_score,roc_auc_score,mean_squared_error,r2_score,mean_absolute_error

import matplotlib.pyplot as plt
import seaborn as sn


# # Data Collection

# In[4]:


df=pd.read_csv('e_commerce.csv')
df


# # EDA

# In[5]:


df.info()


# In[6]:


sn.histplot(df['total product detail views'],bins=10,kde=True)


# In[7]:


sn.histplot(df['product detail view per app session'],bins=10,kde=True)


# In[8]:


sn.histplot(df['customer service calls'],bins=10,kde=True)


# In[9]:


df.isna().sum()


# # Define X and Y

# In[10]:


x=df.drop(['churn'],axis=1)
y=df['churn']


# In[11]:


x


# In[12]:


y


# # Preprocessing

# In[13]:


cat=[]
con=[]

for i in x.columns:
    if x[i].dtypes=='object':
        cat.append(i)
    else:
        con.append(i)


# In[14]:


cat


# In[15]:


con


# In[ ]:





# In[16]:


num_pipe=Pipeline(steps=[('impute',SimpleImputer(strategy='median')),('scalar',StandardScaler())])
cat_pipe=Pipeline(steps=[('impute',SimpleImputer(strategy='most_frequent')),('encode',OrdinalEncoder())])


# In[17]:


pre=ColumnTransformer([('cat_pipe',cat_pipe,cat),('con_pipe',num_pipe,con)])


# In[18]:


x1=pd.DataFrame(pre.fit_transform(x),columns=pre.get_feature_names_out())
x1


# # Spliting dataset

# In[19]:


x_train,x_test,y_train,y_test=train_test_split(x1,y,test_size=0.2,random_state=21)


# # Algorithm Evaluation

# In[20]:


lr=LogisticRegression()
dt=DecisionTreeClassifier()
rf=RandomForestClassifier()
ab=AdaBoostClassifier()
knn=KNeighborsClassifier()
svm=SVC()


# In[21]:


List1=[lr,dt,rf,ab,knn,svm]


# In[22]:


for i in List1:
    i.fit(x_train,y_train)
    
    y_pred_train=i.predict(x_train)
    y_pred=i.predict(x_test)
    
    tr=f1_score(y_pred_train,y_train)
    ts=f1_score(y_pred,y_test)
    
    print('*'*50)
    print(i)
    print('Training F1 score: ',tr)
    print('Testing F1 score: ',ts)
    


# # For Random Forest

# ## Evaluation on Traning data

# In[23]:


y_pred_train=rf.predict(x_train)

acc=accuracy_score(y_pred_train,y_train)
cnf=confusion_matrix(y_pred_train,y_train)
clf=classification_report(y_pred_train,y_train)

print('Accuracy_score:',acc)
print('Confusion matrix:\n',cnf)
print('Classification REport:\n',clf)


# ## Evaluation on Testing data

# In[24]:


y_pred=rf.predict(x_test)

acc1=accuracy_score(y_pred,y_test)
cnf1=confusion_matrix(y_pred,y_test)
clf1=classification_report(y_pred,y_test)

print('Accuracy_score:',acc1)
print('Confusion matrix:\n',cnf1)
print('Classification REport:\n',clf1)


# # Hyper tuning for best Algorithm

# In[25]:


grid={
    'n_estimators':range(1,300),
    'criterion':['gini','entropy'],
    
    'max_depth':range(1,300),
    'min_samples_split':range(1,12),
    'min_samples_leaf':range(1,12)
}


# In[26]:


rs=RandomizedSearchCV(rf,param_distributions=grid,cv=3)
rs.fit(x_train,y_train)


# In[27]:


rs.best_params_


# In[28]:


rf1=rs.best_estimator_
rf1


# # Training data Evaluation

# In[29]:


y_pred_train=rf1.predict(x_train)

acc=accuracy_score(y_pred_train,y_train)
cnf=confusion_matrix(y_pred_train,y_train)
clf=classification_report(y_pred_train,y_train)

print('Accuracy_score:',acc)
print('Confusion matrix:\n',cnf)
print('Classification REport:\n',clf)


# # Testing data Evaluation

# In[30]:


y_pred=rf1.predict(x_test)

acc1=accuracy_score(y_pred,y_test)
cnf1=confusion_matrix(y_pred,y_test)
clf1=classification_report(y_pred,y_test)

print('Accuracy_score:',acc1)
print('Confusion matrix:\n',cnf1)
print('Classification REport:\n',clf1)


# In[ ]:




