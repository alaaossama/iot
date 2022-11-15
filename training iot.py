#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set()
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import recall_score
from sklearn.model_selection import StratifiedKFold
kFold = StratifiedKFold(n_splits=5)
from sklearn import metrics


# In[2]:


data = pd.read_csv('arrhythmia.csv',header=None)
data.head()


# In[3]:


data.tail()


# In[4]:


data.shape


# In[5]:


data.info()


# In[6]:


data.describe().transpose().head()


# In[7]:


#Let's see how many missing data we have and replace them with NaN's:
count=0
for i in range(0,452):
    for j in range(0,280):
        if (data.iloc[i,j]=='?'):
            count =count+1
print(count)
data = data.replace('?', np.NaN)


# In[8]:


#Now let's see the distribution of our missing data: 
pd.isnull(data).sum().plot()
plt.xlabel('Attributes')
plt.ylabel('Count of NaN')


# In[9]:


#zooming in :
pd.isnull(data).sum()[5:25].plot(kind='bar')
plt.xlabel('Attributes')
plt.ylabel('Count of NaN')


# In[10]:


#dropping column 13
data.drop(columns = 13, inplace=True)


# In[11]:


import fancyimpute
data_no_missing = fancyimpute.KNN(k=5).complete(data)


# In[12]:


data_no_missing=pd.DataFrame(data_no_missing)
data_no_missing.head()


# In[13]:


#Adding column names
X_columns=["Age","Sex","Height","Weight","QRS_Dur",
"P-R_Int","Q-T_Int","T_Int","P_Int","QRS","T","P","J","Heart_Rate",
"Q_Wave","R_Wave","S_Wave","R'_Wave","S'_Wave","Int_Def","Rag_R_Nom",
"Diph_R_Nom","Rag_P_Nom","Diph_P_Nom","Rag_T_Nom","Diph_T_Nom", 
"DII00", "DII01","DII02", "DII03", "DII04","DII05","DII06","DII07","DII08","DII09","DII10","DII11",
"DIII00","DIII01","DIII02", "DIII03", "DIII04","DIII05","DIII06","DIII07","DIII08","DIII09","DIII10","DIII11",
"AVR00","AVR01","AVR02","AVR03","AVR04","AVR05","AVR06","AVR07","AVR08","AVR09","AVR10","AVR11",
"AVL00","AVL01","AVL02","AVL03","AVL04","AVL05","AVL06","AVL07","AVL08","AVL09","AVL10","AVL11",
"AVF00","AVF01","AVF02","AVF03","AVF04","AVF05","AVF06","AVF07","AVF08","AVF09","AVF10","AVF11",
"V100","V101","V102","V103","V104","V105","V106","V107","V108","V109","V110","V111",
"V200","V201","V202","V203","V204","V205","V206","V207","V208","V209","V210","V211",
"V300","V301","V302","V303","V304","V305","V306","V307","V308","V309","V310","V311",
"V400","V401","V402","V403","V404","V405","V406","V407","V408","V409","V410","V411",
"V500","V501","V502","V503","V504","V505","V506","V507","V508","V509","V510","V511",
"V600","V601","V602","V603","V604","V605","V606","V607","V608","V609","V610","V611",
"JJ_Wave","Amp_Q_Wave","Amp_R_Wave","Amp_S_Wave","R_Prime_Wave","S_Prime_Wave","P_Wave","T_Wave",
"QRSA","QRSTA","DII170","DII171","DII172","DII173","DII174","DII175","DII176","DII177","DII178","DII179",
"DIII180","DIII181","DIII182","DIII183","DIII184","DIII185","DIII186","DIII187","DIII188","DIII189",
"AVR190","AVR191","AVR192","AVR193","AVR194","AVR195","AVR196","AVR197","AVR198","AVR199",
"AVL200","AVL201","AVL202","AVL203","AVL204","AVL205","AVL206","AVL207","AVL208","AVL209",
"AVF210","AVF211","AVF212","AVF213","AVF214","AVF215","AVF216","AVF217","AVF218","AVF219",
"V1220","V1221","V1222","V1223","V1224","V1225","V1226","V1227","V1228","V1229",
"V2230","V2231","V2232","V2233","V2234","V2235","V2236","V2237","V2238","V2239",
"V3240","V3241","V3242","V3243","V3244","V3245","V3246","V3247","V3248","V3249",
"V4250","V4251","V4252","V4253","V4254","V4255","V4256","V4257","V4258","V4259",
"V5260","V5261","V5262","V5263","V5264","V5265","V5266","V5267","V5268","V5269",
"V6270","V6271","V6272","V6273","V6274","V6275","V6276","V6277","V6278","V6279"]


# In[14]:


X = data_no_missing.drop(columns = 278)
X.head()


# In[15]:


X.columns = X_columns
X.head()


# In[16]:


y = data[279]
y.head()


# In[17]:


y.columns = ["Class"]
y.head()


# In[18]:


g = sns.PairGrid(X, vars=['Age', 'Sex', 'Height', 'Weight'],
                 hue='Sex', palette='RdBu_r')
g.map(plt.scatter, alpha=0.8)
g.add_legend();


# According to scatter plots, there are few outliers in 'height' and 'weight' attributes.
# I'll check the maximums of heights and weights

# In[19]:


sorted(X['Height'], reverse=True)[:10]


# The tallest person ever lived in the world was **272** cm (1940). His followers were **267** cm(1905) and **263.5** cm(1969) 
# 
# Replacing **780** and **608** with **108** and **180** cm

# In[20]:


X['Height']=X['Height'].replace(608,108)
X['Height']=X['Height'].replace(780,180)


# In[21]:


sorted(X['Weight'], reverse=True)[:10]


# Looks like **176** kgs is a possible weight. I'll keep them.

# In[22]:


g = sns.PairGrid(X, vars=['Age', 'Sex', 'Height', 'Weight'],
                 hue='Sex', palette='RdBu_r')
g.map(plt.scatter, alpha=0.8)
g.add_legend();


# In[23]:


sns.boxplot(data=X[["QRS_Dur","P-R_Int","Q-T_Int","T_Int","P_Int"]]);
#sns.swarmplot(data=X[["QRS_Dur","P-R_Int","Q-T_Int","T_Int","P_Int"]]);


# **PR interval** is the period, measured in milliseconds, that extends from the beginning of the P wave  until the beginning of the QRS complex; it is normally between **120 and 200ms** in duration. 

# In[24]:


X['P-R_Int'].value_counts().sort_index().head().plot(kind='bar')
plt.xlabel('P-R Interval Values')
plt.ylabel('Count');


# In[25]:


X['P-R_Int'].value_counts().sort_index().tail().plot(kind='bar')
plt.xlabel('P-R Interval Values')
plt.ylabel('Count');


# #### PR Interval data is including outliers 0(x18). I'll keep them

# QT interval is a measure of the time between the start of the Q wave and the end of the T wave in the heart's electrical cycle. The outlier data appearing in Q-T Interval box might be related to the outlier of T-interval data.

# In[26]:


X['T_Int'].value_counts().sort_index(ascending=False).head().plot(kind='bar')
plt.xlabel('T Interval Values')
plt.ylabel('Count');


# In[27]:


X['T_Int'].value_counts().sort_index(ascending=False).tail().plot(kind='bar')
plt.xlabel('T Interval Values')
plt.ylabel('Count');


# In[28]:


sns.boxplot(data=X[["QRS","T","P","J","Heart_Rate"]]);
#sns.swarmplot(data=X[["QRS_Dur","P-R_Int","Q-T_Int","T_Int","P_Int"]]);


# In[29]:


sns.boxplot(data=X[["Q_Wave","R_Wave","S_Wave"]])
sns.swarmplot(data=X[["Q_Wave","R_Wave","S_Wave"]]);


# In[30]:


sns.boxplot(data=X[["R'_Wave","S'_Wave","Int_Def","Rag_R_Nom"]]);
#sns.swarmplot(data=X[["R'_Wave","S'_Wave","Int_Def","Rag_R_Nom"]])


# #### S'Wave has 0's which is not a NaN. So, we can't assume it as including outliers.

# In[31]:


X["R'_Wave"].value_counts().sort_index(ascending=False)


# In[32]:


X["S'_Wave"].value_counts().sort_index(ascending=False)


# In[33]:


X["Rag_R_Nom"].value_counts().sort_index(ascending=False)


# In[34]:


sns.set(rc={'figure.figsize':(11.7,5.27)})
sns.boxplot(data=X[["Diph_R_Nom","Rag_P_Nom","Diph_P_Nom","Rag_T_Nom","Diph_T_Nom"]]);


# In[35]:


X["Diph_R_Nom"].value_counts().sort_index(ascending=False)


# In[36]:


X["Rag_P_Nom"].value_counts().sort_index(ascending=False)


# In[37]:


X["Diph_P_Nom"].value_counts().sort_index(ascending=False)


# In[38]:


X["Rag_T_Nom"].value_counts().sort_index(ascending=False)


# In[39]:


X["Diph_T_Nom"].value_counts().sort_index(ascending=False)


# In[40]:


sns.set(rc={'figure.figsize':(11.7,5.27)})
sns.boxplot(data=X[["DII00", "DII01","DII02", "DII03", "DII04","DII05","DII06","DII07","DII08","DII09","DII10","DII11"]]);


# In[41]:


sns.set(rc={'figure.figsize':(11.7,5.27)})
sns.boxplot(data=X[["DIII00","DIII01","DIII02", "DIII03", "DIII04","DIII05","DIII06",
                       "DIII07","DIII08","DIII09","DIII10","DIII11"]]);


# In[42]:


sns.set(rc={'figure.figsize':(11.7,5.27)})
sns.boxplot(data=X[["AVR00","AVR01","AVR02","AVR03","AVR04","AVR05",
                       "AVR06","AVR07","AVR08","AVR09","AVR10","AVR11"]]);


# In[43]:


sns.set(rc={'figure.figsize':(11.7,5.27)})
sns.boxplot(data=X[["AVL00","AVL01","AVL02","AVL03","AVL04","AVL05","AVL06","AVL07","AVL08","AVL09","AVL10","AVL11"]]);


# In[44]:


sns.set(rc={'figure.figsize':(11.7,5.27)})
sns.boxplot(data=X[["AVF00","AVF01","AVF02","AVF03","AVF04","AVF05","AVF06","AVF07","AVF08","AVF09","AVF10","AVF11"]]);


# In[45]:


sns.set(rc={'figure.figsize':(11.7,5.27)})
sns.boxplot(data=X[["V100","V101","V102","V103","V104","V105","V106","V107","V108","V109","V110","V111"]]);


# In[46]:


X["V101"].value_counts().sort_index(ascending=False)


# **V101** has an outlier, but when we look at other sets (V201, V301, V501) we can see that there's an outlier similarly. Since our data is heavily biased, I can't say these outliers should be dropped. 
# 
# For example, when we look at our data, we can see that class # 8 (Supraventricular Premature Contraction) **has only 2 instances**. Or # 3 (Ventricular Premature Contraction (PVC)) has only 3. The outliers appearing with our plots might belong to these instances and needs to be kept.

# In[47]:


sns.set(rc={'figure.figsize':(11.7,5.27)})
sns.boxplot(data=X[["V200","V201","V202","V203","V204","V205","V206","V207","V208","V209","V210","V211"]]);


# In[48]:


sns.set(rc={'figure.figsize':(11.7,5.27)})
sns.boxplot(data=X[["V300","V301","V302","V303","V304","V305","V306","V307","V308","V309","V310","V311"]]);


# In[49]:


sns.set(rc={'figure.figsize':(11.7,5.27)})
sns.boxplot(data=X[["V400","V401","V402","V403","V404","V405","V406","V407","V408","V409","V410","V411"]]);


# In[50]:


sns.set(rc={'figure.figsize':(11.7,5.27)})
sns.boxplot(data=X[["V500","V501","V502","V503","V504","V505","V506","V507","V508","V509","V510","V511"]]);


# In[51]:


sns.set(rc={'figure.figsize':(11.7,5.27)})
sns.boxplot(data=X[["V600","V601","V602","V603","V604","V605","V606","V607","V608","V609","V610","V611"]]);


# In[52]:


sns.set(rc={'figure.figsize':(11.7,5.27)})
sns.boxplot(data=X[["JJ_Wave","Amp_Q_Wave","Amp_R_Wave","Amp_S_Wave","R_Prime_Wave","S_Prime_Wave","P_Wave","T_Wave"]]);


# In[53]:


sns.set(rc={'figure.figsize':(13.7,5.27)})
sns.boxplot(data=X[["QRSA","QRSTA","DII170","DII171","DII172","DII173","DII174","DII175","DII176","DII177","DII178","DII179"]]);


# In[54]:


sns.set(rc={'figure.figsize':(11.7,5.27)})
sns.boxplot(data=X[["DIII180","DIII181","DIII182","DIII183","DIII184","DIII185","DIII186","DIII187","DIII188","DIII189"]]);


# Now we can see outlier within the last two attributes of each series(DIII188, DIII189, AVR198, AVR199, AVL208, AVL209, AVF218, AVF219, V2238, V2239, V3248, V3249,V4258, V4259,V5268, V5269, V6278, V6279). Similiarly assuming that these outliers might belong to the classes with few instances.

# In[55]:


sns.set(rc={'figure.figsize':(11.7,5.27)})
sns.boxplot(data=X[["AVR190","AVR191","AVR192","AVR193","AVR194","AVR195","AVR196","AVR197","AVR198","AVR199"]]);


# In[56]:


sns.set(rc={'figure.figsize':(11.7,5.27)})
sns.boxplot(data=X[["AVL200","AVL201","AVL202","AVL203","AVL204","AVL205","AVL206","AVL207","AVL208","AVL209"]]);


# In[57]:


sns.set(rc={'figure.figsize':(11.7,5.27)})
sns.boxplot(data=X[["AVF210","AVF211","AVF212","AVF213","AVF214","AVF215","AVF216","AVF217","AVF218","AVF219"]]);


# In[58]:


sns.set(rc={'figure.figsize':(11.7,5.27)})
sns.boxplot(data=X[["V1220","V1221","V1222","V1223","V1224","V1225","V1226","V1227","V1228","V1229"]]);


# In[59]:


sns.set(rc={'figure.figsize':(11.7,5.27)})
sns.boxplot(data=X[["V2230","V2231","V2232","V2233","V2234","V2235","V2236","V2237","V2238","V2239"]]);


# In[60]:


sns.set(rc={'figure.figsize':(11.7,5.27)})
sns.boxplot(data=X[["V3240","V3241","V3242","V3243","V3244","V3245","V3246","V3247","V3248","V3249"]]);


# In[61]:


sns.set(rc={'figure.figsize':(11.7,5.27)})
sns.boxplot(data=X[["V4250","V4251","V4252","V4253","V4254","V4255","V4256","V4257","V4258","V4259"]]);


# In[62]:


sns.set(rc={'figure.figsize':(11.7,5.27)})
sns.boxplot(data=X[["V5260","V5261","V5262","V5263","V5264","V5265","V5266","V5267","V5268","V5269"]]);


# In[63]:


sns.set(rc={'figure.figsize':(11.7,5.27)})
sns.boxplot(data=X[["V6270","V6271","V6272","V6273","V6274","V6275","V6276","V6277","V6278","V6279"]]);


# ### Train Test Split

# In[64]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state = 10)


# In[65]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ## Choosing ML Models and metrics

# My intention is to try all applicable ML models in order to practice my knowledge.
# 
# Thinking about the classification evaluation metrics, the importance of my models' precitions (I can't accept a result having the probability of saying to a healty person that you have Cardiac Arrhythmia (FN)). 
# 
# I definitely will focus on **Sensitivity** (the percentage of sick people who are correctly identified as having the condition) not Specificity (percentage of healthy people who are correctly identified as not having the condition). 
# 
# ![image.png](attachment:image.png)
# 
# So, I'll use recall for my models and sklearn has a **"weighted"** metric which accounts for class imbalance by computing the average of metrics in which each class’s score is weighted by its presence in the true data sample.
# 
# I'll use GridSearchCV to obtain the best parameters for each model and get my results by applying those parameters.
# 

# ### KNN Clasification

# In[66]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

knn_clf = KNeighborsClassifier()

param_grid = {'n_neighbors' : [1,2,3,4,5,7,10]}

grid_search = GridSearchCV (knn_clf, param_grid, cv=kFold,scoring = 'recall_weighted', return_train_score=True)

grid_search.fit(X_train, y_train)


# In[67]:


grid_search.best_params_


# In[68]:


grid_search.best_score_


# In[69]:


knn_clf = KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(X_train, y_train)


# In[70]:


y_pred_train = knn_clf.predict(X_train)
y_pred_test = knn_clf.predict(X_test)

knn_train_recall_score = recall_score(y_train, y_pred_train, average='weighted')
knn_test_recall_score = recall_score(y_test, y_pred_test, average='weighted')

print('Train Recall score: {}'
      .format(knn_train_recall_score))
print('Test Recall score: {}'
      .format(knn_test_recall_score))

metrics.confusion_matrix(y_test, y_pred_test)


# Model fit well but KNN results are not good. 

# ### Logistic Regression

# In[71]:


from sklearn.linear_model import LogisticRegression

lreg_clf = LogisticRegression(multi_class='multinomial', solver='lbfgs')

param_grid = {'C': [0.00001,0.0001,0.001,0.01,0.1,1,10,100]}

grid_search = GridSearchCV(lreg_clf, param_grid, scoring = 'recall_weighted',cv=kFold, return_train_score=True)
grid_search.fit(X_train,y_train)


# In[72]:


grid_search.best_params_


# In[73]:


grid_search.best_score_


# In[74]:


lreg_clf= LogisticRegression(multi_class='multinomial', solver='lbfgs', C=1)
lreg_clf.fit(X_train, y_train)


# In[75]:


y_pred_train = lreg_clf.predict(X_train)
y_pred_test = lreg_clf.predict(X_test)

lreg_train_recall_score = recall_score(y_train, y_pred_train, average='weighted')
lreg_test_recall_score = recall_score(y_test, y_pred_test, average='weighted')
print('Train Recall score: {}'
      .format(lreg_train_recall_score))
print('Test Recall score: {}'
      .format(lreg_test_recall_score))

metrics.confusion_matrix(y_test, y_pred_test)


# The large difference between train and test scores showing that the model is overfitting and low test score is also showing that Logistic Regression doesn't perform well.

# ### Linear SVM

# In[76]:


from sklearn.svm import LinearSVC

LSVC_clf = LinearSVC(multi_class='crammer_singer')

param_grid = {'C': [0.00001,0.0001,0.001,0.01,0.1,1,10,100]}

grid_search = GridSearchCV(LSVC_clf, param_grid, scoring = 'recall_weighted',cv=kFold, return_train_score=True)
grid_search.fit(X_train,y_train)


# In[77]:


grid_search.best_params_


# In[78]:


grid_search.best_score_


# In[79]:


LSVC_clf = LinearSVC(multi_class='crammer_singer', C=0.1)
LSVC_clf.fit(X_train, y_train)


# In[80]:


y_pred_train = LSVC_clf.predict(X_train)
y_pred_test = LSVC_clf.predict(X_test)

lsvc_train_recall_score = recall_score(y_train, y_pred_train, average='weighted')
lsvc_test_recall_score = recall_score(y_test, y_pred_test, average='weighted')

print('Train Recall score: {}'
      .format(lsvc_train_recall_score))
print('Test Recall score: {}'
      .format(lsvc_test_recall_score))

metrics.confusion_matrix(y_test, y_pred_test)


# Test score is better than KNN and Logistic Regression. Also model is good fit as there is not much difference between train and test score.

# ### Kernelized SVM

# In[81]:


from sklearn import svm

KSVC_clf = svm.SVC(kernel='rbf')

param_grid = {'C': [0.0001,0.001,0.01,0.1,1,10],
          'gamma': [0.0001,0.001,0.01,0.1,1,10]}

grid_search = GridSearchCV(KSVC_clf, param_grid, scoring = 'recall_weighted',cv=kFold, return_train_score=True)
grid_search.fit(X_train,y_train)


# In[82]:


grid_search.best_params_


# In[83]:


grid_search.best_score_


# In[84]:


KSVC_clf = svm.SVC(kernel='rbf',C=10,gamma=0.1)

KSVC_clf.fit(X_train, y_train)


# In[85]:


y_pred_train = KSVC_clf.predict(X_train)
y_pred_test = KSVC_clf.predict(X_test)

ksvc_train_recall_score = recall_score(y_train, y_pred_train, average='weighted')
ksvc_test_recall_score = recall_score(y_test, y_pred_test, average='weighted')

print('Train Recall score: {}'
      .format(ksvc_train_recall_score))
print('Test Recall score: {}'
      .format(ksvc_test_recall_score))

metrics.confusion_matrix(y_test, y_pred_test)


# We can see that test score is poor and Kernalised SVM doesn't perform well. Also model is overfitting as there is large difference between train and test score.

# ### Naive Bayes

# In[86]:


from sklearn.naive_bayes import MultinomialNB

mnb_clf = MultinomialNB()
param_grid = {'alpha':[0,1.0,10], 'fit_prior':(True, False)}

grid_search = GridSearchCV(mnb_clf, param_grid,n_jobs=-1)
grid_search.fit(X_train,y_train)


# In[87]:


grid_search.best_params_


# In[88]:


grid_search.best_score_


# In[89]:


mnb_clf = MultinomialNB(alpha=0, fit_prior=True)

mnb_clf.fit(X_train, y_train)


# In[90]:


y_pred_train = mnb_clf.predict(X_train)
y_pred_test = mnb_clf.predict(X_test)

mnb_train_recall_score = recall_score(y_train, y_pred_train, average='weighted')
mnb_test_recall_score = recall_score(y_test, y_pred_test, average='weighted')

print('Train Recall score: {}'
      .format(mnb_train_recall_score))
print('Test Recall score: {}'
      .format(mnb_test_recall_score))

metrics.confusion_matrix(y_test, y_pred_test)


# ### Decision Tree

# In[91]:


from sklearn.tree import DecisionTreeClassifier

dt_clf = DecisionTreeClassifier()
param_grid = {'max_depth': [2, 3, 4, 5,6, 10, 20]}

grid_search = GridSearchCV(dt_clf, param_grid, scoring = 'recall_weighted',cv=kFold, return_train_score=True)
grid_search.fit(X_train,y_train)


# In[92]:


grid_search.best_params_


# In[93]:


grid_search.best_score_


# In[94]:


dt_clf = DecisionTreeClassifier(max_depth=6)
dt_clf.fit(X_train, y_train)


# In[95]:


y_pred_train = dt_clf.predict(X_train)
y_pred_test = dt_clf.predict(X_test)

dt_train_recall_score = recall_score(y_train, y_pred_train, average='weighted')
dt_test_recall_score = recall_score(y_test, y_pred_test, average='weighted')

print('Train Recall score: {}'
      .format(dt_train_recall_score))
print('Test Recall score: {}'
      .format(dt_test_recall_score))

metrics.confusion_matrix(y_test, y_pred_test)


# Test score is poor and decision tree doesn't perform well. Also model is somewhat overfitting as there is difference between train and test score.

# ## Random Forest

# In[96]:


from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators=300, criterion='gini',n_jobs= -1,random_state=10)

param_grid = {'max_features': [100,125,150,200],
          'max_depth': [6,8,10,12,14],
           'max_leaf_nodes':[20,22,30,50]}

grid_search = GridSearchCV(rf_clf, param_grid, scoring = 'recall_weighted',cv=kFold, return_train_score=True)
grid_search.fit(X_train,y_train)


# In[97]:


grid_search.best_params_


# In[98]:


grid_search.best_score_


# In[99]:


rf_clf = RandomForestClassifier(n_estimators=300, criterion='gini',max_features=100,max_depth=10,max_leaf_nodes=30)
rf_clf.fit(X_train, y_train)


# In[100]:


y_pred_train = rf_clf.predict(X_train)
y_pred_test = rf_clf.predict(X_test)

rf_train_recall_score = recall_score(y_train, y_pred_train, average='weighted')
rf_test_recall_score = recall_score(y_test, y_pred_test, average='weighted')

print('Train Recall score: {}'
      .format(rf_train_recall_score))
print('Test Recall score: {}'
      .format(rf_test_recall_score))

metrics.confusion_matrix(y_test, y_pred_test)


# Test score is good and Random forest performs well as it is ensemble method. But model is overfitting as there is large difference between train and test score.

# In[101]:


from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score

hard_voting_clf = VotingClassifier(estimators=[('knn', knn_clf),('lr',lreg_clf),('lsvc', LSVC_clf),
                                   ('ksvc', KSVC_clf),('dt', dt_clf), ('rt', rf_clf)],voting = 'hard')
hard_voting_clf.fit(X_train, y_train)
print('Train score: {0:0.2f}'.format(hard_voting_clf.score(X_train, y_train)))
print('Test score: {0:0.2f}'.format(hard_voting_clf.score(X_test, y_test)))


# In[102]:


score = cross_val_score(estimator=hard_voting_clf,X=X_train,y=y_train, scoring='recall_weighted', cv=kFold)


# In[103]:


print('Mean Score: {0:0.2f}'.format(score.mean()))
print('Mean Std: {0:0.2f}'.format(score.std()))


# In[104]:


X_scaled = scaler.fit_transform(X)


# ### Bagging with KNN¶

# In[105]:


from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier

#knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
bag_knn = BaggingClassifier(base_estimator=knn_clf, n_estimators=100,bootstrap_features=True, bootstrap=False,
                            max_samples=50, max_features=100)

score = cross_val_score(estimator=bag_knn, X=X_scaled, y=y, scoring='recall_weighted', cv=kFold, n_jobs=-1)


# In[106]:


print('Mean score:', score.mean())


# Bagging is giving us a very low score. It doesn't improve our model

# ### Pasting with KNN

# In[107]:


from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier

#knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
bag_knn = BaggingClassifier(base_estimator=knn_clf, n_estimators=100,bootstrap_features=True, bootstrap=False,
                            max_samples=50, max_features=100)

score = cross_val_score(estimator=bag_knn, X=X_scaled, y=y, scoring='recall_weighted', cv=kFold, n_jobs=-1)


# In[108]:


print('Mean score:', score.mean())


# Pasting is also giving us a very low score. It doesn't improve our model

# ### Bagging with Logistic Reg

# In[109]:


from sklearn.ensemble import BaggingClassifier

bag_log = BaggingClassifier(base_estimator=lreg_clf, n_estimators=100,bootstrap_features=True, max_samples=50, max_features=100)

score = cross_val_score(estimator=bag_log, X=X_scaled, y=y, scoring='recall_weighted', cv=kFold, n_jobs=-1)
print('Mean score:', score.mean())


# ### Bagging with Linear SVC

# In[110]:


bag_lsvc = BaggingClassifier(base_estimator=LSVC_clf, n_estimators=100,bootstrap_features=True, max_samples=50, max_features=100)

score = cross_val_score(estimator=bag_lsvc, X=X_scaled, y=y, scoring='recall_weighted', cv=kFold, n_jobs=-1)
print('Mean score:', score.mean())


# The bagging algorithm has raised the bias and doesn't improve our model

# ### Bagging with SVM

# In[111]:


bag_ksvc = BaggingClassifier(base_estimator=KSVC_clf, n_estimators=100,bootstrap_features=True, max_samples=50, max_features=100)

score = cross_val_score(estimator=bag_ksvc, X=X_scaled, y=y, scoring='recall_weighted', cv=kFold, n_jobs=-1)
print('Mean score:', score.mean())


# The bagging algorithm has raised the bias and doesn't improve our model

# ### Bagging with Decision Tree

# In[112]:


bag_dt = BaggingClassifier(base_estimator=dt_clf, n_estimators=100,bootstrap_features=True, max_samples=50, max_features=100)

score = cross_val_score(estimator=bag_dt, X=X_scaled, y=y, scoring='recall_weighted', cv=kFold, n_jobs=-1)
print('Mean score:', score.mean())


# The bagging algorithm has raised the bias and doesn't improve our model

# ### Bagging with Random Forest

# In[113]:


bag_rf = BaggingClassifier(base_estimator=rf_clf, n_estimators=100,bootstrap_features=True, max_samples=50, max_features=100)

score = cross_val_score(estimator=bag_rf, X=X_scaled, y=y, scoring='recall_weighted', cv=kFold, n_jobs=-1)
print('Mean score:', score.mean())


# The bagging algorithm has raised the bias and doesn't improve our model

# ### Adaptive boosting with decision tree classifier

# In[114]:


from sklearn.ensemble import AdaBoostClassifier

adaboost_clf = AdaBoostClassifier(base_estimator = dt_clf, learning_rate = 0.5)
adaboost_clf.fit(X_train, y_train)
print('Train score: {0:0.2f}'.format(adaboost_clf.score(X_train, y_train)))
print('Test score: {0:0.2f}'.format(adaboost_clf.score(X_test, y_test)))


# Adaptive boosting did in fact raise the average training accuracy for the Decision Tree but the test accuracy got reduced

# ### AdBoosting with Random Forest

# In[115]:


from sklearn.ensemble import AdaBoostClassifier
adaboost_clf = AdaBoostClassifier(base_estimator = rf_clf, learning_rate = 0.5)
adaboost_clf.fit(X_train, y_train)
print('Train score: {0:0.2f}'.format(adaboost_clf.score(X_train, y_train)))
print('Test score: {0:0.2f}'.format(adaboost_clf.score(X_test, y_test)))


# Adaptive boosting did in fact raise the average training accuracy for the Decision Tree but the test accuracy got reduced. It is still overfitting

# ### Adaptive Boosting with Logistic Regression

# In[116]:


from sklearn.ensemble import AdaBoostClassifier

adaboost_clf = AdaBoostClassifier(base_estimator = lreg_clf, learning_rate = 0.5)
adaboost_clf.fit(X_train, y_train)
print('Train score: {0:0.2f}'.format(adaboost_clf.score(X_train, y_train)))
print('Test score: {0:0.2f}'.format(adaboost_clf.score(X_test, y_test)))


# Adaptive boosting improved our test accuracy with logistic model and has an improved model fit

# ### Adaptive Boosting with LinearSVC

# In[117]:


from sklearn.ensemble import AdaBoostClassifier

adaboost_clf = AdaBoostClassifier(base_estimator = LSVC_clf, algorithm='SAMME')
adaboost_clf.fit(X_train, y_train)
print('Train score: {0:0.2f}'.format(adaboost_clf.score(X_train, y_train)))
print('Test score: {0:0.2f}'.format(adaboost_clf.score(X_test, y_test)))


# Adaptive boosting did in fact reduced the average test accuracy for the Decision Tree but the test accuracy got increased. Therefore, it has a poor fit with Linear SVC

# ### Adaptive Boosting with KNN

# KNeighborsClassifier does not support sample weights, so we will not be able to use Adaptive Boosting to lower the model bias.

# ### Adaptive Boosting with Kernel SVC

# In[118]:


from sklearn.ensemble import AdaBoostClassifier

adaboost_clf = AdaBoostClassifier(base_estimator = KSVC_clf, algorithm='SAMME')
adaboost_clf.fit(X_train, y_train)
print('Train score: {0:0.2f}'.format(adaboost_clf.score(X_train, y_train)))
print('Test score: {0:0.2f}'.format(adaboost_clf.score(X_test, y_test)))


# Adaptive boosting has very low accuracy for Kernel SVC. So it's not a good fit

# ### Gradient Boosting

# In[119]:


from sklearn.ensemble import GradientBoostingClassifier

gb_clf = GradientBoostingClassifier(learning_rate = 0.05)
gb_clf.fit(X_train, y_train)
print('Train score: {0:0.2f}'.format(gb_clf.score(X_train, y_train)))
print('Test score: {0:0.2f}'.format(gb_clf.score(X_test, y_test)))


# Gradient Boosting is overfitting the model

# In[120]:


GB = GradientBoostingClassifier()
score = cross_val_score(estimator=GB, X=X_scaled, y=y, cv=kFold, n_jobs=-1)
gb_clf.fit(X_train, y_train)
print('Train score: {0:0.2f}'.format(gb_clf.score(X_train, y_train)))
print('Test score: {0:0.2f}'.format(gb_clf.score(X_test, y_test)))
print('Mean Accuracy:', score.mean())


# Gradient Boosting is overfitting the model

# In[121]:


GB2 = GradientBoostingClassifier(min_samples_leaf=9, learning_rate=0.05, n_estimators=100)
score = cross_val_score(estimator=GB, X=X_scaled, y=y, cv=kFold, n_jobs=-1)
gb_clf.fit(X_train, y_train)
print('Train score: {0:0.2f}'.format(gb_clf.score(X_train, y_train)))
print('Test score: {0:0.2f}'.format(gb_clf.score(X_test, y_test)))
print('Mean Accuracy:', score.mean())


# 
# It is clear that, while the bagging and boosting techniques mentioned above are
# usually effective, most did not do much to improve the models.
#     Due to reasons like imbalanced classes, high dimensionality and lack of observations, we couldn't get an optimal model.
# 
# 

# In[122]:


from sklearn.decomposition import PCA

pca = PCA(n_components=100, svd_solver='auto')
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
X_comb_pca = np.concatenate((X_train_pca, X_test_pca), axis=0)


# In[123]:


X_train_pca.shape


# ### KNN Classification with PCA
# 

# In[124]:


from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier(n_jobs=-1)

param_grid={'n_neighbors':[1,2,3,4,5,7,10]}

grid_search = GridSearchCV(knn_clf, param_grid, scoring = 'recall_weighted',cv=kFold, return_train_score=True)
grid_search.fit(X_train_pca,y_train)


# In[125]:


grid_search.cv_results_


# In[126]:


grid_search.best_params_


# In[127]:


grid_search.best_score_


# In[128]:


knn_clf=KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(X_train_pca, y_train)


# In[129]:


y_pred_train = knn_clf.predict(X_train_pca)
y_pred_test = knn_clf.predict(X_test_pca)

knn_pca_train_recall_score = recall_score(y_train, y_pred_train, average='weighted')
knn_pca_test_recall_score = recall_score(y_test, y_pred_test, average='weighted')
print('Train Recall score: {}'
      .format(knn_pca_train_recall_score))
print('Test Recall score: {}'
      .format(knn_pca_test_recall_score))

metrics.confusion_matrix(y_test, y_pred_test)


# KNN shows an improved model accuracy and fit after applying PCA

# ### Logistic Regression with PCA

# In[130]:


from sklearn.linear_model import LogisticRegression

lreg_clf = LogisticRegression(multi_class='multinomial', solver='lbfgs')

param_grid = {'C': [0.00001,0.0001,0.001,0.01,0.1,1,10,100]}

grid_search = GridSearchCV(lreg_clf, param_grid, scoring = 'recall_weighted',cv=kFold, return_train_score=True)
grid_search.fit(X_train_pca,y_train)


# In[131]:


grid_search.cv_results_


# In[132]:


grid_search.best_params_


# In[133]:


grid_search.best_score_


# In[134]:


lreg_clf = LogisticRegression(multi_class='multinomial', solver='lbfgs',C=1)
lreg_clf.fit(X_train_pca, y_train)


# In[135]:


y_pred_train = lreg_clf.predict(X_train_pca)
y_pred_test = lreg_clf.predict(X_test_pca)

logreg_pca_train_recall_score = recall_score(y_train, y_pred_train, average='weighted')
logreg_pca_test_recall_score = recall_score(y_test, y_pred_test, average='weighted')
print('Train Recall score: {}'
      .format(logreg_pca_train_recall_score))
print('Test Recall score: {}'
      .format(logreg_pca_test_recall_score))

metrics.confusion_matrix(y_test, y_pred_test)


# Logistics Regression shows an improved model accuracy and fit after applying PCA.

# ### Linear SVM with PCA

# In[136]:


from sklearn.svm import LinearSVC

LSVC_clf = LinearSVC(multi_class='crammer_singer')

param_grid = {'C': [0.00001,0.0001,0.001,0.01,0.1,1,10,100]}

grid_search = GridSearchCV(LSVC_clf, param_grid, scoring = 'recall_weighted',cv=kFold, return_train_score=True)
grid_search.fit(X_train_pca,y_train)


# In[137]:


grid_search.cv_results_


# In[138]:


grid_search.best_params_


# In[139]:


grid_search.best_score_


# In[140]:


LSVC_clf = LinearSVC(multi_class='crammer_singer', C=0.1)
LSVC_clf.fit(X_train_pca, y_train)


# In[141]:


y_pred_train = LSVC_clf.predict(X_train_pca)
y_pred_test = LSVC_clf.predict(X_test_pca)

lscvc_pca_train_recall_score = recall_score(y_train, y_pred_train, average='weighted')
lscv_pca_test_recall_score = recall_score(y_test, y_pred_test, average='weighted')
print('Train Recall score: {}'
      .format(lscvc_pca_train_recall_score))
print('Test Recall score: {}'
      .format(lscv_pca_test_recall_score))

metrics.confusion_matrix(y_test, y_pred_test)


# Linear SVM still overfits the model after applying PCA.

# ### Kernalised SVM with PCA

# In[142]:


from sklearn import svm

KSVC_clf = svm.SVC(kernel='rbf')

param_grid = {'C': [0.0001,0.001,0.01,0.1,1,10],
          'gamma': [0.0001,0.001,0.01,0.1,1,10]}

grid_search = GridSearchCV(KSVC_clf, param_grid, scoring = 'recall_weighted',cv=kFold, return_train_score=True)
grid_search.fit(X_train_pca,y_train)


# In[143]:


grid_search.cv_results_


# In[144]:


grid_search.best_params_


# In[145]:


grid_search.best_score_


# In[146]:


from sklearn import svm
KSVC_clf = svm.SVC(kernel='rbf',C=10,gamma=0.1)
KSVC_clf.fit(X_train_pca, y_train)


# In[147]:


y_pred_train = KSVC_clf.predict(X_train_pca)
y_pred_test = KSVC_clf.predict(X_test_pca)

kscv_pca_train_recall_score = recall_score(y_train, y_pred_train, average='weighted')
kscv_pca_test_recall_score = recall_score(y_test, y_pred_test, average='weighted')
print('Train Recall score: {}'
      .format(kscv_pca_train_recall_score))
print('Test Recall score: {}'
      .format(kscv_pca_test_recall_score))

metrics.confusion_matrix(y_test, y_pred_test)


# Kernalized SVM still overfits the model after applying PCA.

# ### Decision Trees with PCA

# In[148]:


from sklearn.tree import DecisionTreeClassifier

dt_clf = DecisionTreeClassifier()
param_grid = {'max_depth': [2, 3, 4, 5,6, 10, 20]}

grid_search = GridSearchCV(dt_clf, param_grid, scoring = 'recall_weighted',cv=kFold, return_train_score=True)
grid_search.fit(X_train_pca,y_train)


# In[149]:


grid_search.cv_results_


# In[150]:


grid_search.best_params_


# In[151]:


grid_search.best_score_


# In[152]:


dt_clf = DecisionTreeClassifier(max_depth=4)
dt_clf.fit(X_train_pca, y_train)


# In[153]:


y_pred_train = dt_clf.predict(X_train_pca)
y_pred_test = dt_clf.predict(X_test_pca)

dt_pca_train_recall_score = recall_score(y_train, y_pred_train, average='weighted')
dt_pca_test_recall_score = recall_score(y_test, y_pred_test, average='weighted')

print('Train Recall score: {}'
      .format(dt_pca_train_recall_score))
print('Test Recall score: {}'
      .format(dt_pca_test_recall_score))

metrics.confusion_matrix(y_test, y_pred_test)


# We can see that decision tree performs really bad with PCA. This may be due to information lose due to dimentionality reduction.
# Also model is also overfitting as there is difference between train and test score.
# 
# 

# ### Random Forest with PCA

# In[154]:


from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators=300, criterion='gini',n_jobs= -1,random_state=10)

param_grid = {'max_features': [25,50,75,100],
          'max_depth': [6,8,10,12,14],
           'max_leaf_nodes':[20,22,30,50]}

grid_search = GridSearchCV(rf_clf, param_grid, scoring = 'recall_weighted',cv=kFold, return_train_score=True)
grid_search.fit(X_train_pca,y_train)


# In[155]:


grid_search.cv_results_


# In[156]:


grid_search.best_params_


# In[157]:


grid_search.best_score_


# In[158]:


rf_clf = RandomForestClassifier(n_estimators=300, criterion='gini',max_features=50,max_depth=12,max_leaf_nodes=50)
rf_clf.fit(X_train_pca, y_train)


# In[159]:


y_pred_train = rf_clf.predict(X_train_pca)
y_pred_test = rf_clf.predict(X_test_pca)

rf_pca_train_recall_score = recall_score(y_train, y_pred_train, average='weighted')
rf_pca_test_recall_score = recall_score(y_test, y_pred_test, average='weighted')

print('Train Recall score: {}'
      .format(rf_pca_train_recall_score))
print('Test Recall score: {}'
      .format(rf_pca_test_recall_score))

metrics.confusion_matrix(y_test, y_pred_test)


# In[160]:


y_test.value_counts()


# 
# Random forest is also overfitting the model after applying PCA.

# In[161]:


train_recall_scores= [knn_train_recall_score, lreg_train_recall_score, lsvc_train_recall_score, ksvc_train_recall_score, 
                      mnb_train_recall_score, dt_train_recall_score, rf_train_recall_score, knn_pca_train_recall_score,
                      logreg_pca_train_recall_score, lscvc_pca_train_recall_score, kscv_pca_train_recall_score, 
                      dt_pca_train_recall_score, rf_pca_train_recall_score]

test_recall_scores= [knn_test_recall_score, lreg_test_recall_score, lsvc_test_recall_score, ksvc_test_recall_score,
                     mnb_test_recall_score, dt_test_recall_score, rf_test_recall_score, knn_pca_test_recall_score,
                     logreg_pca_test_recall_score, lscv_pca_test_recall_score, kscv_pca_test_recall_score, 
                     dt_pca_test_recall_score, rf_pca_test_recall_score]

classifiers = ['KNN Clasification', 'Logistic Regression', 'Linear SVM', 'Kernelized SVM', 'Naive Bayes', 'Decision Tree',
               'Random Forest', 'KNN Classification with PCA', 'Logistic Regression with PCA', 'Linear SVM with PCA',
               'Kernalised SVM with PCA', 'Decision Trees with PCA', 'Random Forest with PCA']


# In[162]:


for_plot = pd.DataFrame ([train_recall_scores, test_recall_scores], columns=classifiers, index=['Train Recall Score', 'Test Recall Score'])
for_plot=for_plot.T


# In[163]:


for_plot


# In[164]:


for_plot.plot(kind='bar', figsize=(14,3), color='gbmykc',  edgecolor='k')

plt.title('Train&Test Scores of Classifiers')
plt.xlabel('Models')
plt.ylabel('Scores')
plt.legend(loc=4 , bbox_to_anchor=(1.2, 0))
plt.show();


# The Results have improved after using PCA. We get the best result using Linear SVC after applying PCA
