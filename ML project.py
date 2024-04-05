#!/usr/bin/env python
# coding: utf-8

# # Machine Learning Project 01
# ### Objectives
# 
# 1. Predicting Heart-disease
# 2. Inorm decision-making of Cardiologist
# 
# ### Model in action
# 1. Model can Inform but should not make decision
# 2. Especially important in healthcare
# 
# ### Patient ---> HeartDisease / No HeartDisease
# 
# ### steps:
# 1. Problem Understanding
# 2. Data Collection and Preperation
# 3. Feature Engineering
# 4. Data Modelling
# 5. Model Training
# 6. Model Evaluation
# 7. Model Deployment
# 8. Model Monitoring
# 
# ### What you will expect while using the system
# 1. How accurate? ---> Accuracy
# 2. Is the system Reliable? ---> Reliability
# 3. How secure the system is? ---> Security
# 4. Does the system interpret the result well? ---> Interpretibility
# 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer #missing value
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from scipy.stats import chi2_contingency

from sklearn.ensemble import ExtraTreesClassifier #feature selection
#hyperparameter tuning
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv("https://raw.githubusercontent.com/ds-mahbub/24MLE01_Machine-Learning-Engineer/KNN/Classification/data/heart_disease.csv")


# In[3]:


df.shape


# In[4]:


df.head()


# # Data Cleaning

# In[5]:


df.duplicated().sum()


# In[6]:


df.drop_duplicates(inplace = True)


# In[7]:


df.duplicated().sum()


# In[8]:


df.isnull().sum()


# No null values

# In[9]:


df.nunique()


# In[10]:


df['HeartDisease'].value_counts()


# In[11]:


df['Diabetic'].value_counts()


# # Data Encoding

# In[12]:


df = df[df.columns].replace({
'Yes':1, 'No':0, 'Male':1, 'Female':0, 'No, borderline diabetes':0, 'Yes (during pregnancy)':1}
)


# In[13]:


df


# In[14]:


df['Diabetic'].value_counts()


# In[15]:


new_df=df.drop(columns =['AgeCategory', 'Race', 'GenHealth'])
corr = new_df.corr().round(3)

plt.figure(figsize = (12,10))
sns.heatmap(corr,annot = True,cmap='coolwarm')
plt.show()


# In[16]:


sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})


plt.figure(figsize=(16,10))
plt.title('Correlation distribution of features')

corr_values = abs(corr['HeartDisease']).sort_values()[:-1]
bar_colors = sns.color_palette("cubehelix", len(corr_values))  
corr_values.plot.barh(color=bar_colors) 

plt.show()


# In[ ]:





# # Statistical Analysis

# In[17]:


df.columns


# In[18]:


df.dtypes


# In[19]:


df.nunique()


# In[20]:


categorical_vars = []
continuous_vars = []

for col in df.columns:
    unique_values = df[col].nunique()
    if unique_values <= 6:
        categorical_vars.append(col)
    else:
        continuous_vars.append(col)


# In[21]:


categorical_vars


# In[22]:


continuous_vars


# In[23]:


from scipy.stats import chi2_contingency


# In[24]:


p_values = {}
for variable in categorical_vars:
    contingency_table = pd.crosstab(df['HeartDisease'], df[variable])
    chi2, p, _, _ = chi2_contingency(contingency_table)
    p_values[variable] = p


print("P-values for chi-squared test against HeartDisease:")
for variable, p_value in p_values.items():
    print(f"{variable}: {p_value}")


# In[25]:


significant_categorical = []
for column in categorical_vars:
    contingency_table = pd.crosstab(df[column], df['HeartDisease'])
    chi2, p, _, _ = chi2_contingency(contingency_table)
    if p < 0.05:  
        significant_categorical.append(column)
    print(f"Chi-squared p-value for '{column}': {p}")

print("\nStatistically significant categorical features:")
print(significant_categorical)


# In[26]:


from sklearn.preprocessing import LabelEncoder
continuous_vars_with_string = ['AgeCategory', 'GenHealth','Race']
label_encoder = LabelEncoder()
for var in continuous_vars_with_string:
    df[var] = label_encoder.fit_transform(df[var])


# All the p_values shows the categorical variable columns are statistically significant

# In[27]:


p_values_continuous = {}
from scipy.stats import ttest_ind
for var in continuous_vars:
    heart_disease_present = df[df['HeartDisease'] == 1][var]
    heart_disease_absent = df[df['HeartDisease'] == 0][var]
    t_stat, p_value = ttest_ind(heart_disease_present, heart_disease_absent)
    p_values_continuous[var] = p_value
    
print("P-values for t-test against HeartDisease (continuous variables):")
for var, p_value in p_values_continuous.items():
    print(f"{var}: {p_value}")


# P_values show we have all continous data to be statistically significant

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[28]:


X = df.drop('HeartDisease', axis = 1)
y = df['HeartDisease']


# In[29]:


feat_select = ExtraTreesClassifier()
feat_select.fit(X,y)


# In[30]:


feat_imp = pd.Series(feat_select.feature_importances_, index = X.columns)
feat_imp.nlargest(len(df.columns)).plot(kind='barh')
plt.show()


# In[31]:


feat_imp.sort_values(ascending = False)


# In[32]:


X = df[feat_imp[:6].index]


# In[33]:


X


# In[34]:


X_train, X_test, y_train, y_test = train_test_split(X,y,shuffle = True, test_size = 0.25,random_state = 42)


# # Data Preprocessing

# In[ ]:





# In[ ]:





# In[35]:


X


# In[36]:


df.dtypes


# In[ ]:





# In[ ]:





# In[37]:


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
X_train


# In[ ]:





# In[38]:


X_train


# In[ ]:





# In[39]:


X_train.shape


# In[40]:


def model_evaluation(estimator, x_test, y_test):
    from sklearn import metrics
    y_pred = estimator.predict(x_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    prec = metrics.precision_score(y_test, y_pred)
    rec = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    kappa = metrics.cohen_kappa_score(y_test, y_pred)
    # AUC (Area Under the Curve)
    y_pred_proba = estimator.predict_proba(x_test)[::,1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    cm = metrics.confusion_matrix(y_test, y_pred)
    return {'accuracy': acc, 'precision': prec, 'recall':rec, 'f1_score': f1, 'kappa':kappa,
           'fpr':fpr, 'tpr': tpr, 'auc': auc, 'cm': cm}


# In[41]:


knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train, y_train)
knn_eval = model_evaluation(knn, X_test, y_test)
print('Accuracy: ', knn_eval['accuracy'])
print('Precision: ', knn_eval['precision'])
print('Recall: ', knn_eval['recall'])
print('f1_score', knn_eval['f1_score'])
print('Cohens Kappa Score: ', knn_eval['kappa'])
print('Area Under Curve: ', knn_eval['auc'])


# In[42]:


y_pred = knn.predict(X_test)


# In[43]:


confusion_matrix = knn_eval['cm']


# In[44]:


confusion_matrix


# In[45]:


from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier(random_state = 42)
DT.fit(X_train, y_train)
clf_eval = model_evaluation(DT, X_test, y_test)
print('Accuracy: ', clf_eval['accuracy'])
print('Precision: ', clf_eval['precision'])
print('Recall: ', clf_eval['recall'])
print('f1_score', clf_eval['f1_score'])
print('Cohens Kappa Score: ', clf_eval['kappa'])
print('Area Under Curve: ', clf_eval['auc'])


# In[46]:


y_pred = DT.predict(X_test)
confusion_matrix = clf_eval['cm']
confusion_matrix


# In[47]:


# Intitialize figure with two plots
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Model Comparison', fontsize=16, fontweight='bold')
fig.set_figheight(7)
fig.set_figwidth(14)
fig.set_facecolor('white')

# First plot
## set bar size
barWidth = 0.2
clf_score = [clf_eval['accuracy'], clf_eval['precision'], clf_eval['recall'], clf_eval['f1_score'], clf_eval['kappa']]
knn_score = [knn_eval['accuracy'], knn_eval['precision'], knn_eval['recall'], knn_eval['f1_score'], knn_eval['kappa']]


## Set position of bar on X axis
r1 = np.arange(len(clf_score))
r2 = [x + barWidth for x in r1]

## Make the plot
ax1.bar(r1, clf_score, width=barWidth, edgecolor='white', label='Decision Tree')
ax1.bar(r2, knn_score, width=barWidth, edgecolor='white', label='K-Nearest Neighbors')

## Configure x and y axis
ax1.set_xlabel('Metrics', fontweight='bold')
labels = ['Accuracy', 'Precision', 'Recall', 'F1', 'Kappa']
ax1.set_xticks([r + (barWidth * 1.5) for r in range(len(clf_score))], )
ax1.set_xticklabels(labels)
ax1.set_ylabel('Score', fontweight='bold')
ax1.set_ylim(0, 1)

## Create legend & title
ax1.set_title('Evaluation Metrics', fontsize=14, fontweight='bold')
ax1.legend()

# Second plot
## Comparing ROC Curve
ax2.plot(clf_eval['fpr'], clf_eval['tpr'], label='Decision Tree, auc = {:0.5f}'.format(clf_eval['auc']))
ax2.plot(knn_eval['fpr'], knn_eval['tpr'], label='K-Nearest Nieghbor, auc = {:0.5f}'.format(knn_eval['auc']))

## Configure x and y axis
ax2.set_xlabel('False Positive Rate', fontweight='bold')
ax2.set_ylabel('True Positive Rate', fontweight='bold')

## Create legend & title
ax2.set_title('ROC Curve', fontsize=14, fontweight='bold')
ax2.legend(loc=4)

plt.show()


# In[48]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, roc_auc_score


# In[49]:


random_forest = RandomForestClassifier()
random_forest.fit(X_train, y_train)
random_forest_eval = model_evaluation(random_forest, X_test, y_test)
print('Random Forest:')
print('Accuracy: ', random_forest_eval['accuracy'])
print('Precision: ', random_forest_eval['precision'])
print('Recall: ', random_forest_eval['recall'])
print('F1 Score:', random_forest_eval['f1_score'])
print('Cohen\'s Kappa Score: ', random_forest_eval['kappa'])
print('Area Under Curve: ', random_forest_eval['auc'])
print()


# In[50]:


logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train)
logistic_regression_eval = model_evaluation(logistic_regression, X_test, y_test)
print('Logistic Regression:')
print('Accuracy: ', logistic_regression_eval['accuracy'])
print('Precision: ', logistic_regression_eval['precision'])
print('Recall: ', logistic_regression_eval['recall'])
print('F1 Score:', logistic_regression_eval['f1_score'])
print('Cohen\'s Kappa Score: ', logistic_regression_eval['kappa'])
print('Area Under Curve: ', logistic_regression_eval['auc'])
print()


# In[51]:


# Intitialize figure with two plots
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Model Comparison', fontsize=16, fontweight='bold')
fig.set_figheight(7)
fig.set_figwidth(14)
fig.set_facecolor('white')

# First plot
## set bar size
barWidth = 0.2
logistic_regression_score = [logistic_regression_eval['accuracy'],logistic_regression_eval['precision'],logistic_regression_eval['recall'], logistic_regression_eval['f1_score'], logistic_regression_eval['kappa']]
knn_score = [knn_eval['accuracy'], knn_eval['precision'], knn_eval['recall'], knn_eval['f1_score'], knn_eval['kappa']]


## Set position of bar on X axis
r1 = np.arange(len(logistic_regression_score))
r2 = [x + barWidth for x in r1]

## Make the plot
ax1.bar(r1, clf_score, width=barWidth, edgecolor='white', label='Logistic Regression')
ax1.bar(r2, knn_score, width=barWidth, edgecolor='white', label='K-Nearest Neighbors')

## Configure x and y axis
ax1.set_xlabel('Metrics', fontweight='bold')
labels = ['Accuracy', 'Precision', 'Recall', 'F1', 'Kappa']
ax1.set_xticks([r + (barWidth * 1.5) for r in range(len(logistic_regression_score))], )
ax1.set_xticklabels(labels)
ax1.set_ylabel('Score', fontweight='bold')
ax1.set_ylim(0, 1)

## Create legend & title
ax1.set_title('Evaluation Metrics', fontsize=14, fontweight='bold')
ax1.legend()

# Second plot
## Comparing ROC Curve
ax2.plot(logistic_regression_eval['fpr'], logistic_regression_eval['tpr'], label='Logistic Regression, auc = {:0.5f}'.format(clf_eval['auc']))
ax2.plot(knn_eval['fpr'], knn_eval['tpr'], label='K-Nearest Nieghbor, auc = {:0.5f}'.format(knn_eval['auc']))

## Configure x and y axis
ax2.set_xlabel('False Positive Rate', fontweight='bold')
ax2.set_ylabel('True Positive Rate', fontweight='bold')

## Create legend & title
ax2.set_title('ROC Curve', fontsize=14, fontweight='bold')
ax2.legend(loc=4)

plt.show()


# In[ ]:




