# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 16:13:39 2020

@author: Zhangjun Zhou

Insight data challenge week 5
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


## loading data

os.getcwd()
os.chdir('C:/Users/Huajun/Desktop/Data challenge/')
file = 'breast-cancer-wisconsin.txt'
data = pd.read_table(file, sep = ",")
data.shape
data.info()

## Cleaning data
# drop duplicate records
len(data['ID'].unique())
data.drop_duplicates(subset = 'ID', keep = 'first', inplace = True)

def convert_class(value):
    """convert Class variable into benign vs. malignant"""
    if value == '4':
        return 0
    elif value == '2':
        return 1
    else:
        return None

data['malignant'] = data['Class'].apply(convert_class)
data['malignant'].value_counts()

data = data.dropna()
data.info()

data['breast cell'] = np.where(data['malignant'] == 1,
                         'Malignant', 'Benign')

columns = ['Uniformity of Cell Shape', 'Uniformity of Cell Size', 'Marginal Adhesion', \
     'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', \
     'Normal Nucleoli', 'Mitoses']

   
def convert_int(value):
    """convert str values into int values"""
    if value.isdigit():
        return int(value)
    else:
        return None
    
for label in columns:
    data[label] = data[label].apply(convert_int)

data = data.dropna()
data.info()

## EDA 
sns.countplot(data['breast cell'], label = "Count")

# plot the distribution of each feature by outcome classes
features = ['Clump Thickness', 'Uniformity of Cell Shape', 'Uniformity of Cell Size', \
            'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', \
            'Bland Chromatin', 'Normal Nucleoli', 'Mitoses']
for x in features:
    print(data.groupby('malignant')[x].mean())
    
dshort = data[features]
dshort['breast cell'] = data['breast cell']
df = pd.melt(dshort, id_vars = 'breast cell', \
             value_vars = features, value_name = 'value')

bins=np.linspace(df.value.min(), df.value.max(), 10)
g = sns.FacetGrid(df, col='variable', hue='breast cell', palette="Set1", col_wrap=2)
g.map(plt.hist, 'value', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()

# visualize the correlation among features
plt.figure(figsize=(12,12))
cor = dshort.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

## split train/test set and scale features for models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = data[features]
Y = data[['malignant']]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2,\
                                                    random_state = 0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

## different models
def models(X_train,Y_train):
  
    from sklearn.linear_model import LogisticRegression
    log = LogisticRegression(random_state = 0)
    log.fit(X_train, Y_train)
      
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    knn.fit(X_train, Y_train)
    
    from sklearn.svm import SVC
    svc_lin = SVC(kernel = 'linear', random_state = 0)
    svc_lin.fit(X_train, Y_train)
    
    from sklearn.svm import SVC
    svc_rbf = SVC(kernel = 'rbf', random_state = 0)
    svc_rbf.fit(X_train, Y_train)
    
    from sklearn.naive_bayes import GaussianNB
    gauss = GaussianNB()
    gauss.fit(X_train, Y_train)
    
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    tree.fit(X_train, Y_train)
    
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    forest.fit(X_train, Y_train)
      
    #print model accuracy on the training data.
    print('[0]Logistic Regression Training Accuracy:', log.score(X_train, Y_train))
    print('[1]K Nearest Neighbor Training Accuracy:', knn.score(X_train, Y_train))
    print('[2]Support Vector Machine (Linear Classifier) Training Accuracy:', svc_lin.score(X_train, Y_train))
    print('[3]Support Vector Machine (RBF Classifier) Training Accuracy:', svc_rbf.score(X_train, Y_train))
    print('[4]Gaussian Naive Bayes Training Accuracy:', gauss.score(X_train, Y_train))
    print('[5]Decision Tree Classifier Training Accuracy:', tree.score(X_train, Y_train))
    print('[6]Random Forest Classifier Training Accuracy:', forest.score(X_train, Y_train))
      
    return log, knn, svc_lin, svc_rbf, gauss, tree, forest

model = models(X_train, Y_train)


## metrics for all models above
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

for i in range(len(model)):
    cm = confusion_matrix(Y_test, model[i].predict(X_test))
      
    TN = cm[0][0]
    TP = cm[1][1]
    FN = cm[1][0]
    FP = cm[0][1]
      
    print('Model[{}] Confusion Matrix is:'.format(i))
    print(cm)
    print('Model[{}] Testing Accuracy = "{}"'.format(i, (TP + TN) / (TP + TN + FN + FP)))
    print(classification_report(Y_test, model[i].predict(X_test)))
    print()


## use random forest classifier model
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
forest.fit(X_train, Y_train)
print(forest.feature_importances_)
feat_importances = pd.Series(forest.feature_importances_, index=X.columns)
feat_importances = feat_importances.sort_values()
feat_importances.plot(kind='barh')
plt.show()

cm = confusion_matrix(Y_test, forest.predict(X_test))
print(cm)
print(classification_report(Y_test, forest.predict(X_test)))
print(accuracy_score(Y_test, forest.predict(X_test)))

## test out how each feature might contribute to FPR and FNR
FPR_list = []
FNR_list = []
removed_list = []

for i in range(len(features)):
    new_features = features[:]
    removed_list.append(new_features[i])
    new_features.remove(new_features[i])
    
    X = data[new_features]
    Y = data[['malignant']]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2,\
                                                    random_state = 0)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    forest.fit(X_train, Y_train)
    cm = confusion_matrix(Y_test, forest.predict(X_test))      
    TN = cm[0][0]
    TP = cm[1][1]
    FN = cm[1][0]
    FP = cm[0][1]
    FPR = FP / (FP + TN)
    FNR = FN / (FN + TP)
    FPR_list.append(FPR)
    FNR_list.append(FNR)

result = pd.DataFrame(columns = ['removed feature', 'FPR', 'FNR'])
result['removed feature'] = removed_list
result['FPR'] = FPR_list
result['FNR'] = FNR_list
result
