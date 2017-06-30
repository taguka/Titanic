import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import GridSearchCV,cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew


########## Loading files########################
train=pd.read_csv('C:/Kaggle/Titanic/train.csv')
test=pd.read_csv('C:/Kaggle/Titanic/test.csv')
test['Survived']=0

print('Skew of Fare(continuous number feature) train:')
print(skew(train['Fare']))
train['Fare']=np.log1p(train['Fare'])
print(skew(train['Fare']))  

print('Skew of Fare(continuous number feature) test:')
print(skew(test['Fare'].dropna()))
test['Fare']=np.log1p(test['Fare'])
print(skew(test['Fare'].dropna()))  

print('Skew of Fare(continuous number feature) train:')
print(skew(train['Fare']))
train['Fare']=np.log1p(train['Fare'])
print(skew(train['Fare']))  
    
train_shape=train.shape[0]
full=pd.concat([train, test])
######### check null values
print('Missing data:')
print(full.isnull().sum())
print('#############################################')
print('Numeric data: ')
print(train.select_dtypes(exclude=['object']).columns.tolist())
print('Categoric data: ')
print(train.select_dtypes(include=['object']).columns.tolist())
print('#############################################')
print('Probability of surviving:')
print(train.groupby(['Sex'])['Survived'].agg(['mean', 'count']))
print(train.groupby(['Pclass'])['Survived'].agg(['mean', 'count']))
print(train.groupby(['Embarked'])['Survived'].agg(['mean', 'count']))
print(train.groupby(['SibSp'])['Survived'].agg(['mean', 'count']))
print(train.groupby(['Parch'])['Survived'].agg(['mean', 'count']))
print('#############################################')
#Fill missing values
full['Embarked']=full['Embarked'].fillna('S')
full['Age']=full['Age'].fillna(np.mean(full['Age']))
full['Cabin']=full['Cabin'].fillna('')
full.loc[full['Ticket']=='3701','Fare']=np.mean(full.loc[full['Pclass']==3,'Fare'])
full.loc[(full['Fare']==0)&(full['Pclass']==1),'Fare']=np.mean(full.loc[full['Pclass']==1,'Fare'])
full.loc[(full['Fare']==0)&(full['Pclass']==2),'Fare']=np.mean(full.loc[full['Pclass']==2,'Fare'])
full.loc[(full['Fare']==0)&(full['Pclass']==3),'Fare']=np.mean(full.loc[full['Pclass']==3,'Fare'])

#Features engineering
full['Family_num']=full['Parch']+full['SibSp']
full['IsAlone']=[1 if x==0 else 0 for x in full['Family_num']]
full['Adult']=[1 if x>14 else 0 for x in full['Age']]
full['Title']= [re.search(' ([A-Za-z]+)\.', x).group(1) for x in full['Name']]
mapping_title={'Mrs': 'Mrs', 'Miss': 'Miss', 'Mr': 'Mr', 'Master': 'Master',
               'Capt': 'Other','Col': 'Other','Countess': 'Miss','Don': 'Other',
               'Dona': 'Mrs','Dr': 'Other','Jonkheer': 'Other','Lady': 'Mrs',
               'Major': 'Other','Mlle': 'Miss','Mme': 'Miss','Miss': 'Other',
               'Rev': 'Other','Sir': 'Other'}
full['Title']=full['Title'].map(mapping_title)
full['IsCabin']=[0 if x=='' else 1 for x in full['Cabin']]
full['Ticket_char']=[x[0] for x in full['Ticket']]
full.loc[full['Ticket_char'].str.isnumeric(),'Ticket_char']='Number'
cols_del=['Cabin','Name','Parch','PassengerId','SibSp','Survived','Ticket']


full=full.drop(cols_del,axis=1)
full=pd.get_dummies(full)
print('Final features: ')
print(full.columns.tolist())
print('#############################################')
X_train=full[:train_shape].values
X_test=full[train_shape:].values
       
sc=StandardScaler()           
X_train_sc=sc.fit_transform(full[:train_shape])
X_test_sc=sc.transform(full[train_shape:])
           
y_train=train.Survived        
kfold = KFold(n_splits=10)
print('SVC model:')
svc=SVC()
scores = np.mean(cross_val_score(svc,X_train,y_train, cv=kfold, scoring='accuracy'))
print('CV cross validation: %s' % scores)

scores = np.mean(cross_val_score(svc,X_train_sc,y_train, cv=kfold, scoring='accuracy'))
print('CV cross validation std: %s' % scores)

param_grid = [{'kernel': ['rbf'],'C': [0.001, 0.01, 0.1, 1, 10, 100],'gamma': [0.001, 0.01, 0.1, 1, 10, 100]},
              {'kernel': ['linear'],'C': [0.001, 0.01, 0.1, 1, 10, 100]}]
grid=GridSearchCV(svc,param_grid=param_grid,cv=kfold, scoring='accuracy')
grid.fit(X_train, y_train)
print('Best parameters: ', grid.best_params_)
print('CV best model: ', grid.score(X_train, y_train))

grid.fit(X_train_sc, y_train)
print('Best parameters std: ', grid.best_params_)
print('CV best model std: ', grid.score(X_train_sc, y_train))
y_test=grid.predict(X_test_sc)
sub_svc=pd.DataFrame({'PassengerId':test['PassengerId'], 'Survived':pd.Series(y_test, name='Survived')})

sub_svc.to_csv('C:/Kaggle/Titanic/sub_svc.csv', index=False)
