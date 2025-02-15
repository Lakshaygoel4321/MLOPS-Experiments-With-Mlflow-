from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import GridSearchCV
import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub
dagshub.init(repo_owner='iamhimanshu12goel', repo_name='MLOPS-Experiments-With-Mlflow-', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/iamhimanshu12goel/MLOPS-Experiments-With-Mlflow-.mlflow/")

breast = load_breast_cancer()

x = pd.DataFrame(breast.data,columns=breast.feature_names)
y = pd.Series(breast.target,name='target')

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)


clf = RandomForestClassifier()
#clf.fit(x_train,y_train)

# y_pred = clf.predict(x_test)
# accuracy = accuracy_score(y_test,y_pred)

params = {
    'n_estimators':[100,50,60],
    'max_depth':[None,4,6,8]
}

grid = GridSearchCV(estimator=clf,param_grid=params,cv=3,scoring='accuracy')

with mlflow.start_run(run_name='second'):
    
    grid.fit(x_train,y_train)

    y_pred = grid.predict(x_test)
    
    #mlflow.log_param('n_estimators',n_estimators)
    best_param = grid.best_params_
    best_score = grid.best_score_

    mlflow.log_params(best_param)
    mlflow.log_metric('accuracy',best_score)
    
    train_df = x_train.copy()
    train_df['target'] = y_train

    test_df = x_test.copy()
    test_df['target'] = y_test

    test_df = mlflow.data.from_pandas(test_df)
    mlflow.log_input(test_df,'test_df')

    mlflow.log_artifact(__file__)

    mlflow.sklearn.log_model(grid.best_estimator_,'Best_model')

    print(best_param)
    print(best_score)