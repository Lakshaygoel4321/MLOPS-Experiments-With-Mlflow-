from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")

breast = load_breast_cancer()

x = breast.data
y = breast.target

# train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

n_estimators = 45
max_depth = 5


with mlflow.start_run(run_name='second'):
    clf = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,random_state=42)
    clf.fit(x_train,y_train)
    
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test,y_pred)

    mlflow.log_metric("accuracy_score",accuracy)
    mlflow.log_param("n_estimators",n_estimators)
    mlflow.log_param("max_depth",max_depth)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=breast.target_names, yticklabels=breast.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    # save plot
    plt.savefig("Confusion-matrix.png")

    # log artifacts using mlflow
    mlflow.log_artifact("Confusion-matrix.png")
    mlflow.log_artifact(__file__)
