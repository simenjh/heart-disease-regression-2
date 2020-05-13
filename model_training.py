import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss



def train_and_evaluate(X_train_std, X_cv_std, y_train, y_cv, dim, costs_dimensions):
    clf = LogisticRegression(max_iter=500).fit(X_train_std, np.ravel(y_train))

    
    print(f"Training accuracy: {clf.score(X_train_std, y_train)}")
    print(f"CV accuracy: {clf.score(X_cv_std, y_cv)}\n")

       
    train_pred = clf.predict_proba(X_train_std)
    cv_pred = clf.predict_proba(X_cv_std)

    train_cost = log_loss(y_train, train_pred)
    cv_cost = log_loss(y_cv, cv_pred)

    costs_dimensions["costs_train"].append(train_cost)
    costs_dimensions["costs_cv"].append(cv_cost)
    costs_dimensions["dimensions"].append(dim)
        

    # print(f"Train error {dim}D: {log_loss(y_train, train_pred)}")
    # print(f"CV error {dim}D: {log_loss(y_cv, cv_pred)}\n")        




        

        
