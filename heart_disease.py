import numpy as np
import data_processing
import dataplot
from sklearn.model_selection import train_test_split
import model_training as mt


def heart_disease(data_file):
    dataset = data_processing.read_dataset(data_file)

    X, y = data_processing.preprocess(dataset)
    # parameters = mt.init_parameters(n_features)

    X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size=0.3)
    

    costs_dimensions = {"costs_train": [], "costs_cv": [], "dimensions": []}


    
    for dim in range(1, 6):
        X_train_exp, X_cv_exp, X_test_exp = data_processing.expand_features(dim, X_train, X_cv, X_test)
            
        X_train_std, X_cv_std, X_train_std = data_processing.standardize(X_train_exp, X_cv_exp, X_train_exp)

        mt.train_and_evaluate(X_train_std, X_cv_std, y_train, y_cv, dim, costs_dimensions)


    dataplot.plot_learning_curves(costs_dimensions)
    
        
        
    
