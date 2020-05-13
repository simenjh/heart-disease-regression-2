# Heart disease detection through logistic regression 2

This project is a continuation of [Heart disease detection through logistic regression 1](https://github.com/simenjh/heart-disease-regression-1)


The above project indicated that the trained linear model was suffering from high bias.


## The purposes of this project:
* Visualize the cross-validation cost of different models. 
* Aim to find a model with lower bias, using polynomial features (1D-5D).
* Analyze results.


Dataset: [Heart disease UCI](https://www.kaggle.com/ronitf/heart-disease-uci)


All the trained models have been regularized.

## Run program
Call heart_disease(data_file) in heart_disease.py.


<br/> <br/>

![](images/learning_curves.png?raw=true)

As can be seen from the above figure, the variance is increasing with larger model polynomial degree. The linear model looks to be the best in terms of bias-variance tradeoff.




## Key takeaways
* None of the higher-order polynomials beat the linear model in terms of bias-variance tradeoff.
* Possibly, other methods of feature expansion might give better results. Linear or cubic splines could be worth a shot.
* Training the higher-order polynomial models with more data is likely to bring down the variance.
* Methods like SVMs and neural networks in order to train more advanced models, might be worth exploring. 
