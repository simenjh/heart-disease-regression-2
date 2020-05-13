import matplotlib.pyplot as plt



def plot_learning_curves(costs_dimensions):
    costs_train = costs_dimensions["costs_train"]
    costs_cv = costs_dimensions["costs_cv"]
    dimensions = costs_dimensions["dimensions"]
    
    plt.style.use("seaborn")
    plt.plot(dimensions, costs_train, label='Training cost')
    plt.plot(dimensions, costs_cv, label='Cross-validation cost')
    plt.ylabel('Cost', fontsize=14)
    plt.xlabel('d (polynomial degree)', fontsize=14)
    plt.title('Learning curves', fontsize=18, y=1.03)
    plt.legend()
    plt.ylim(0, 1)
    plt.show()


