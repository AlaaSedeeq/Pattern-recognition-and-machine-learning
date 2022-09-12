import numpy as np
import matplotlib.pyplot as plt


def create_mesh(X, model, h = 0.01):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    return xx, yy, Z

def plot_DR(X, y, model, title='', ax=None):

    xx, yy, Z = create_mesh(X, model)
    
    if ax:
        ax.contourf(xx, yy, Z, cmap='crest')
        ax.set_title(title)
        plt.set_xlabel('X1')
        plt.set_ylabel('X2')
    else:
        plt.figure(figsize=(12, 6))
        plt.title(title)
        plt.contourf(xx, yy, Z, cmap='crest')
        plt.xlabel('X1')
        plt.ylabel('X2')
    # create scatter plot for samples from each class
    for class_value in range(2):
        # get row indexes for samples with this class
        row_ix = np.where(y == class_value)
        # create scatter of these samples
        if ax: ax.scatter(X[row_ix, 0], X[row_ix, 1], cmap='Paired')
        else : plt.scatter(X[row_ix, 0], X[row_ix, 1], cmap='Paired')            
    if ax: return ax
    plt.show()
    
def plot_DB(X, y, model, title=''):
    xx, yy, Z = create_mesh(X, model)
    fig = plt.figure(figsize=(12, 6))
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], 'ro')
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], 'bo')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title(title)
    plt.contour(xx, yy, Z)
    plt.show()