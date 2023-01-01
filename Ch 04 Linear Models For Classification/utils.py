import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def create_mesh(X, model, h):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    return xx, yy, Z

def plot_DR(X, y, model, title='', h=0.01, ax=None):

    xx, yy, Z = create_mesh(X, model, h)
    sns.color_palette('Spectral', as_cmap=True)

    if ax:
        ax.contourf(xx, yy, Z)
        ax.set_title(title)
        plt.set_xlabel('X1')
        plt.set_ylabel('X2')
    else:
        plt.figure(figsize=(12, 6))
        plt.title(title)
        plt.contourf(xx, yy, Z)
        plt.xlabel('X1')
        plt.ylabel('X2')
    # create scatter plot for samples from each class
    for yi in np.unique(y):
        # create scatter of these samples
        if ax: ax.scatter(X[:, 0][y==yi], X[:, 1][y==yi])
        else : plt.scatter(X[:, 0][y==yi], X[:, 1][y==yi])            
    if ax: return ax
    plt.show()
    
def plot_DB(X, y, model, title='', h=0.01):
    sns.color_palette('Spectral', as_cmap=True)
    xx, yy, Z = create_mesh(X, model, h)
    fig = plt.figure(figsize=(12, 6))
    for yi in np.unique(y):
        plt.scatter(X[:, 0][y==yi], X[:, 1][y==yi])
#     plt.plot(X[:, 0][y==1], X[:, 1][y==1])
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title(title)
    plt.contour(xx, yy, Z)
    plt.show()
