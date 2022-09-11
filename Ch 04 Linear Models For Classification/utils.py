import numpy as np
import matplotlib.pyplot as plt

def plot_decision_region(X, y, model, ax=None, title=''):

    # define bounds of the domain
    min1, max1 = X[:, 0].min()-1, X[:, 0].max()+1
    min2, max2 = X[:, 1].min()-1, X[:, 1].max()+1
    # define the x and y scale
    x1grid = np.arange(min1, max1, 0.01)
    x2grid = np.arange(min2, max2, 0.01)
    # create all of the lines and rows of the grid
    xx, yy = np.meshgrid(x1grid, x2grid)
    # flatten each grid to a vector
    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
    # horizontal stack vectors to create x1,x2 input for the model
    grid = np.hstack((r1,r2))
    # make predictions for the grid
    yhat = model.predict(grid)
    # reshape the predictions back into a grid
    zz = yhat.reshape(xx.shape) 
    # plot the grid of x, y and z values as a surface
    if ax:
        ax.contourf(xx, yy, zz, cmap='crest')
        ax.set_title(title)
    else:
        plt.figure(figsize=(10, 5))
        plt.title(title)
        plt.contourf(xx, yy, zz, cmap='crest')
    # create scatter plot for samples from each class
    for class_value in range(2):
        # get row indexes for samples with this class
        row_ix = np.where(y == class_value)
        # create scatter of these samples
        if ax:
            ax.scatter(X[row_ix, 0], X[row_ix, 1], cmap='Paired')
        else:
            plt.scatter(X[row_ix, 0], X[row_ix, 1], cmap='Paired')
            
    if ax: return ax
    plt.show()
