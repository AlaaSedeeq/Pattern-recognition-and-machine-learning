import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal

def plot_dist_3d(mu, cov, range_, line, ax, title=''):
    '''
    Create grid and multivariate normal
    '''
    x = np.linspace(range_[0], range_[1], line)
    y = np.linspace(range_[0], range_[1], line)
    
    X, Y = np.meshgrid(x,y)
    
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X; pos[:, :, 1] = Y
    
    rv = multivariate_normal(mu, cov)

    ax.plot_surface(X, Y, rv.pdf(pos), cmap='viridis', linewidth=0)
    ax.set_title(title)
    ax.set_xlabel('X1 axis')
    ax.set_ylabel('X2 axis')
    ax.set_zlabel('Y axis')
    
def solve(m1,m2,std1,std2):
    '''
    To find the intersection of two normal distribution
    '''
    a = 1/(2*std1**2) - 1/(2*std2**2)
    b = m2/(std2**2) - m1/(std1**2)
    c = m1**2 /(2*std1**2) - m2**2 / (2*std2**2) - np.log(std2/std1)
    return np.roots([a,b,c])

def plot_dist(dataset, col, target, inter):
    '''
    Plot distribution of variable and all class-conditional distributions
    '''
    cls = dataset[target].unique()
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    sns.kdeplot(dataset[col], ax=axes[0])
    axes[0].set_title('%s data distribution P(X1)'%col)
    sns.kdeplot(x=dataset[dataset['class']==cls[0]][col], ax=axes[1])
    sns.kdeplot(x=dataset[dataset['class']==cls[1]][col], ax=axes[1])
    axes[1].set_title('Conditional distribution P(C|%s)'%col)
    axes[1].axvline(x=inter[1], ymin=0, ymax=0.95)
    axes[1].legend(['Class 1', 'Class 2'])
    

def plot_results(y_act, pred):
    from sklearn.metrics import (accuracy_score, confusion_matrix, 
                                 f1_score, accuracy_score)

    print("Accuracy: {}".format(round(accuracy_score(y_act, pred), 4)))
    print("F1-Score: {}".format(round(f1_score(y_act, pred)), 4))

    cm = confusion_matrix(y_act, pred)
    ax = sns.heatmap(cm, annot=True, cmap='Blues')
    # Set axes labels and title
    ax.set_title('Confusion Matrix\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');
    # Ticket labels
    ax.xaxis.set_ticklabels(['False','True'])
    ax.yaxis.set_ticklabels(['False','True'])
    # Display the visualization of the Confusion Matrix.
    plt.show()