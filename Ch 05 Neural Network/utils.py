import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import namedtuple
import torch
import torch.nn as nn
from torch.optim import SGD
from torch import from_numpy
from torch.autograd import Variable
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

def simulator(f, sample_size, std, l=0, h=1):
    x = np.linspace(l, h, sample_size)
    noise = np.random.normal(scale=std, size=x.shape)
    t = f(x) + noise
    return x.reshape(sample_size, -1), t

# generate data
def genrate_data(f, N_tr=10, N_ts=100, std=0.3, l=0, h=1):
    x_train, y_train = simulator(f, N_tr, std, l=l, h=h)
    x_test = np.linspace(l, h, N_ts)
    y_test = f(x_test)
    return x_train.reshape(N_tr, -1), y_train, x_test.reshape(N_ts, -1), y_test

class MLE_MAP:
    def __init__(self, basis, degree=None, b_mu=None, b_s=None, mle_mape='MLE', lamba=0.9):
        self.basis = basis
        self.degree = degree
        self.b_mu = b_mu
        self.b_s = b_s
        self.mle_mape = mle_mape.lower()
        self.lamba = lamba # regularization
    
    # get the basis of the input (Gaussian, Poly, Sigmoidal)
    def _basis(self, X):
        if isinstance(self.b_mu, int):
            self.b_mu = [self.b_mu] 
            
        X_trans = []
        if self.basis.lower()=='gauss':
            for m in self.b_mu:
                X_trans.append(np.exp(-0.5 * np.square(X - m)/self.b_s))

        elif self.basis.lower()=='poly':
            for i in range(0, self.degree+1):
                X_trans.append(X**i)

        elif self.basis.lower()=='sig':
            for m in self.b_mu:
                X_trans.append((np.tanh((X - m) / self.b_s) + 1 ) / 2)
        else:
            raise "Only Gauss, Poly, Sig"

        return np.asarray(X_trans).squeeze().transpose()
    def _scale_fit(self, X, L=-1, H=1):
        P = X.shape[1]
        self.a = []; self.b = []
        for p in range(P):
            self.a.append(((H * min(X[:, p])) - (L * max(X[:, p])))/(H-L))
            self.b.append((max(X[:, p]) - min(X[:, p]))/(H - L))

    def _scale_transform(self, X):
#         np.seterr(invalid='ignore')
        X_scaled = np.divide((X - self.a), self.b)
        return X_scaled.reshape(X.shape[0], -1)
    
    def _scale_inverse(X):
        return (X * self.b) + selfa
    
    def fit(self, X, y, scale=True):
        
        self.scale = scale
        
        if self.basis:
            X = self._basis(X)
        
        self.N, self.P = X.shape
        
        # add cloumn of ones in X for the bias term
        X = np.append(X, np.ones((self.N, 1)), axis=1)
        y = y.reshape(self.N, 1)
        
        self.N, self.P = X.shape
        
        if self.scale:
            self._scale_fit(X)
            X = self._scale_transform(X)
        
        if self.mle_mape=='mle':
            # The Normal Equation W(ML) = (X^T X)^-1 X^T Y
            self.W = np.linalg.inv(X.T @ X) @ X.T @ y
            self.B_inv = np.mean(np.square(X @ self.W - y))

        elif self.mle_mape=='map':
            # The Normal Equation W(ML) = (X^T X + lambda I)^-1 X^T Y
            self.W = np.linalg.inv(self.lamba * np.eye(self.P + 1) + np.dot(X.T, X)) @ X.T @ y
            self.B_inv = np.mean(np.square(X @ self.W - y))
            
        else:
            raise('Only MLE and MAP')

    def predict(self, X, std=False):
        
        if self.basis:
            X = self._basis(X)
        
        if self.scale:
            X = self._scale_transform(X)
        
#         X = X.reshape(-1, self.N)
        X = np.append(X, np.ones((X.shape[0],1)), axis=1)
        preds = np.dot(X, self.W)
        if std:
            std_ = np.sqrt(self.B_inv) + np.zeros_like(preds)
            return preds, std_
        return preds

def plot_results(X_tr, y_tr, X_ts, y_ts, pred, ax, std=0, title=None, label='', pred_label='', true_label='', text=None):
    ax.scatter(X_tr, y_tr, facecolor="none", edgecolor="b", s=50, label=label)
    ax.plot(X_ts, pred, 'r', label=pred_label)
    ax.plot(X_ts, y_ts, c="g", label=true_label)
    
    if title:
        ax.set_title(title, fontsize=16)
    if text:
        ax.set_ylim(-1.6, 2)
        ax.text(0, 1.7, text, size=20)
    ax.set_xlabel('$X$', size=15)
    ax.set_ylabel('$Y$', rotation=0, size=15)
    if label: ax.legend()
    return ax

# generate polynomial basis
def PolynomialFeature(X, degrees):
    
    X_poly = X.copy()
    
    for i in range(2, degrees+2):
        X_poly = np.append(X_poly, X**i, axis=1)
    
    return X_poly


class Least_Squares:
    
    def __init__(self, basis=None, scale=True, degree=None, b_mu=None, b_s=None):
        self.basis = basis
        self.scale = scale
        self.degree = degree
        self.b_mu = b_mu
        self.b_s = b_s

    def one_of_K_coding(self, y):
        ys = len(np.unique(y))
        N = len(y)
        y_encoded = np.zeros((N, ys))
        for i in range(N):
            y_encoded[i][y[i]] = 1
        return y_encoded
    
    def _scale_fit(self, X, L=-1, H=1):
        P = X.shape[1]
        self.a = []; self.b = []
        for p in range(P):
            self.a.append(((H * min(X[:, p])) - (L * max(X[:, p])))/(H-L))
            self.b.append((max(X[:, p]) - min(X[:, p]))/(H - L))

    def _scale_transform(self, X):
        X_scaled = (X - self.a)/self.b
        return X_scaled.reshape(X.shape[0], -1)
    
    def _scale_inverse(X):
        return (X * self.b) + self.a
    
    def fit(self, X:np.ndarray, y:np.ndarray):
        
        if self.scale:
            self._scale_fit(X)
            X = self._scale_transform(X)
        
        X = np.append(X, np.ones((X.shape[0], 1)), axis=1)
        
        y = y.reshape(-1, 1)
        self.y = self.one_of_K_coding(y)
        
        self.W = np.linalg.pinv(X).dot( self.y)

    def predict(self, X:np.ndarray):
        if self.scale:
            X = self._scale_transform(X)
        X = np.append(X, np.ones((X.shape[0], 1)), axis=1)
        return np.argmax(X.dot(self.W), axis=-1)

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
    
    if ax:
        ax.contourf(xx, yy, Z)
        ax.set_title(title)
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
    else:
        plt.figure(figsize=(12, 6))
        plt.title(title)
        plt.contourf(xx, yy, Z)
        plt.xlabel('X1')
        plt.ylabel('X2')
    # create scatter plot for samples from each class
    for class_value in range(2):
        # get row indexes for samples with this class
        row_ix = np.where(y == class_value)
        # create scatter of these samples
        if ax: ax.scatter(X[row_ix, 0], X[row_ix, 1])
        else : plt.scatter(X[row_ix, 0], X[row_ix, 1])            
    if ax: return ax
    plt.show()
    
def plot_DB(X, y, model, title='', h=0.01):
    xx, yy, Z = create_mesh(X, model, h)
    fig = plt.figure(figsize=(12, 6))
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], 'ro')
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], 'bo')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title(title)
    plt.contour(xx, yy, Z)
    plt.show()
    
#########################################
#            Trainer Class              #
#########################################

class Trainer:
    
    def __init__(self, model, optimizer, criterion, n_epochs, scheduler=None, load_path=None):
        self.__class__.__name__ = "PyTorch Trainer"
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.n_epochs = n_epochs
        
        # if model exist
        if load_path:
            self.model = torch.load(load_path)
        
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        
    def run(self, train_loader, val_loader=None):
        min_valid_loss = np.inf
        ## Setup Metric class
        Metric = namedtuple('Metric', ['loss', 'agv_train_error', 'avg_val_error'])
        self.metrics = []
        self.model.train() 
        min_valid_loss = np.inf
#         np.mean([self.criterion(self.model(g),l).item() for g, l in val_loader])
        for epoch in range(self.n_epochs):
            train_loss = 0.0
            lr = self.scheduler.get_last_lr()[0] if self.scheduler else\
                 self.optimizer.param_groups[0]['lr']
            data_iter = iter(train_loader)
            prog_bar = tqdm(range(len(train_loader)))
            for step in prog_bar: # iter over batches

                ######################
                # Get the data ready #
                ######################
                # get the input images and their corresponding labels
                X, y = next(data_iter)
                X, y = X.float(), y.float()
                
                ################
                # Forward Pass #
                ################
                # Feed input images
                y_hat = self.model(X.float()).view_as(y)
                # Find the Loss
                loss = self.criterion(y, y_hat)

                #################
                # Backward Pass #
                #################
                # Calculate gradients
                loss.backward()
                # Update Weights
                self.optimizer.step()
                # clear the gradient
                self.optimizer.zero_grad()
                
                #################
                # Training Logs #
                #################
                # Calculate total Loss
                train_loss += loss.item()
                # Calculate total samples
                
                prog_bar.set_description('Epoch {}/{}, Loss: {:.4f}, lr={:.2f}'.format(epoch+1, self.n_epochs, loss.item(),lr))
                
                del X; del y_hat; del loss
                
            if val_loader:
                valid_loss = 0.0
                self.model.eval() # Optional when not using Model Specific layer
                with torch.no_grad():
                    for X, y in (val_prog_bar := tqdm(val_loader)):
                        # Forward Pass
                        y_hat = self.model(y)
                        # Find the Loss
                        loss = self.criterion(y, y_hat)

                        # Calculate Loss
                        valid_loss += loss.item()

                        val_prog_bar.set_description('Validation, Loss: {:.4f}'\
                                                     .format(epoch+1, self.n_epochs, loss.item()))

                #Check point
                if min_valid_loss > valid_loss:
                    print('Validation Loss Decreased ({:.6f} ===> {:.6f}) \nSaving The Model'.format(min_valid_loss/len(val_loader), 
                                                                                                     valid_loss/len(val_loader)))

                    min_valid_loss = valid_loss/len(val_loader)

                self.metrics.append(Metric(loss=train_loss, 
                                           agv_train_error=train_loss/len(train_loader.dataset),
                                           avg_val_error=valid_loss/len(train_loader.dataset)))

            # Decrease the lr
            if self.scheduler:
                scheduler.step()
                

class NpDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = np.double(y)
    def __len__(self): 
        return len(self.X)
    def __getitem__(self, i): 
        return torch.tensor(self.X[i]), torch.tensor(self.y[i])