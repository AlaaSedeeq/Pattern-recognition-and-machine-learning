import numpy as np

class GaussNB:
    
    def fit(self, X, y):
        '''
        Fit the model tothe data
        '''
        # Calculate P(X|y)
        C = np.unique(y)
        Xy0 = X[y == y[0]]
        Xy1 = X[y == y[1]]
        
        # calculate P(y)
        self.priory0 = len(Xy0) / len(X)
        self.priory1 = len(Xy1) / len(X)
        
        # k distributions for each feature, k=number of classes
        # create PDFs for y==0
        self.distX1y0 = self.fit_distribution(Xy0[:, 0])
        self.distX2y0 = self.fit_distribution(Xy0[:, 1])
        # create PDFs for y==1
        self.distX1y1 = self.fit_distribution(Xy1[:, 0])
        self.distX2y1 = self.fit_distribution(Xy1[:, 1])

    
    def predict(self, X):
        '''
        Make predictions
        '''
        pred = [
            np.argmax([
                self.probability(x, self.priory0, self.distX1y0, self.distX2y0), 
                self.probability(x, self.priory1, self.distX1y1, self.distX2y1)
            ])
            for x in X]
        
        return pred
    
    def fit_distribution(self, data):
        from numpy import mean, std
        from scipy.stats import multivariate_normal, norm
        '''
        Estimate data parameters and fit a distribution
        '''
        # estimate parameters
        mu = mean(data)
        sigma = std(data)
        # fit distribution
        dist = norm(mu, sigma)
        return dist

    def probability(self, X, prior, dist1, dist2):
        '''
        Calculate the independent conditional probability
        '''
        return prior * (dist1.pdf(X[0]) * dist2.pdf(X[1]))