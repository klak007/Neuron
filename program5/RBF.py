from scipy import *
from scipy.linalg import norm, pinv
 
from matplotlib import pyplot as plt

from numpy.linalg import cond,det
from scipy import *
from scipy.linalg import norm, pinv, svd, inv

from numpy.linalg import cond,det
 
from scipy.spatial.distance import cdist

LAMBDA=0.0

def gendist(X,Y):
  return cdist(X,Y,metric='euclidean')

    
class RBF:
     
    def __init__(self, centers, outdim, R):
        indim = centers.shape[1]
        numCenters = centers.shape[0]
        self.indim = indim
        self.outdim = outdim
        self.numCenters = numCenters
        self.centers = centers
        self.R = R
        self.W = np.random.random((self.numCenters, self.outdim))
        self.TRAINING_X_DATA = []
        
        
        
    def _basisfunc(self, c, d):
        assert len(d) == self.indim
        return np.exp(-1/(self.R)**2 * norm(c-d)**2)
    
    
    def _basisfuncFast(self, c, d, dist_mat):
        print("_basisfuncFast ---> STARTED")
        ret = np.exp(-1/(self.R)**2 * dist_mat**2)
        print("_basisfuncFast ---> ENDED")
        return ret
     
    def _calcAct(self, X):
        # calculate activations of RBFs
        G = np.zeros((X.shape[0], self.numCenters), float)
        for ci, c in enumerate(self.centers):
            for xi, x in enumerate(X):
                G[xi,ci] = self._basisfunc(c, x)
        
       
        if G.shape[0] == G.shape[1]:
           print('det(G) = ', det(G))
        
        return G
    
    
    def _calcActFast(self, X):
        G = np.zeros((X.shape[0], self.numCenters), float)

        self.centers = np.vstack(self.centers[:,:]).astype(np.float64)
        X = np.vstack(X[:,:]).astype(np.float64)

        dist_mat = gendist(self.centers, X)
        print("distance matrix ...  beeing calculated - STARTED")
        G = self._basisfuncFast(self.centers, X, dist_mat)
        print("distance matrix ...  beeing calculated - ENDEDED")

        print('X.shape[0]=',X.shape[0], 'self.numCenters=',self.numCenters)
       
        if G.shape[0] == G.shape[1]:
           print('det(G) = ', det(G))
        
        return G

    def wypiszZbior(self, X, tekst):
        print(tekst)
        for i in range(X.shape[0]):
            print(i,X[i,:])
        
    def train(self, X, Y):
        """ X: matrix of dimensions n x indim 
            y: column vector of dimension n x 1 """
         
        self.TRAINING_X_DATA = X
        
        self.wypiszZbior(X,'ZBIOR TRENINGOWY ----------------------->')
        G = self._calcAct(X)
         
        # calculate output weights (pseudoinverse)
        self.W = np.dot(pinv(G), Y)

        
    def trainRegularized(self, X, Y, _lambda=0):
        print("Regularized RBF training.")
        self.wypiszZbior(X,'ZBIOR TRENINGOWY ----------------------->')
        
        self.TRAINING_X_DATA = X
        
        G = self._calcActFast(X)
        U, S, VT = svd(G)
        self.wartosci_szczegolne = S
        
        S_inv = S*S-_lambda
        
        print('VT.shape=',VT.shape)
        print('S.shape=',S.shape)
        print('U.shape=',U.shape)
        
        N = G.shape[0]
        
        r = N
        for i in range(N):
            if S[i] < _lambda:
                r = i
                break
                
        W_final = np.zeros((G.shape[0], Y.shape[1]))
    
        for ii in range(Y.shape[1]):
            w_lambda = np.zeros(G.shape[0])
            for i in range(r):
                sigma = S[i]
                uTi = U[:,i].transpose()
                y = Y[:,ii]
                vi = VT.transpose()[:,i]
                f = sigma**2/(sigma**2+_lambda**2)
                w_lambda += 1/sigma*f*np.dot(np.dot(uTi,y),vi)
            W_final[:,ii] = w_lambda

        
        print('r=',r)
        
        self.W = W_final
        


    
    def test(self, X, treningowy = False):
        """ X: matrix of dimensions n x indim """
        
        G = self._calcAct(X)
        Y = np.dot(G, self.W)
        return Y




rbf = RBF(centers, nout, r/proporcja)
rbf.train(XX_train, YY_train)

rbf.test(XX_test)
