import math
import numpy as np
from linear_regression import *
from sklearn.datasets import make_regression
# Note: please don't add any new package, you should solve this problem using only the packages above.
#-------------------------------------------------------------------------
'''
    Problem 2: Apply your Linear Regression
    In this problem, use your linear regression method implemented in problem 1 to do the prediction.
    Play with parameters alpha and number of epoch to make sure your test loss is smaller than 1e-2.
    Report your parameter, your train_loss and test_loss 
    Note: please don't use any existing package for linear regression problem, use your own version.
'''

#--------------------------

n_samples = 200
X,y = make_regression(n_samples= n_samples, n_features=4, random_state=1)
y = np.array(y).T
X = np.array(X)
Xtrain, Ytrain, Xtest, Ytest = X[::2], y[::2], X[1::2], y[1::2]

#########################################
## INSERT YOUR CODE HERE
alpha_values = [0.001, 0.01, 0.1]
n_epochs = [100, 500, 1000]
results = []
for alpha in alpha_values:
    for n_epoch in n_epochs:
        w = train(Xtrain, Ytrain, alpha=alpha, n_epoch=n_epoch)
        yhat_train = compute_yhat(Xtrain, w)
        yhat_test = compute_yhat(Xtest, w)
        tr_loss = compute_L(yhat_train, Ytrain)
        te_loss = compute_L(yhat_test, Ytest)
        results.append((alpha, n_epoch, tr_loss, te_loss))

for alpha, n_epoch, tr_loss, te_loss in results:
    print(f"Alpha: {alpha}, Epochs: {n_epoch}, Train Loss: {tr_loss:.4f}, Test Loss: {te_loss:.4f}")

for alpha, n_epoch, tr_loss, te_loss in results:
    if te_loss < 0.01:
        print(f"Optimal configuration: Alpha: {alpha}, Epochs: {n_epoch}, Train Loss: {tr_loss:.4f}, Test Loss: {te_loss:.4f}")
#########################################