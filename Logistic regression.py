# User defined logistic regression

# packages
import numpy as np
import pandas as pd
from patsy import dmatrices
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score,confusion_matrix

# data
data = pd.read_csv("HR_comma_sep.csv")

y, X = dmatrices('left~satisfaction_level+last_evaluation+number_project+average_montly_hours+time_spend_company+Work_accident+promotion_last_5years+C(sales)+C(salary)', data, return_type='dataframe')
X = np.asmatrix(X)
y = np.ravel(y)

# Normalization by columns 
for i in range(1, X.shape[1]):
    xmin = X[:,i].min()
    xmax = X[:,i].max()
    X[:, i] = (X[:, i] - xmin) / (xmax - xmin)

# Training model
np.random.seed(1)
alpha = 2  # set learning rate
beta = np.random.randn(X.shape[1]) # randomly initialize parameters (beta) 
losses = []
error_rates = []
for T in range(200):
    prob = np.array(1. / (1 + np.exp(-np.matmul(X, beta)))).ravel()  # calculate the current probability 
    prob_y = list(zip(prob, y))
    loss = -sum([np.log(p) if y == 1 else np.log(1 - p) for p, y in prob_y]) / len(y) # cross entropy: calculate the current loss
    error_rate = 0
    for i in range(len(y)):
        if ((prob[i] > 0.5 and y[i] == 0) or (prob[i] <= 0.5 and y[i] == 1)):
            error_rate += 1;
    error_rate /= len(y)
    losses.append(loss)
    error_rates.append(error_rate)
    
    # gradient descend
    if T % 5 ==0 :
        print('T=' + str(T) + ' loss=' + str(loss) + ' error=' + str(error_rate))
    # calculate the derivatives of loss in terms of each component
    deriv = np.zeros(X.shape[1])
    for i in range(len(y)):
        deriv += np.asarray(X[i,:]).ravel() * (prob[i] - y[i])
    deriv /= len(y)
    # update parameters in the opposite direction of the derivative
    beta -= alpha * deriv


prob = np.array(1. / (1 + np.exp(-np.matmul(X, beta)))).ravel()
prediction = 1*(prob>0.5)
print ("AUC Score (test): %f" % roc_auc_score(y,prob))
print("Precison is",precision_score(y, prediction, average='binary'))
print("Recall is",recall_score(y, prediction , average='binary'))
print("F1 score is", f1_score(y, prediction , average='binary'))
print(confusion_matrix(y, prediction))

plt.plot(range(200), error_rates,'b^') # error_rate
plt.show()
plt.plot(range(200), losses, 'r*')
plt.show()
