import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
rng =np.random.default_rng()
single_val = rng.uniform()
X =rng.uniform(low=0.0,high=10.0, size=150)
noise_1 = rng.normal(loc=0.0, scale=0.1,size=100)
noise_2 = rng.normal(loc=0.0, scale=0.1,size=50)
X_train = X[:100]
X_test =X[-50:]
y_train = np.sin(X_train)+noise_1
y_test = np.sin(X_test) + noise_2
degree = 1
mean_train = X_train.mean()
std_train = X_train.std()
X_train_scaled=( X_train-mean_train )/std_train
X_test_scaled=(X_test-mean_train )/std_train
train_errors = []
test_errors = []
train_mses = []
test_mses = []
for degree in (1, 3, 15):
    coefficient=np.polyfit(X_train_scaled,y_train,degree)
    y_trainpred=np.polyval(coefficient,X_train_scaled)
    y_testpred=np.polyval(coefficient,X_test_scaled)
    error0=(  y_trainpred-y_train)
    error1=(y_testpred-y_test)
    mean0 = (error0**2).mean()
    train_errors.append(mean0)
    mean1 = (error1**2).mean()
    test_errors.append(mean1)
    mse_new = np.mean((y_train - y_trainpred)**2)
    train_mses.append(mse_new)
    mse_test = np.mean((y_test - y_testpred)**2)
    test_mses.append(mse_test)
print(f" Train MSES: {train_mses}\n")
print(f"Test MSES: {test_mses}\n")
    
X_axis =[1, 3, 15]
y_trainaxis = train_errors
y_testaxis = test_errors
# plt.plot(X_axis,y_trainaxis, color='blue')
# plt.plot(X_axis,y_testaxis,color="red")
# plt.show()
X_design = np.vander(X_train_scaled, 15, increasing=True)
X_test_design = np.vander(X_test_scaled, 15, increasing=True)
X_transpose_X = X_design.T @ X_design
p = [0.001, 0.01, 0.1, 1, 10, 100]
ridge_train_mses = []
ridge_test_mses = []
for i in p:
    I = np.eye(15)
    inverse_mat = (X_transpose_X + i * I)
    theta = np.linalg.inv(inverse_mat) @ X_design.T @ y_train
    predictions = X_design @theta 
    test_predictions = X_test_design @ theta
    Mse = np.mean((y_train -predictions)**2)
    ridge_train_mses.append(Mse)
    Mse_test = np.mean((y_test -test_predictions)**2)
    ridge_test_mses.append(Mse_test)
    print(f"Ridge Train MSE :{Mse}\n")
    print(f"Ridge Test MSE :{Mse_test}\n")
plt.semilogx(p,ridge_test_mses, color='blue')
plt.semilogx(p,ridge_train_mses, color='black')
plt.show()