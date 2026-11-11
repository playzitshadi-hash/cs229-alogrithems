import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

class0 = [2, 2]
class1 =[5, 5]
cov = [[1.0, 0.5], [0.5, 1.0]]
samples=np.random.multivariate_normal(class0, cov, size=100,)
samples1=np.random.multivariate_normal(class1, cov, size=100,)
zeros = np.zeros(100)
ones= np.ones(100)
X = np.vstack((samples,samples1))
y = np.hstack((zeros, ones))
y_t = y.T
plt.scatter(X,y_t)
plt.show()
m = len(y)
predictions=[]
phi = 1/m *(y==1).sum()
mean0 = ((y == 0)[:, np.newaxis] * X).sum(axis=0) / (y == 0).sum()
mean1=((y == 1)[:, np.newaxis] * X).sum(axis=0) / (y == 1).sum()
means = np.array([mean0,mean1])
X_centered = np.copy(X)
X_centered[y == 0] = X_centered[y == 0] - mean0
X_centered[y == 1] = X_centered[y == 1] - mean1
cov_sum = X_centered.T @ X_centered
Sigma = (1 / m) * cov_sum
x_sample = X[0]
d=len(x_sample)
x_class0 = X[y == 0, 0]
y_class0 = X[y == 0, 1]

x_class1 = X[y == 1, 0]
y_class1 = X[y == 1, 1]

plt.scatter(x_class0, y_class0, color='blue', label='Class 0')
plt.scatter(x_class1, y_class1, color='red', label='Class 1')

def gaussian(x_sample,mean,Sigma):
    diff =  x_sample -mean
    denominator = (2 * np.pi) ** (d / 2) * np.sqrt(np.linalg.det(Sigma))
    constant = 1/denominator
    inner_math = -0.5 * (diff.T @ np.linalg.inv(Sigma) @ diff)
    probablity_density = constant*np.exp(inner_math)
    return probablity_density

probablity_density = gaussian(x_sample,mean0, Sigma)
print(probablity_density)
for x in X:
    dd = gaussian(x,mean0,Sigma)
    ff = gaussian(x,mean1,Sigma)
    score0 = dd * (1- phi)
    score1= ff * phi
    if score0 > score1:
     print("model predicts class 0")
     predictions.append(0)  # Append class label 0
    else:
     print("model predicts class 1")
     predictions.append(1)
accuracy=np.mean(predictions== y)
print(f"{accuracy *100}%")
tp = 0
tn = 0
fp = 0
fn = 0
for  actual, predicted in zip(y,predictions):
    if actual == 0 and predicted == 0:
        tn +=1
    elif actual == 1 and predicted == 1:
        tp +=1
    elif actual == 0 and predicted == 1:
        fp +=1
    elif  actual == 1 and predicted == 0:
        fn +=1
confusion_matrix =np.array([[tn,fp],
                           [fn,tp]])
print(confusion_matrix)



mean_plus = 0.5*(mean0+mean1)
mean_minus = (mean0-mean1)
inv_Sigma = np.linalg.inv(Sigma)
decision_boundary = mean_plus @ inv_Sigma @ mean_minus - np.log((1 - phi) / phi)
weights = mean_minus @ inv_Sigma 
x_min = X[:, 0].min()
x_max = X[:, 0].max()
num_steps = 500
x_range = np.linspace(x_min, x_max, num_steps)
w1= weights[0]
w2 = weights[1]
y_range = (decision_boundary - (w1 * x_range)) / w2
plt.plot(x_range,y_range)
plt.show()


