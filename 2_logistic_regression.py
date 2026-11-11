import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix



#logistic Regression
X_large = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 
                  5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0])

# New labels (Notice the noise: some low hours passed, some high hours failed)
y_large = np.array([0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 
                  1, 1, 1, 0, 1, 1, 1, 0, 1, 1])
np.random.seed(42)
shuffled_indices = np.random.permutation(len(X_large))  # Shuffle 0-19
train_idx = shuffled_indices[:14]
test_idx = shuffled_indices[14:]
X_train,X_test,y_train,y_test = train_test_split(X_large,y_large,test_size=0.2,random_state=42)
X_train = X_large[train_idx]
y_train = y_large[train_idx]
X_test = X_large[test_idx ]
y_test = y_large[test_idx ]
X_train_reshaped = X_train.reshape(-1, 1)
X_test_reshaped = X_test.reshape(-1, 1)
model =LogisticRegression()
model.fit(X_train_reshaped,y_train)
predictions = model.predict(X_test_reshaped)



theta0_test = 0
theta1_test = 0
alpha =0.1
for i in range(1000):
    z = theta0_test + (theta1_test * X_train)
    p = 1 / (1 + np.exp(-z))  # Sigmoid
    p_clipped = np.clip(p, 1e-9, 1 - 1e-9) 
    m =len(X_train)
    loss_logistic = -(1/m) * np.sum(y_train * np.log(p_clipped) + (1-y_train) * np.log(1-p_clipped))
    a0 = np.mean(p_clipped-y_train)
    a1=np.mean((p_clipped-y_train)*X_train)
    theta0_test = theta0_test - alpha * a0
    theta1_test = theta1_test - alpha * a1
    if i % 100 == 0:
        print(f"Iteration {i}: Loss = {loss_logistic:.4f}, theta0 = {theta0_test:.4f}, theta1 = {theta1_test:.4f}")

print(f"Final Loss: {loss_logistic}")





z_train = theta0_test + theta1_test * X_test
p_train = 1 / (1 + np.exp(-z_train))

pred_train = (p_train >= 0.5).astype(int)

accuracy = np.mean(pred_train == y_test)
cor_p = np.sum(pred_train == y_test)
print(f"Test accuracy: {accuracy:.1%}") 
accuracy = np.mean(predictions == y_test)
print(f"Scikit-Learn Test Accuracy: {accuracy:.1%}")
print("Test size : 6 Correct predictions",cor_p )
print(pred_train)
print(predictions)
print(np.array_equal(pred_train, predictions))
cm = confusion_matrix(y_test, pred_train)
print(cm)

tp = 0
tn = 0
fp = 0
fn = 0
for  actual, predicted in zip(y_test,pred_train):
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
if (tp+ fp) == 0:
 Precision=0
else:
 Precision = tp/ (tp+fp)
 print(Precision)
if (tp+fn)== 0:
   recall =0
else: 
 recall =tp/(tp+fn)
 print(recall)
if (Precision + recall) == 0:
   f1_score =0
else: 
 f1_score = 2 *(Precision *recall) / (Precision + recall)
 print(f1_score)
print(tn+tp+fp+fn)
plt.scatter(X_train,y_train)
plt.plot(X_test,p_train)
plt.show()
