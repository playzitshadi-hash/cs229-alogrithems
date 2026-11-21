import numpy as np
import matplotlib.pyplot as plt
X = np.array([
    [-1.42, 0.85], [0.63, -1.12], [1.18, 0.44], [-0.75, -0.63], [2.05, 1.30],
    [-1.90, -1.45], [0.15, 1.72], [-0.38, -0.95], [1.54, -0.28], [-1.12, 1.18],
    [0.88, 0.76], [-1.65, -0.42], [1.92, -1.55], [-0.22, 0.55], [0.47, -1.80],
    [-0.94, 0.21], [1.31, 1.08], [-1.28, -1.20], [0.72, 0.12], [1.68, -0.85]
], dtype=np.float64)

y = np.array([
    -6.48, 6.51, 6.02, -0.74, 8.05,
    -4.32, -1.18, 1.65, 8.95, -5.89,
    3.92, -5.12, 13.20, -0.58, 7.18,
    -3.15, 5.24, -1.75, 4.48, 10.85
], dtype=np.float64)
X_train = X[:16]
y_train = y[:16]


X_test = X[16:]
y_test = y[16:]
learning_rate = 0.01
epochs = 1000
w0 = 0.0
w1 = 0.0
w2 = 0.0

# --- 1. TRAINING LOOP ---
for epoch in range(epochs):
    indices = np.random.permutation(len(X_train))
    for i in indices:
        x1 = X_train[i,0]
        x2 = X_train[i,1]
        y_true = y_train[i]
        
        y_pred = w0 + (w1*x1) + (w2 * x2)
        loss = y_pred - y_true
        
        grad_0 = loss * 1
        grad_1 = loss * x1
        grad_2 = loss * x2
        
        w0 = w0 - (learning_rate * grad_0)
        w1 = w1 - (learning_rate * grad_1)
        w2 = w2 - (learning_rate * grad_2)

# --- 2. TESTING (OUTSIDE THE TRAINING LOOP) ---
print("Testing on unseen data:")
test_predictions = [] # Create an empty list to store predictions

for i in range(len(X_test)):
    x1_test = X_test[i, 0]
    x2_test = X_test[i, 1]
    true_val = y_test[i]

    # Forward pass with the FINAL learned weights
    pred = w0 + (w1 * x1_test) + (w2 * x2_test)
    test_predictions.append(pred) # Save it to the list
    
    print(f"True: {true_val:.2f} | Predicted: {pred:.2f}")

# --- 3. CALCULATE MSE ---
test_predictions = np.array(test_predictions) # Convert list to array
mse = np.mean((test_predictions - y_test)**2) # Compare the 4 predictions to the 4 true values

print(f"\nFinal Weights: w0={w0:.4f}, w1={w1:.4f}, w2={w2:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")