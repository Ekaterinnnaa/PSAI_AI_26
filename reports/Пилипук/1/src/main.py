import numpy as np
import matplotlib.pyplot as plt

X = np.array([[6, 1], [-6, 1], [6, -1], [-6, -1]])
y = np.array([0, 0, 0, 1]).reshape(-1, 1)

X_bias = np.c_[X, np.ones(X.shape[0])]

def predict(X_bias, weights):
    return X_bias @ weights

def MSE(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def MSEGrad(X_bias, y_true, y_pred):
    N = len(y_true)
    return (2 / N) * X_bias.T @ (y_pred - y_true)

def train(X_bias, y, eta=0.01, epochs=10000, tol=1e-6):
    weights = np.zeros((X_bias.shape[1], 1))
    history = []
    
    for epoch in range(epochs):
        y_pred = predict(X_bias, weights)
        loss = MSE(y, y_pred)
        history.append(loss)
        
        grad = MSEGrad  (X_bias, y, y_pred)
        weights -= eta * grad
        
        if epoch > 0 and abs(history[-1] - history[-2]) < tol:
            print(f"convergence reached at epoch {epoch} for eta={eta}")
            break
    
    return weights, history

etas = [0.01, 0.001, 0.0001]
histories = []
weights_list = []

for eta in etas:
    weights, history = train(X_bias, y, eta=eta)
    weights_list.append(weights)
    histories.append(history)
    print(f"weights for eta={eta}: {weights.flatten()}")

best_weights = weights_list[0]

plt.figure(figsize=(10, 6))
for hist, eta in zip(histories, etas):
    plt.plot(hist, label=f'eta={eta}')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.title('MSE dependency from diffs eta`s')
plt.legend()
plt.grid(True)
plt.show()

def graph(X, y, weights, new_points=None, new_classes=None):
    plt.figure(figsize=(8, 6))
    
    class0 = X[y.flatten() == 0]
    class1 = X[y.flatten() == 1]
    plt.scatter(class0[:, 0], class0[:, 1], color='blue', label='Class 0')
    plt.scatter(class1[:, 0], class1[:, 1], color='red', label='Class 1')
    
    w1, w2, b = weights.flatten()
    x1_vals = np.linspace(-7, 7, 100)
    if w2 != 0:
        x2_vals = (0.5 - b - w1 * x1_vals) / w2
        plt.plot(x1_vals, x2_vals, color='green', label='Border (\\hat{y}=0.5)')
    else:
        print("w2=0, vertical line - rare situation")
    
    if new_points is not None and new_classes is not None:
        new_points = np.array(new_points)
        for i, cls in enumerate(new_classes):
            color = 'blue' if cls == 0 else 'red'
            plt.scatter(new_points[i, 0], new_points[i, 1], color=color, marker='x', s=100, label=f'New Point {i+1} (Class {cls})' if i == 0 else None)
    
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Data, border line and other dots')
    plt.legend()
    plt.grid(True)
    plt.show()

graph(X, y, best_weights)

def new_point(x1, x2, weights, threshold=0.5):
    x_bias = np.array([x1, x2, 1]).reshape(1, -1)
    y_pred = predict(x_bias, weights)
    class_label = 1 if y_pred > threshold else 0
    return y_pred[0][0], class_label

new_points = []
new_classes = []

print("\nFunctionality mod: enter 2 value in [-6, 6] or 'exit'.")
while True:
    try:
        input_str = input("Enter x1 and x2: ").strip()
        if input_str.lower() == 'exit':
            break
        x1, x2 = map(float, input_str.split())
        if not (-6 <= x1 <= 6 and -6 <= x2 <= 6):
            print("Caution: inputs not in [-6, 6]. There can be anomalies")
        
        pred, cls = new_point(x1, x2, best_weights)
        print(f"Point ({x1}, {x2}): prediction {pred:.4f}, class {cls}")
        
        new_points.append([x1, x2])
        new_classes.append(cls)
        graph(X, y, best_weights, new_points, new_classes)
    except ValueError:
        print("Error.")

