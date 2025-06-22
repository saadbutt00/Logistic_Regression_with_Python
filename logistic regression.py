import numpy as np

print('''It is Logistic Regression (Sigmoid) which takes 2 values. So, always assign numerical values to categorical tags. 
For Example: red(0) and green(1). So, when prediction comes 1, you can easily understand it is green.\n''')

num_feature = int(input('Enter Number of Features - '))
print('')

X = []

for i in range(num_feature):
    a = np.array(list(map(float, input(f'Enter Feature X{i} values (space-separated) : ').split())))
    X.append(a)

Y = np.array(list(map(int, input('Enter Feature Y values : (space-separated & should be 0 or 1) : ').split())))
print('')

m = len(Y)

for e in X:
    if len(e) != m:
        print('Error - Number of values should be Equal')
        exit()

w = np.zeros(num_feature)  
b = 0.0

epoch = int(input('Enter Number of Epochs - '))
lr = float(input('Enter Learning Rate - '))

Xt = np.transpose(X)  

for i in range(epoch):
    z = np.dot(Xt, w) + b
    y_pred = 1 / (1 + np.exp(-z))

    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.mean(Y * np.log(y_pred) + (1 - Y) * np.log(1 - y_pred))

    dw = np.dot(X, (y_pred - Y)) / m
    db = np.sum(y_pred - Y) / m

    w -= lr * dw
    b -= lr * db

    if i % 100 == 0:
        print(f'__________________________Epoch {i}_________________________')
        print(f'Loss: {loss:.4f}')
        print(f'w: {np.round(w, 4)}')
        print(f'b: {b:.4f}')

print("\nTrained w:", np.round(*w, 4))
print("Trained b:", round(b, 4))

z = np.dot(Xt, w) + b
y_pred = 1 / (1 + np.exp(-z))
print("\nPredictions:", np.round(y_pred, 4))
pred_label = (y_pred > 0.50).astype(int)
print('Prediction Labels -', pred_label)