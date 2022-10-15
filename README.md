### EXP NO: 05

### .

# <p align = "center"> Stock Price Prediction </p>

## AIM

To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset


## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:
Import tensorflow library and preprocessing libraries.

### STEP 2:
Load the traning dataset and take one column values and scale it using minmaxscaler.

### STEP 3:
Split x_train(0-60 values) and y_train(61 st value).

### STEP 4:
Create a RNN model with required no of neurons with one output neuron.

### STEP 5:
Fit the model and load testing dataset.For x_test,combine the values of both datasets.

### STEP 6:
Follow the same splitting.Make the prediction and plot graph with mse value





## PROGRAM
```python
# Developed By:KUMARAN.B
# Register Number:212220230026
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import layers
from keras.models import Sequential
dataset_train = pd.read_csv('trainset.csv')
dataset_train.columns
dataset_train.head()
train_set = dataset_train.iloc[:,1:2].values
type(train_set)
train_set.shape
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(train_set)
training_set_scaled.shape
X_train_array = []
y_train_array = []
for i in range(60, 1259):
  X_train_array.append(training_set_scaled[i-60:i,0])
  y_train_array.append(training_set_scaled[i,0])
X_train, y_train = np.array(X_train_array), np.array(y_train_array)
X_train1 = X_train.reshape((X_train.shape[0], X_train.shape[1],1))
X_train.shape
model = Sequential()
model.add(layers.SimpleRNN(70,input_shape=(60,1)))
model.add(layers.Dense(1))
model.compile(optimizer='adam', loss='mae')
model.fit(X_train1,y_train,epochs=100, batch_size=32)
dataset_test = pd.read_csv('testset (1).csv')
test_set = dataset_test.iloc[:,1:2].values
test_set.shape
dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)
inputs = dataset_total.values
inputs = inputs.reshape(-1,1)
inputs_scaled=sc.transform(inputs)
X_test = []
y_test=[]
for i in range(60,1384):
  X_test.append(inputs_scaled[i-60:i,0])
  y_test.append(inputs_scaled[i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1],1))
X_test.shape
predicted_stock_price_scaled = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price_scaled)
plt.plot(np.arange(0,1384),inputs, color='red', label = 'Test(Real) Google stock price')
plt.plot(np.arange(60,1384),predicted_stock_price, color='blue', label = 'Predicted Google stock price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
from sklearn.metrics import mean_squared_error as mse
mse(y_test,predicted_stock_price)
```
## OUTPUT

### True Stock Price, Predicted Stock Price vs time
![Screenshot (446)](https://user-images.githubusercontent.com/75243072/195996829-82b6dce8-3da5-4bbf-af0e-2e1d529492d6.png)

### Mean Square Error
![Screenshot (445)](https://user-images.githubusercontent.com/75243072/195996823-2b267249-07e1-49b7-8a7c-2d9a69815e50.png)



## RESULT
Thus, a Recurrent Neural Network model for stock price prediction is developed.
