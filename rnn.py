# Recurrent Neural Network
# Part 1 - Data Preprocessing
# Importing the libraries


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling
sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
timesteps = 60
X_train = []
y_train = []

for i in range(timesteps, len(training_set_scaled)):
    X_train.append(training_set_scaled[i - timesteps : i, 0])
    y_train.append(training_set_scaled[i,0])

X_train, y_train = np.array(X_train) , np.array(y_train)
# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 64, 
                   return_sequences = True, #connect backflow to next layer
                   input_shape = (X_train.shape[1], 1)  #tell size of inputs
                   ))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 64, 
                   return_sequences = True)) #connect backflow to next layer
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 64, 
                   return_sequences = True)) #connect backflow to next layer
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 64, 
                   return_sequences = False))  #layer will not be getting future feedback
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer='adam', loss = 'mean_squared_error', metrics = 'accuracy')

# Fitting the RNN to the Training set
regressor.fit(x = X_train, y = y_train, epochs=200, batch_size = 32 )


# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[-81:-1].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

X_test = []

for i in range(timesteps, len(inputs)):
    X_test.append(inputs[i - timesteps : i, 0])

X_test = np.array(X_test) 
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price =sc.inverse_transform(predicted_stock_price)


# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = "Predicted Google Stock Price")
plt.title("Comparison of Real vs. Predicted Google Stock Prices Jan 2017")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.legend()
plt.show()








