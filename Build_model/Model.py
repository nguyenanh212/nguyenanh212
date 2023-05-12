import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split



# load data
df = pd.read_csv('data/data_hour_custom.csv')

X = df.iloc[:, 0:9].values
y = df.iloc[:, 8].values

print(X.shape)
print(y.shape)


# shift y prediction by 1-5 hours and stack into a numpy array
label = []
n = -5
for i in range(1,6):

  y_shifted = np.roll(y, -i)
  label.append(y_shifted)
label = np.array(label).T

# shuffle data

for i in range(len(label)):
  k = np.random.randint(0, len(label))
  label[i], label[k] = label[k], label[i]
  X[i], X[k] = X[k], X[i]


X = X[:n, :]
label = label[:n, :]

x_std = np.std(X,axis =  0)
x_mean = np.mean(X,axis = 0)

y_mean = np.mean(label)
y_std = np.std(label)

X = (X - x_mean)/x_std

label = (label - y_mean) / y_std



X = np.array(X)
label = np.array(label)



# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.2,random_state=50)


# reshape input to fit LSTM model
X_train_lstm = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test_lstm = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(64, input_shape = (9, 1), activation='relu', return_sequences=True),
    tf.keras.layers.LSTM(64, activation='relu'),
    tf.keras.layers.Dense(64, activation = 'relu'),
    tf.keras.layers.Dense(32, activation = 'relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(5) 
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

model.fit(X_train_lstm, y_train, epochs= 20, batch_size= 128, validation_split=0.2)

model.evaluate(X_test_lstm, y_test)

model.save('model_temp.h5')