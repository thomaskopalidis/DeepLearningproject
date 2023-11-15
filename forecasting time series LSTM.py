# 1. Import Libraries and Load Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import InputLayer, LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)
csv_path, _ = os.path.splitext(zip_path)
df = pd.read_csv(csv_path)
print(df)

# 2. Data Preprocessing
df = df[5::6]
df.index = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S')

# 3. Data Visualization
temp = df['T (degC)']
temp.plot()
plt.show()

# 4. Data Transformation for LSTM
def df_to_X_y(df, window_size=5):
    df_as_np = df.to_numpy()
    X, y = [], []
    for i in range(len(df_as_np) - window_size):
        row = [[a] for a in df_as_np[i:i + window_size]]
        X.append(row)
        label = df_as_np[i + window_size]
        y.append(label)
    return np.array(X), np.array(y)

WINDOW_SIZE = 5
X, y = df_to_X_y(temp, WINDOW_SIZE)
X_train, y_train = X[:60000], y[:60000]
X_val, y_val = X[60000:65000], y[60000:65000]
X_test, y_test = X[65000:], y[65000:]
print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)

# 5. Building the LSTM Model
model1 = Sequential()
model1.add(InputLayer((WINDOW_SIZE, 1)))
model1.add(LSTM(64))
model1.add(Dense(16, 'relu'))
model1.add(Dense(8, 'relu'))
model1.add(Dense(1, 'linear'))
model1.summary()

# 6. Compiling and Training the Model
cp = ModelCheckpoint('model1/', save_best_only=True)
model1.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
model1.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, callbacks=[cp])

# 7. Making Predictions and Visualizing Results
# Training Data Predictions
train_predictions = model1.predict(X_train).flatten()
train_results = pd.DataFrame({'Train Predictions': train_predictions, 'Actuals': y_train})
print(train_results)
plt.plot(train_results['Train Predictions'][:100])
plt.plot(train_results['Actuals'][:100])
plt.show()

# Validation Data Predictions
val_predictions = model1.predict(X_val).flatten()
val_results = pd.DataFrame({'Val Predictions': val_predictions, 'Actuals': y_val})
print(val_results)
plt.plot(val_results['Val Predictions'][:100])
plt.plot(val_results['Actuals'][:100])
plt.show()

# Test Data Predictions
test_predictions = model1.predict(X_test).flatten()
test_results = pd.DataFrame({'Test Predictions': test_predictions, 'Actuals': y_test})
print(test_results)
plt.plot(test_results['Test Predictions'][:100])
plt.plot(test_results['Actuals'][:100])
plt.show()


# Evaluate on Training Data
train_loss, train_rmse = model1.evaluate(X_train, y_train, verbose=0)
print(f'Training Data - Loss: {train_loss}, RMSE: {train_rmse}')

# Evaluate on Validation Data
val_loss, val_rmse = model1.evaluate(X_val, y_val, verbose=0)
print(f'Validation Data - Loss: {val_loss}, RMSE: {val_rmse}')

# Evaluate on Test Data
test_loss, test_rmse = model1.evaluate(X_test, y_test, verbose=0)
print(f'Test Data - Loss: {test_loss}, RMSE: {test_rmse}')