from data import read_data #python 3.7, tensorflow 2.0, scikit-learn, statsmodels
from math import floor, ceil
import joblib
import time

import math
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import seed
seed(1)
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

import tensorflow
tensorflow.random.set_seed(1)
from tensorflow.python.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

start=time.time()

# Se leen los datos
data = read_data(path='./simulaciones/', file='energizacion', n_sim=9104)
no = 4 # numero de outputs en la data

# Se dividen los datos en entradas (data_x) y salidas (data_y)
nc = len(data.columns) # numero total de columnas en la data
data_x = data.iloc[:, 0:nc-no].to_numpy()
data_y = data.iloc[:, -no:].to_numpy() # -no: significa que se van a predecir las 3 salidas (Vmax_fn_A, Vmax_fn_B, Vmax_ff_AB)

# Se dividen los datos en conjuntos de entrenamiento y validación
train_size = 0.8

train_cnt = floor(data_x.shape[0] * train_size)
X_train = data_x[0:train_cnt,:]
y_train = data_y[0:train_cnt]
X_val = data_x[train_cnt:,:]
y_val = data_y[train_cnt:]

scaler_x = MinMaxScaler()
scaler_x_val = MinMaxScaler()
scaler_y = MinMaxScaler()
scaler_y_val = MinMaxScaler()

print(scaler_x.fit(X_train))
xtrain_scale = scaler_x.transform(X_train)

print(scaler_x_val.fit(X_val))
xval_scale = scaler_x_val.transform(X_val)

print(scaler_y.fit(y_train))
ytrain_scale = scaler_y.transform(y_train)

print(scaler_y_val.fit(y_val))
yval_scale = scaler_y_val.transform(y_val)

# Se construye la red neuronal
n_inp = int(nc-no)
n_out = no
n_hidden = train_cnt*(n_inp+n_out)

model = Sequential()
model.add(Dense(n_inp, input_dim=n_inp, kernel_initializer='normal', activation='relu'))
model.add(Dense(n_hidden, activation='relu'))
model.add(Dense(n_out, activation='linear'))
model.summary()
model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])
history=model.fit(xtrain_scale, ytrain_scale, epochs=100, batch_size=150, verbose=1, validation_split=0.2)
predictions = model.predict(xval_scale)

# Se grafica el error a través de las épocas
print(history.history.keys())
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# Regresamos a la escala original
predictions = scaler_y.inverse_transform(predictions)

# Guardamos la red entrenada y el escalador MinMax
model.save('red_entrenada')
joblib.dump(scaler_x, './escalador/MinMaxScaler_x')
joblib.dump(scaler_y, './escalador/MinMaxScaler_y')

# Predecir un solo valor
# prueba = np.array([data.iloc[100,0:nc-no],])
def sobretensiones(n): #retorna [verdadero,predicción]
    prueba = np.array([data.iloc[n,0:nc-no],])
    prueba_scale = scaler_x.transform(prueba)
    prueba_pred = model.predict(prueba_scale)
    return {'Verdaderos': data.iloc[n,-no:].to_numpy(),
            'Predicciones': scaler_y.inverse_transform(prueba_pred)[0]}

e1=np.mean(100*np.divide(np.abs(y_val[:,0]+np.multiply(predictions[:,0],-1)),y_val[:,0]))
e2=np.mean(100*np.divide(np.abs(y_val[:,1]+np.multiply(predictions[:,1],-1)),y_val[:,1]))
e3=np.mean(100*np.divide(np.abs(y_val[:,2]+np.multiply(predictions[:,2],-1)),y_val[:,2]))
e4=np.mean(100*np.divide(np.abs(y_val[:,3]+np.multiply(predictions[:,3],-1)),y_val[:,3]))

print(f'Error FN Emisor: {round(e1,2)}%')
print(f'Error FF Emisor: {round(e2,2)}%')
print(f'Error FN Receptor: {round(e3,2)}%')
print(f'Error FF Receptor: {round(e4,2)}%')
print(f'Error promedio de la red: {round(np.mean([e1,e2,e3,e4]),2)}%')

end = time.time()

print(f'Tiempo de procesamiento y entrenamiento: {end-start}[seg]')