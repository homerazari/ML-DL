# allocated appropriate amount of GPU for ML/DL
import tensorflow as tf
from keras import backend as k
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 1.0
k.tensorflow_backend.set_session(tf.Session(config=config))
# supporting libraries
import matplotlib.pyplot as plt
from pandas import DataFrame, read_csv, concat, Grouper, Series
from math import floor
from sklearn.preprocessing import MinMaxScaler, StandardScaler,RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV
from numpy import mean, std, array, sqrt, shape, asscalar, save, load, nan, zeros
from numpy.random import seed as sed
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error as mse
import pickle
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Dropout
from keras.layers.convolutional import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D
from keras.wrappers.scikit_learn import KerasRegressor
from keras.optimizers import Adagrad
from keras.constraints import maxnorm


# Load Data and convert to dataset

dataframe = read_csv('shampoo.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
dataframe_shifted = concat([dataframe.shift(1), dataframe], axis=1)
dataframe_shifted.columns = ['t-1', 't+1']
df_final = dataframe_shifted.drop(dataframe.index[0])
data_v = df_final.values
data_v = data_v.astype('float32')

# Power transform of source data

scaler = MinMaxScaler(feature_range=(0,1))
fit_data = scaler.fit(data_v.reshape(-1,1))

data_processed = fit_data.transform(data_v.reshape(-1,1))
data_processed = data_processed.reshape(shape(data_v))

# Split into train, test sets

train_size = int(0.67 * len(data_processed))
train, test = data_processed[0:train_size], data_processed[train_size:]
trainX, trainY = train[:,0], train[:,1]
testX, testY = test[:,0], test[:,1]

# Reshape data to fit LSTM input req
trainX = trainX.reshape(len(trainX),1,1)
trainY = trainY.reshape(len(trainY),1)
testX = testX.reshape(len(testX),1,1)
testY = testY.reshape(len(testY),1)


#LSTM

model = Sequential()
model.add(LSTM(5, batch_input_shape=(1,1,1), stateful=True,return_sequences=False))
model.add(Dense(1))
model.compile(optimizer='Adam', loss='mse')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=0, shuffle=False)


# Predictions
train_predict = model.predict(trainX, batch_size=1) 
model.reset_states()
test_predict = model.predict(testX, batch_size=1)
model.reset_states()

# Inverse predictions, test set and source data

src_inverse = fit_data.inverse_transform(data_processed)
p_train_inverse = fit_data.inverse_transform(train_predict)
p_test_inverse = fit_data.inverse_transform(test_predict)
trainY_inverse = fit_data.inverse_transform(trainY)
testY_inverse = fit_data.inverse_transform(testY)

# Evaluate efficacy with RMSE

train_RMSE = sqrt(model.evaluate(p_train_inverse.reshape(len(p_train_inverse),1,1),trainY_inverse, verbose=0, batch_size=1))
model.reset_states()
test_RMSE = sqrt(model.evaluate(p_test_inverse.reshape(len(p_test_inverse),1,1),testY_inverse, verbose=0, batch_size=1))
print("train score: %.3f, test score: %.3f" % (train_RMSE, test_RMSE))


# Shifted train and test sets

shift_test_plot = zeros(len(src_inverse))

shift_test_plot[:] = nan

shift_test_plot[len(p_train_inverse):] = p_test_inverse.reshape(len(p_test_inverse))


# Graph and compare predictions to original values

plt.plot(src_inverse[:,0])
plt.plot(p_train_inverse)
plt.plot(shift_test_plot)
plt.show()

'''
The initial RMSE score is pretty bad. Worse thant the MLP model. train: 239.469 and test: 491.043
I picked the right transform as that the other ones performed poorer. 
Now going to grid search for best hyperparameters.
'''

# Grid Searching neural net for best results

optimizer = 'Adam'
loss = 'mse'
metrics=['mse', 'mae', 'mape', 'cosine']

seed = 1
sed(seed)

def LSTM_model():
	model = Sequential()
	model.add(LSTM(5, batch_input_shape=(1,1,1), stateful=False, return_sequences=False))
	model.add(Dense(1))
	model.compile(optimizer=optimizer, loss=loss, metrics= metrics)
	return model

# Reset data

train_size = int(0.67 * len(data_processed))
train, test = data_processed[0:train_size], data_processed[train_size:]
trainX, trainY = train[:,0], train[:,1]
testX, testY = test[:,0], test[:,1]

# Reshape data to fit LSTM input req
trainX = trainX.reshape(len(trainX),1,1)
trainY = trainY.reshape(len(trainY),1)
testX = testX.reshape(len(testX),1,1)
testY = testY.reshape(len(testY),1)


epochs = [10, 50, 100, 200, 400]


kfold = KFold(n_splits=3, random_state=seed)

model = KerasRegressor(build_fn=LSTM_model, verbose=0)

param_grid = dict(epochs=epochs)
scoring = 'neg_mean_squared_error'

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=kfold,scoring=scoring )

grid_result = grid.fit(trainX, trainY,batch_size=1)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

'''
best epoch setting was 10. The score didn't change much from the base score.
Next will be finding the optimal optimizer
'''

optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
loss = 'mse'
metrics=['mse', 'mae', 'mape', 'cosine']

seed = 1
sed(seed)

def LSTM_model(optimizer='adam'):
	model = Sequential()
	model.add(LSTM(5, batch_input_shape=(1,1,1), stateful=False, return_sequences=False))
	model.add(Dense(1))
	model.compile(optimizer=optimizer, loss=loss, metrics= metrics)
	return model

# Reset data

train_size = int(0.67 * len(data_processed))
train, test = data_processed[0:train_size], data_processed[train_size:]
trainX, trainY = train[:,0], train[:,1]
testX, testY = test[:,0], test[:,1]

# Reshape data to fit LSTM input req
trainX = trainX.reshape(len(trainX),1,1)
trainY = trainY.reshape(len(trainY),1)
testX = testX.reshape(len(testX),1,1)
testY = testY.reshape(len(testY),1)

kfold = KFold(n_splits=3, random_state=seed)

model = KerasRegressor(build_fn=LSTM_model, verbose=0)

param_grid = dict(optimizer=optimizer)
scoring = 'neg_mean_squared_error'

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=kfold,scoring=scoring )

grid_result = grid.fit(trainX, trainY,batch_size=1)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


'''
Adagrad was the best optimizer. It performs well with sparse data, so that makes sense. 
Again, the score didn't improve much. Going to grid search the Adagrad parameters now.
'''

loss = 'mse'
metrics=['mse', 'mae', 'mape', 'cosine']

seed = 1
sed(seed)

def LSTM_model(lr=0.01):
	model = Sequential()
	model.add(LSTM(5, batch_input_shape=(1,1,1), stateful=False, return_sequences=False))
	model.add(Dense(1))
	model.compile(optimizer=Adagrad(lr=learning_rate), loss=loss, metrics= metrics)
	return model

# Reset data

train_size = int(0.67 * len(data_processed))
train, test = data_processed[0:train_size], data_processed[train_size:]
trainX, trainY = train[:,0], train[:,1]
testX, testY = test[:,0], test[:,1]

# Reshape data to fit LSTM input req
trainX = trainX.reshape(len(trainX),1,1)
trainY = trainY.reshape(len(trainY),1)
testX = testX.reshape(len(testX),1,1)
testY = testY.reshape(len(testY),1)

kfold = KFold(n_splits=3, random_state=seed)

model = KerasRegressor(build_fn=LSTM_model, verbose=0)

learning_rate = [0.0001, 0.001, 0.01, 0.1, 1.0]

param_grid = dict(lr = learning_rate)
scoring = 'neg_mean_squared_error'

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=kfold,scoring=scoring )

grid_result = grid.fit(trainX, trainY,batch_size=1)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

'''
The best learning rate parameter was 0.1. Again it didn't improve the score by much. 
Now Going to grid search the number of neurons. From here, I'll also stack the network for the best results.
'''

optimizer = 'Adagrad'
loss = 'mse'
metrics=['mse', 'mae', 'mape', 'cosine']

seed = 1
sed(seed)

def LSTM_model(neurons=5):
	model = Sequential()
	model.add(LSTM(neurons, batch_input_shape=(1,1,1), stateful=False, return_sequences=False))
	model.add(Dense(1))
	model.compile(optimizer=optimizer, loss=loss, metrics= metrics)
	return model

# Reset data

train_size = int(0.67 * len(data_processed))
train, test = data_processed[0:train_size], data_processed[train_size:]
trainX, trainY = train[:,0], train[:,1]
testX, testY = test[:,0], test[:,1]

# Reshape data to fit LSTM input req
trainX = trainX.reshape(len(trainX),1,1)
trainY = trainY.reshape(len(trainY),1)
testX = testX.reshape(len(testX),1,1)
testY = testY.reshape(len(testY),1)


neurons = [1, 10, 20, 50, 100]


kfold = KFold(n_splits=3, random_state=seed)

model = KerasRegressor(build_fn=LSTM_model, verbose=0)

param_grid = dict(neurons=neurons)
scoring = 'neg_mean_squared_error'

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=kfold,scoring=scoring )

grid_result = grid.fit(trainX, trainY,batch_size=1)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

'''
slight improvement in the score was achieved with 10 neurons. Adding layers now.
'''


optimizer = 'Adagrad'
loss = 'mse'
metrics=['mse', 'mae', 'mape', 'cosine']

seed = 1
sed(seed)

def LSTM_model(neurons=5):
	model = Sequential()
	model.add(LSTM(10, batch_input_shape=(1,1,1), stateful=False, return_sequences=True))
	model.add(LSTM(10, stateful=False, return_sequences=True))
	model.add(LSTM(neurons, stateful=False, return_sequences=False))
	model.add(Dense(1))
	model.compile(optimizer=optimizer, loss=loss, metrics= metrics)
	return model

# Reset data

train_size = int(0.67 * len(data_processed))
train, test = data_processed[0:train_size], data_processed[train_size:]
trainX, trainY = train[:,0], train[:,1]
testX, testY = test[:,0], test[:,1]

# Reshape data to fit LSTM input req
trainX = trainX.reshape(len(trainX),1,1)
trainY = trainY.reshape(len(trainY),1)
testX = testX.reshape(len(testX),1,1)
testY = testY.reshape(len(testY),1)


neurons = [1, 10, 20, 50, 100]


kfold = KFold(n_splits=3, random_state=seed)

model = KerasRegressor(build_fn=LSTM_model, verbose=0)

param_grid = dict(neurons=neurons)
scoring = 'neg_mean_squared_error'

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=kfold,scoring=scoring )

grid_result = grid.fit(trainX, trainY,batch_size=1)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

'''
Unfornuately, even with the grid searching of hyper parameters that I did, the score didn't improve much from the basic model.
This may be because there wasn't enough data and LSTM models do need a lot data.
Below is the final model and graphs.
'''
model = Sequential()
model.add(LSTM(10, batch_input_shape=(1,1,1), stateful=True,return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer=Adagrad(lr=0.1), loss='mse')
model.fit(trainX, trainY, epochs=10, batch_size=1, verbose=0, shuffle=False)


# Predictions

train_predict = model.predict(trainX, batch_size=1) 
model.reset_states()
test_predict = model.predict(testX, batch_size=1)
model.reset_states()

# Inverse predictions, test set and source data

scaler = MinMaxScaler(feature_range=(0,1))
data_fit  = scaler.fit(data_v.reshape(-1,1))
src_inverse = data_fit.inverse_transform(data_processed)
p_train_inverse = data_fit.inverse_transform(train_predict)
p_test_inverse = data_fit.inverse_transform(test_predict)
trainY_inverse = data_fit.inverse_transform(trainY)
testY_inverse = data_fit.inverse_transform(testY)

# Evaluate efficacy with RMSE

train_RMSE = sqrt(model.evaluate(p_train_inverse.reshape(len(p_train_inverse),1,1),trainY_inverse, verbose=0, batch_size=1))
model.reset_states()
test_RMSE = sqrt(model.evaluate(p_test_inverse.reshape(len(p_test_inverse),1,1),testY_inverse, verbose=0, batch_size=1))
print("train score: %.3f, test score: %.3f" % (train_RMSE, test_RMSE))


# Shifted train and test sets

shift_test_plot = zeros(len(src_inverse))

shift_test_plot[:] = nan

shift_test_plot[len(p_train_inverse):] = p_test_inverse.reshape(len(p_test_inverse))


# Graph and compare predictions to original values

plt.plot(src_inverse[:,0])
plt.plot(p_train_inverse)
plt.plot(shift_test_plot)
plt.show()


