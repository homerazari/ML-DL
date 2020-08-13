# allocated appropriate amount of GPU for ML/DL

import tensorflow as tf
from keras import backend as k
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.75
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
from keras.optimizers import Adam
from keras.constraints import maxnorm

# Load Data and convert to dataset

dataframe = read_csv('dfbirths.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
dataframe_shifted = concat([dataframe.shift(1), dataframe], axis=1)
dataframe_shifted.columns = ['t-1', 't+1']
df_final = dataframe_shifted.drop(dataframe.index[0])
data_v = df_final.values


# Split into train, test sets

train_size = int(0.67 * len(data_v))
train, test = data_v[0:train_size], data_v[train_size:]
trainX, trainY = train[:,0], train[:,1]
testX, testY = test[:,0], test[:,1]

# MLP model

model = Sequential()
model.add(Dense(100, activation='relu', input_dim=1))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(trainX,trainY, epochs=100, verbose=0)

# Evaluate efficacy with RMSE

train_RMSE = sqrt(model.evaluate(trainX,trainY, verbose=0))
test_RMSE = sqrt(model.evaluate(testX,testY, verbose=0))
print("train score: %.3f, test score: %.3f" % (train_RMSE, test_RMSE))

# train score: 8.965, test score: 9.039 
# Scored without power transform, standardizing or grid searching

# Shift dataset to plot and graph predictions to test

# Predictions

train_predict = model.predict(trainX) 
test_predict = model.predict(testX)

# Shifted train and test sets

shift_test_plot = zeros(len(data_v))

shift_test_plot[:] = nan

shift_test_plot[len(train_predict)-1:-1] = test_predict.reshape(len(test_predict))


# Graph and compare predictions to original values

plt.plot(dataframe.values)
plt.plot(train_predict)
plt.plot(shift_test_plot)
plt.show()


'''
The basic neural net performed well (train score: 8.965, test score: 9.039). 
I'm going to try to improve it.
First by deseasonalizing the data and apply normalization.
Then by grid search the best depth and width of the neural net
'''

# 'Deseasonalize' data by differencing

def diff(dataset, interval=1):
	diff =list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return array(diff)

# Revert dataset back from 'deseasonlization'

def inverse_diff(history, yhat, interval=1):
	return yhat + history[-interval]

# Split

trainX, trainY = train[:,0], train[:,1]
testX, testY = test[:,0], test[:,1]

# Normalize

scaler = MinMaxScaler(feature_range=(0,1))

fit_trainX = scaler.fit(trainX.reshape(-1,1))
fit_testX = scaler.fit(testX.reshape(-1,1))
fit_trainY = scaler.fit(trainY.reshape(-1,1))
fit_testY = scaler.fit(testY.reshape(-1,1))

norm_trainX = fit_trainX.transform(trainX.reshape(-1,1))
norm_testX = fit_testX.transform(testX.reshape(-1,1))
norm_trainY = fit_trainY.transform(trainY.reshape(-1,1))
norm_testY = fit_testY.transform(testY.reshape(-1,1))

# Deseason

deseason_trainX = diff(norm_trainX)
deseason_testX = diff(norm_testX)
deseason_trainY = diff(norm_trainY)
deseason_testY = diff(norm_testY)


# MLP model

model = Sequential()
model.add(Dense(100, activation='relu', input_dim=1))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(deseason_trainX, deseason_trainY, epochs=100, verbose=0)

# Evaluate efficacy with RMSE

train_RMSE = sqrt(model.evaluate(trainX, trainY, verbose=0))
test_RMSE = sqrt(model.evaluate(testX, testY, verbose=0))
print("train score: %.3f, test score: %.3f" % (train_RMSE, test_RMSE))

# Predictions

train_predict = model.predict(trainX) 
test_predict = model.predict(testX)

# Shifted train and test sets

train_graph = train_predict
test_graph = test_predict
shift_test_plot = zeros(len(data_v))

shift_test_plot[:] = nan

shift_test_plot[len(train_graph):] = test_graph.reshape(len(test_graph))

# Graph and compare predictions to original values

plt.plot(dataframe.values)
plt.plot(train_graph)
plt.plot(shift_test_plot)
plt.show()

'''
Deseasoned and normalized score: train score: 61.402, test score: 68.085
Normalized score: train score: 34.697, test score: 38.460

It appears deseasoning and normalizing the score, worsens the data output.
'''

# Grid Searching neural net for best results

optimizer = 'Adam'
loss = 'mse'
metrics=['mse', 'mae', 'mape', 'cosine']

seed = 2
sed(seed)

def MLP_model():
	model = Sequential()
	model.add(Dense(100, activation='relu', input_dim=1))
	model.add(Dense(1))
	model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
	return model

train_size = int(0.67 * len(data_v))
train, test = data_v[0:train_size], data_v[train_size:]
trainX, trainY = train[:,0], train[:,1]
testX, testY = test[:,0], test[:,1]

batch_size = [1,2,3,4,5,6,7,8,9,10,11]
epochs = [1, 10, 50, 100, 200, 400]


kfold = KFold(n_splits=3, random_state=seed)

model = KerasRegressor(build_fn=MLP_model, verbose=0)

param_grid = dict(epochs=epochs, batch_size=batch_size)
scoring = 'neg_mean_squared_error'

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=kfold,scoring=scoring )

grid_result = grid.fit(trainX, trainY)

# summarize results

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


'''
Best batch size is 1 and the best epoch number is 400.
train score: 6.727, test score: 8.208
Upon applying these settings to the model, there was a slight improvement in the score.
Next, testing optimizers
'''

optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
loss = 'mse'
metrics=['mse', 'mae', 'mape', 'cosine']

seed = 2
sed(seed)

def MLP_model(optimizer='adam'):
	model = Sequential()
	model.add(Dense(100, activation='relu', input_dim=1))
	model.add(Dense(1))
	model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
	return model

train_size = int(0.67 * len(data_v))
train, test = data_v[0:train_size], data_v[train_size:]
trainX, trainY = train[:,0], train[:,1]
testX, testY = test[:,0], test[:,1]


kfold = KFold(n_splits=3, random_state=seed)

model = KerasRegressor(build_fn=MLP_model, batch_size=7, epochs=100, verbose=0)

param_grid = dict(optimizer=optimizer)
scoring = 'neg_mean_squared_error'

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=kfold,scoring=scoring )

grid_result = grid.fit(trainX, trainY)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

'''
SGD was the best optimizer.
Upon applying to the model the score was slightly worse than using default 'Adam'.
Next optimizing Adam optimizer.
'''

loss = 'mse'
metrics=['mse', 'mae', 'mape', 'cosine']

seed = 2
sed(seed)

def MLP_model(learning_rate=0.1, rho=0.1):
	model = Sequential()
	model.add(Dense(100, activation='relu', input_dim=1))
	model.add(Dense(1))
	model.compile(optimizer='Adam', loss=loss, metrics=metrics)
	return model

train_size = int(0.67 * len(data_v))
train, test = data_v[0:train_size], data_v[train_size:]
trainX, trainY = train[:,0], train[:,1]
testX, testY = test[:,0], test[:,1]


kfold = KFold(n_splits=3, random_state=seed)

model = KerasRegressor(build_fn=MLP_model, batch_size=7, epochs=100, verbose=0)

learn_rate = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 10]
param_grid = dict(learning_rate=learn_rate)
scoring = 'neg_mean_squared_error'

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=kfold,scoring=scoring )

grid_result = grid.fit(trainX, trainY)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


'''
Best learning rate for the Adam optimizer was 0.01.
When applied to model the score showed slight improvement.
train score: 6.924, test score: 7.459
Next network weight initialization 
'''

loss = 'mse'
metrics= ['mse', 'mae', 'mape', 'cosine']
optimizer = 'adam'

seed = 2
sed(seed)

def MLP_model(init_mode='zero'):
	model = Sequential()
	model.add(Dense(100, kernel_initializer=init_mode, activation='relu', input_dim=1))
	model.add(Dense(1, kernel_initializer=init_mode))
	model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
	return model

train_size = int(0.67 * len(data_v))
train, test = data_v[0:train_size], data_v[train_size:]
trainX, trainY = train[:,0], train[:,1]
testX, testY = test[:,0], test[:,1]


kfold = KFold(n_splits=3, random_state=seed)

model = KerasRegressor(build_fn=MLP_model, batch_size=7, epochs=100, verbose=0)

init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']

param_grid = dict(init_mode=init_mode)
scoring = 'neg_mean_squared_error'

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=kfold,scoring=scoring )

grid_result = grid.fit(trainX, trainY)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

'''
initialization should be  'glorot_normal'.
Applying this to the model worsened the score. 
I will leave it out.
Next is the neuron activation for hidden layers. best guess is 'relu'. 
'''

loss = 'mse'
metrics= ['mse', 'mae', 'mape', 'cosine']
optimizer = 'adam'

seed = 2
sed(seed)

def MLP_model(activation='relu'):
	model = Sequential()
	model.add(Dense(100, activation=activation, input_dim=1))
	model.add(Dense(1))
	model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
	return model

train_size = int(0.67 * len(data_v))
train, test = data_v[0:train_size], data_v[train_size:]
trainX, trainY = train[:,0], train[:,1]
testX, testY = test[:,0], test[:,1]


kfold = KFold(n_splits=3, random_state=seed)

model = KerasRegressor(build_fn=MLP_model, batch_size=7, epochs=100, verbose=0)

activation = ['softmax', 'softplus', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']

param_grid = dict(activation=activation)
scoring = 'neg_mean_squared_error'

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=kfold,scoring=scoring )

grid_result = grid.fit(trainX, trainY)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

'''
Best activation method was 'tanh'.
Upon application to the model, it performed worse.
remaining with 'relu'.
Based on a paper, I will set dropout at 20% and test weights.
'''

loss = 'mse'
metrics= ['mse', 'mae', 'mape', 'cosine']
optimizer = 'adam'

seed = 2
sed(seed)

def MLP_model(weight_constraint=1):
	model = Sequential()
	model.add(Dense(100, activation='relu', input_dim=1, kernel_constraint=maxnorm(weight_constraint)))
	model.add(Dropout(0.2))
	model.add(Dense(1))
	model.compile(optimizer= optimizer, loss=loss, metrics=metrics)
	return model

train_size = int(0.67 * len(data_v))
train, test = data_v[0:train_size], data_v[train_size:]
trainX, trainY = train[:,0], train[:,1]
testX, testY = test[:,0], test[:,1]


kfold = KFold(n_splits=3, random_state=seed)

model = KerasRegressor(build_fn=MLP_model, batch_size=7, epochs=100, verbose=0)

weight_constraint = [1, 2, 3, 4, 5]

param_grid = dict(weight_constraint=weight_constraint)
scoring = 'neg_mean_squared_error'

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=kfold,scoring=scoring )

grid_result = grid.fit(trainX, trainY)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

'''
Weight constraint of 3 is ideal.
No improvment noticed.
Next is tuning the number of neurons. I will do this for each additional layer I add. 
Until returns are diminished.
'''

loss = 'mse'
metrics= ['mse', 'mae', 'mape', 'cosine']
optimizer = Adam(lr=0.01)

seed = 2
sed(seed)

def MLP_model(neurons=1):
	model = Sequential()
	model.add(Dense(50, activation='relu', input_dim=1))
	model.add(Dense(neurons, activation='relu', input_dim=1))
	model.add(Dense(1))
	model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
	return model

train_size = int(0.67 * len(data_v))
train, test = data_v[0:train_size], data_v[train_size:]
trainX, trainY = train[:,0], train[:,1]
testX, testY = test[:,0], test[:,1]


kfold = KFold(n_splits=3, random_state=seed)

model = KerasRegressor(build_fn=MLP_model, batch_size=7, epochs=100, verbose=0)

neurons = [1, 10, 50, 100, 200, 500]

param_grid = dict(neurons=neurons)
scoring = 'neg_mean_squared_error'

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=kfold,scoring=scoring )

grid_result = grid.fit(trainX, trainY)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

'''
Ideal neurons in the first layer was 100. 
Adding additional layers did not improve the performance. 
Final model below.
'''

# Final model

model = Sequential()
model.add(Dense(100, activation='relu', input_dim=1))
model.add(Dense(1))
model.compile(optimizer=Adam(lr=0.01), loss='mse')
model.fit(trainX,trainY, epochs=400, batch_size=1, verbose=0)

# Prep final model data for graphing  

fin_train_predict = model.predict(trainX)
fin_test_predict = model.predict(testX)

fin_shift_test_plot = zeros(len(data_v))

fin_shift_test_plot[:] = nan

fin_shift_test_plot[len(fin_train_predict):] = fin_test_predict.reshape(len(fin_test_predict))

# Naive model

model = Sequential()
model.add(Dense(100, activation='relu', input_dim=1))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(trainX,trainY, epochs=100, verbose=0)


# Prep naive model data for graphing 

nve_train_predict = model.predict(trainX)
nve_test_predict = model.predict(testX)

nve_shift_test_plot = zeros(len(data_v))

nve_shift_test_plot[:] = nan

nve_shift_test_plot[len(nve_train_predict):] = nve_test_predict.reshape(len(nve_test_predict))

# Graph naive v. final model predictions

fig, axs = plt.subplots(2)
fig.suptitle('Naive v. Final Model Predictions')

axs[0].plot(nve_train_predict)
axs[0].plot(nve_shift_test_plot)
axs[0].plot(dataframe.values)

axs[1].plot(fin_train_predict)
axs[1].plot(fin_shift_test_plot)
axs[1].plot(dataframe.values)

plt.show()

'''
The final model with hyperparameters tuned performed noticably better than the standard model. 
'''

