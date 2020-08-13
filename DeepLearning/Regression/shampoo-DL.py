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
from keras.optimizers import Adadelta
from keras.constraints import maxnorm

# Load Data and convert to dataset

dataframe = read_csv('shampoo.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
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

# Shift dataset to plot and graph predictions to test

# Predictions

train_predict = model.predict(trainX) 
test_predict = model.predict(testX)

# Shifted train and test sets

shift_train_plot = zeros(len(train_predict))
shift_test_plot = zeros(len(data_v))

shift_train_plot[:] = nan
shift_test_plot[:] = nan

shift_train_plot[len(train_predict)] = train_predict.reshape(len(train_predict))
shift_test_plot[len(train_predict)-1:-1] = test_predict.reshape(len(test_predict))


# Graph and compare predictions to original values

plt.plot(dataframe.values)
plt.plot(shift_train_plot)
plt.plot(shift_test_plot)
plt.show()

'''
The basic neural net performed well (train: 88.84, test: 139.55). I'm going to try to improve it.
First by deseasonalizing the data and apply robust normalization.
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


# Power transform fuction
def processing(data, transform = 0):
	dim = shape(data)
	# returns data as is
	if transform == 0:
		return data
	# returns data normalized
	elif transform == 1:
		data = data.reshape(-1,1)
		scaler = MinMaxScaler(feature_range=(0,1))
		transformed = scaler.fit_transform(data)
		transformed = transformed.reshape(dim)
		return transformed
	# returns data standardized	
	elif transform == 2:
		data = data.reshape(-1,1)
		scaler = StandardScaler().fit(data)
		transformed = scaler.transform(data)
		transformed = transformed.reshape(dim)
		return transformed
	# returns data normalized with less influence by outliers
	elif transform == 3:
		data = data.reshape(-1,1)
		scaler = RobustScaler().fit(data)
		transformed = scaler.transform(data)
		transformed = transformed.reshape(dim)
		return transformed
	# returns failure message and destroys data
	else:
		print('nothing worked')
		data = "try preprocessing again"


trainX, trainY = train[:,0], train[:,1]
testX, testY = test[:,0], test[:,1]

norm_trainX = processing(trainX,3)
norm_testX = processing(testX,3)
norm_trainY = processing(trainY,3)
norm_testY = processing(testY,3)

deseason_trainX = diff(norm_trainX)
deseason_testX = diff(norm_testX)
deseason_trainY = diff(norm_trainY)
deseason_testY = diff(norm_testY)



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

# Predictions
train_predict = model.predict(trainX) 
test_predict = model.predict(testX)

# Shifted train and test sets
train_graph = train_predict
test_graph = test_predict
shift_test_plot = zeros(len(data_v))

shift_test_plot[:] = nan

shift_test_plot[len(train_graph):] = test_graph

# Graph and compare predictions to original values
plt.plot(dataframe.values)
plt.plot(train_graph)
plt.plot(shift_test_plot)
plt.show()


'''
The deseasoned and normalized neural net performed well (train: 99.476, test: 182.603).
When observing the graph, this instance of the neural net did not perform as well. Strange since in the ARIMA modeling, transforming the data improved the score. 
First, figuring out optimal batch size and epochs.
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
epochs = [10, 50, 100, 200, 400]


kfold = KFold(n_splits=3, random_state=seed)

model = KerasRegressor(build_fn=MLP_model, verbose=0)

param_grid = dict(batch_size=batch_size, epochs=epochs)
scoring = 'neg_mean_squared_error'

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=kfold,scoring=scoring )

grid_result = grid.fit(trainX, trainY)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

'''
The optimal batch_size is 7 and an epoch of 100. 
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
Although I got varying results. Adadelta was the most redundant result.
Upon applying to the model the score was comprable to the original optimizer.
next, optimizing Adadelta's learning rate and rho.
'''

loss = 'mse'
metrics=['mse', 'mae', 'mape', 'cosine']

seed = 2
sed(seed)

def MLP_model(learning_rate=0.1, rho=0.1):
	model = Sequential()
	model.add(Dense(100, activation='relu', input_dim=1))
	model.add(Dense(1))
	model.compile(optimizer='Adadelta', loss=loss, metrics=metrics)
	return model

train_size = int(0.67 * len(data_v))
train, test = data_v[0:train_size], data_v[train_size:]
trainX, trainY = train[:,0], train[:,1]
testX, testY = test[:,0], test[:,1]


kfold = KFold(n_splits=3, random_state=seed)

model = KerasRegressor(build_fn=MLP_model, batch_size=7, epochs=100, verbose=0)

learn_rate = [0.001, 0.01, 0.1, 1.0, 10]
rho = [0.001, 0.01, 0.1, 1.0, 10]
param_grid = dict(learning_rate=learn_rate, rho=rho)
scoring = 'neg_mean_squared_error'

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=kfold,scoring=scoring )

grid_result = grid.fit(trainX, trainY)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

'''
Optimal Adadelta settings, learning_rate: 1.0 and rho: 0.1
the specific optimizer settings applied to the model showed slight improvement.
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
initialization should be 'lecun_uniform'.
Although it did improve the training score, it worsened the test score.
Leaving out of the final model. 
Next is neuron activation for hidden layers. I'm going to guess relu as it's not prone vanishing gradients. 
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
softplus is the best performing activation function.
When applied to the final model it improved score all around.
Next is setting up the drop out weight constraints. 
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
Slight improvement to the final model. I will add it.
Next is tuning the number of neurons. I will do this for each additional layer I add. 
Until returns are diminished.
'''

loss = 'mse'
metrics= ['mse', 'mae', 'mape', 'cosine']
optimizer = 'adam'

seed = 2
sed(seed)

def MLP_model(neurons=1):
	model = Sequential()
	model.add(Dense(100, activation='softplus', input_dim=1, kernel_constraint=maxnorm(3)))
	model.add(Dropout(0.2))
	model.add(Dense(neurons, activation='softplus', input_dim=1, kernel_constraint=maxnorm(3)))
	model.add(Dropout(0.2))
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
First hidden layer should have 100 neurons. 
I will build new layers and test the neruons in each layer until there's a diminished return.
Below is the final depthed of the neural net
'''

loss = 'mse'
metrics=['mse']

seed = 2
sed(seed)

def MLP_model(neurons=1):
	model = Sequential()
	model.add(Dense(neurons, kernel_initializer='he_normal', activation='relu', input_dim=1, kernel_constraint=maxnorm(3)))
	model.add(Dropout(0.2))
	model.add(Dense(neurons, kernel_initializer='he_normal', activation='relu', input_dim=1, kernel_constraint=maxnorm(3)))
	model.add(Dropout(0.2))
	model.add(Dense(1, kernel_initializer='he_normal'))
	model.compile(optimizer=Adadelta(lr=1.0, rho=0.1), loss=loss, metrics=metrics)
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

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=kfold,scoring=scoring )

grid_result = grid.fit(trainX, trainY)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

'''
Unfortunately adding new layers did not improve the model.
final model below:
'''

# Final model
 
model = Sequential()
model.add(Dense(100, activation='softplus', input_dim=1, kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(1))
optimizer = Adadelta(lr=1.0, rho=0.1)
model.compile(optimizer=optimizer, loss='mse')
model.fit(trainX,trainY, epochs=100, batch_size=7,verbose=0)


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
I wonder if I played around with the normalizing and deseasonalizing of the data if I could get better performance. 
Normalizing and deseasonalizing should have produced better results, but on this attempt it did not. 
'''

