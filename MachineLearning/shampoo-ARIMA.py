# allocated appropriate amount of GPU for ML/DL
import tensorflow as tf
from keras import backend as k
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.85
k.tensorflow_backend.set_session(tf.Session(config=config))
# supporting libraries
import matplotlib.pyplot as plt
from pandas import DataFrame, read_csv, concat, Grouper, Series
from statsmodels.tsa.stattools import adfuller as adf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from math import floor
from sklearn.preprocessing import MinMaxScaler, StandardScaler,RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV
from numpy import mean, std, array, sqrt, shape, asscalar, save, load
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error as mse
import pickle
from statsmodels.tsa.arima_model import ARIMA, ARIMAResults


# Load Data and split/save into training dataset and validation dataset
series = read_csv('shampoo.csv', header=0, parse_dates=True, index_col=0, squeeze=True)
# Find index number of where to split for validation then split
split_pt = len(series)-12
dataset, validation = series[0:split_pt], series[split_pt:]
# save into individual files
dataset.to_csv('shampoo-ARMIA-dataset.csv', header=False)
validation.to_csv('shampoo-ARIMA-validate.csv', header=False)


# Prepare dataset.csv for training and modeling
series = read_csv('shampoo-ARMIA-dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
X = series.values
X = X.astype('float32')
train_size = int(len(X) * 0.66)
train, test = X[0:train_size], X[train_size:]

# Create Naive (Persistence) Forecast
history = [x for x in train]
predictions = list()

for i in range(len(test)):
	yhat = history[-1]
	predictions.append(yhat)
	obs = int(test[i])
	history.append(obs)
	print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))

m_s_e = mse(test, predictions)
rmse = sqrt(m_s_e)
print('RMSE of Naive Forecast: %.3f' % (rmse))

# Score for naive forecast to judge against: 93.008

# ARIMA modeling 

# Plot data on graphs to analyze shape

# Line plot and Seasonal Line plot

series.plot()
groups = series['2001':'2002'].groupby(Grouper(freq='A'))
years = Dataframe()
plt.figure()
i = 1
n_groups = len(groups)
for name, group in groups:
	plt.subplot((n_groups*100) + 10 + i)
	i += 1
	plt.plot(group)

plt.show()

# Seasonal data confirmed. May perform better with power transform

# Density Plot

plt.figure(1)
plt.subplot(211)
series.hist()
plt.subplot(212)
series.plot(kind='kde')
plt.show()

# Not a Gaussian distribution. May perform better with power transform.


# 'Deseasonalize' data by differencing

def diff(dataset, interval=12):
	diff =list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)

# Revert dataset back from 'deseasonlization'
def inverse_diff(history, yhat, interval=12):
	return yhat + history[-interval]


# Determine initial p,d,q values for ARIMA

station = diff(X)

# Check if stationary

result = adf(station)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for percent, value in result[4].items():
	print('\t%s: %.3f' % (percent, value))

# P value is smaller than 1%. dataset is stationary and null hypothesis can be rejected.
# d will start with a value of 0
# Determine p,q values by plotting ACF and PACF
# distribution is not Gaussian, so ACF may be useless

plt.figure()
plt.subplot(211)
plot_acf(station, ax=plt.gca())
plt.subplot(212)
plot_pacf(station, ax=plt.gca())
plt.show()

# ACF shows no correlation. Going to try p = 1, q = 0. based on outliers

# ARIMA with (1,0,0) values

history = [x for x in train]
predictions = list()

for i in range(len(test)):
	station = diff(history)
	model = ARIMA(station, order=(1,0,0))
	print('okay')
	model_fit = model.fit(trend='nc', disp=0)
	yhat = model_fit.forecast()[0]
	yhat = inverse_diff(history, yhat)
	predictions.append(yhat)
	obs = test[i]
	history.append(obs)
	print('Predicted: %.3f, Expected: %.3f' % (yhat, obs))

m_s_e = mse(test, predictions)
rmse = sqrt(m_s_e)
print('RMSE: %.3f' % rmse)

# ARIMA model with order = 1,0,0 performed worse than naive forecast. 
# RMSE: 156.281
# Will try to power transform dataset

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

history = [x for x in train]
predictions = list()

for i in range(len(test)):
	history = processing(array(history), 3)
	station = diff(history)
	model = ARIMA(station, order=(1,0,0))
	model_fit = model.fit(trend='nc', disp=0)
	yhat = model_fit.forecast()[0]
	yhat = inverse_diff(history, yhat)
	predictions.append(yhat)
	obs = test[i]
	history = history.tolist()
	history.append(obs)
	print('Predicted: %.3f, Expected: %.3f' % (yhat, obs))

scaler_1 = MinMaxScaler(feature_range=(0,1))
scaler_2 = StandardScaler()
scaler_3 = RobustScaler()

s1_fit = scaler_1.fit(array(history).reshape(-1,1))
s2_fit = scaler_2.fit(array(history).reshape(-1,1))
s3_fit = scaler_3.fit(array(history).reshape(-1,1))

s1_inv = s1_fit.inverse_transform(array(predictions).reshape(-1,1))
s2_inv = s2_fit.inverse_transform(array(predictions).reshape(-1,1))
s3_inv = s3_fit.inverse_transform(array(predictions).reshape(-1,1))

m_s_e = mse(test, s1_inv)
m_s_e = mse(test, s2_inv)
m_s_e = mse(test, s3_inv)

rmse = sqrt(m_s_e)
print('RMSE: %.3f' % rmse)

'''
RMSE for MinMaxScaler(): 287.731
RMSE for StandardScaler(): 274.838
RMSE for RobustScaler(): 295.496

Power Transform StandardScaler() performed worse than no transform.
Continuing to configure ARIMA with no transform.
'''

# Grid Searching ARIMA model

# function to test a configuration of hyperparameters on ARIMA model

values = [0,1,2,3,4,5,6,7,8,9,10,11,12]
p_val = values
d_val = values
q_val = values

# use total dataset since there aren't enough values

series = read_csv('shampoo.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
X = series.values
X = X.astype('float32')
train_size = int(len(X) * 0.66)
train, test = X[0:train_size], X[train_size:]


def test_model(hyper_order):
	history = [x for x in train]
	predictions = list()
	for i in range(len(test)):
		p_hist = processing(array(history), 0)
		station = diff(p_hist)
		model = ARIMA(station, order=hyper_order)
		model_fit = model.fit(trend='nc', disp=0)
		yhat = model_fit.forecast()[0]
		yhat = inverse_diff(p_hist, yhat)
		predictions.append(asscalar(yhat))
		obs = test[i]
		history = p_hist.tolist()
		history.append(obs)
	m_s_e = mse(test, predictions)
	rmse = sqrt(m_s_e)
	print(test, predictions)
	return rmse



def config_model(p_val=p_val, d_val=d_val, q_val=q_val):
	best_score, best_config = float("inf"), None
	for p in p_val:
		for d in d_val:
			for q in q_val:
				order = (p,d,q)
				try:
					config = test_model(order)
					print(config)
					if config < best_score:
						best_score, best_config = config, order
					print('ARIMA order: %s RMSE: %.3f' % (order, config))
				except:
					continue
	print('Best configuration of hyperparameters is: %s with a RMSE score of: %.3f' % (best_config, best_score))

config_model()


"""
Conclusive hyperparameter search result:
Best configuration of hyperparameters is: (3, 1, 0) with a RMSE score of: 258.178
"""

# Retrieve mean bias

history = [x for x in train]
predictions = list()
for i in range(len(test)):
	p_hist = processing(array(history), 0)
	station = diff(p_hist)
	model = ARIMA(station, order=(0,1,1))
	model_fit = model.fit(trend='nc', disp=0)
	yhat = model_fit.forecast()[0]
	yhat = inverse_diff(p_hist, yhat)
	predictions.append(asscalar(yhat))
	obs = test[i]
	history = p_hist.tolist()
	history.append(obs)

m_s_e = mse(test, predictions)
rmse = sqrt(m_s_e)

# Errors

# errors
residuals = [test[i]-predictions[i] for i in range(len(test))]
residuals = DataFrame(residuals)
print(residuals.describe())

# Mean bias: 52.47


# Save model

series = read_csv('shampoo-ARMIA-dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
X = series.values
X = X.astype('float32')
history = [x for x in X]
p_hist = processing(array(history), 3)
station = diff(p_hist)
model = ARIMA(station, order=(0,1,1))
model_fit = model.fit(trend='nc', disp=0)
bias = 125.35
model_fit.save('shampoo-ARIMA-model-fin.pkl')
save('sAfb.npy', [bias])


# Validate model 

# Load data
series = read_csv('shampoo-ARMIA-dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
X = series.values
X = X.astype('float32')

series = read_csv('shampoo-ARIMA-validate.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
y = series.values
y = y.astype('float32')

history = [x for x in X]
predictions = list()

model_fit = ARIMAResults.load('shampoo-ARIMA-model-fin.pkl')
bias = load('sAfb.npy')

# compare predictions to validation
for i in range(len(y)):
	p_hist = processing(array(history), 0)
	station = diff(p_hist)
	model = ARIMA(station, order=(3,1,0))
	model_fit = model.fit(trend='nc', disp=0)
	yhat = model_fit.forecast()[0]
	yhat = bias + inverse_diff(p_hist, yhat)
	predictions.append(asscalar(yhat))
	obs = y[i]
	history = p_hist.tolist()
	history.append(obs)
	print('Predicted= %.3f, Expected= %.3f' % (yhat, obs))

m_s_e = mse(y, predictions)
rmse = sqrt(m_s_e)
print('RMSE= %.3f' % rmse)

# graph it
pyplot.plot(y)
pyplot.plot(predictions, color='red')
pyplot.show()


'''
Due to not having enough data in the dataset.
I had to finagle around with which data set to use at which point.
This threw off the predictions. 
All and all not bad with the results. 
The predictions albeit off, have moments were they weren't off by too much.
'''
