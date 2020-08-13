# allocated appropriate amount of GPU for ML/DL
import tensorflow as tf
from keras import backend as k
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.85
k.tensorflow_backend.set_session(tf.Session(config=config))
# supporting libraries
import matplotlib.pyplot as plt
from pandas import DataFrame, read_csv, concat
from math import floor
from sklearn.preprocessing import MinMaxScaler, StandardScaler,RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV
from numpy import mean, std, array, sqrt, zeros, nan
from sklearn.model_selection import KFold
import pickle
# models
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import Lars
from sklearn.linear_model import LassoLars
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import TheilSenRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Function to preprocess the data in different ways
def processing(data, transform = 0):
	# returns data as is
	if transform == 0:
		return data
	# returns data normalized
	elif transform == 1:
		scaler = MinMaxScaler(feature_range=(0,1))
		transformed = scaler.fit_transform(data)
		return transformed
	# returns data standardized	
	elif transform == 2:
		scaler = StandardScaler().fit(data)
		transformed = scaler.transform(data)
		return transformed
	# returns data normalized with less influence by outliers
	elif transform == 3:
		scaler = RobustScaler().fit(data)
		transformed = scaler.transform(data)
		return transformed
	# returns failure message and destroys data
	else:
		print('nothing worked')
		data = "try preprocessing again"

# Function for all Regressive models
def regress_models(models={}):
	# linear models
	models['lr'] = LinearRegression()
	alpha = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
	for a in alpha:
		models['lasso-'+str(a)] = Lasso(alpha=a)
	for a in alpha:
		models['ridge-'+str(a)] = Ridge(alpha=a)
	for a1 in alpha:
		for a2 in alpha:
			name = 'en-' + str(a1) + '-' + str(a2)
			models[name] = ElasticNet(a1, a2)
	models['huber'] = HuberRegressor()
	models['lars'] = Lars()
	models['llars'] = LassoLars()
	models['pa'] = PassiveAggressiveRegressor(max_iter=1000, tol=1e-3)
	models['ranscac'] = RANSACRegressor()
	models['sgd'] = SGDRegressor(max_iter=1000, tol=1e-3)
	models['theil'] = TheilSenRegressor()	
	# non-linear models
	n_neighbors = range(1, 5)
	for k in n_neighbors:
		models['knn-'+str(k)] = KNeighborsRegressor(n_neighbors=k)
	models['cart'] = DecisionTreeRegressor()
	models['extra'] = ExtraTreeRegressor()
	models['svml'] = SVR(kernel='linear', gamma='auto')
	models['svmp'] = SVR(kernel='poly', gamma='scale')
	c_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
	for c in c_values:
		models['svmr'+str(c)] = SVR(C=c, gamma='auto')
	# ensemble models
	n_trees = 100
	models['ada'] = AdaBoostRegressor(n_estimators=n_trees)
	models['bag'] = BaggingRegressor(n_estimators=n_trees)
	models['rf'] = RandomForestRegressor(n_estimators=n_trees)
	models['et'] = ExtraTreesRegressor(n_estimators=n_trees)
	models['gbm'] = GradientBoostingRegressor(n_estimators=n_trees)
	return models


# Load Data

data = read_csv('dfbirths.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)


# view histogram

data.hist()

# view Density plot

data.plot(kind='density', subplots=True, sharex=False)
data.plot(kind='box', subplots=True, sharex=False,sharey=False)

plt.show()

# raw data is near Gaussian, which is great! Needs to be standardized and has a little tail.


# create lagged dataset

values = DataFrame(data.values.astype('float32'))
dataframe = concat([values.shift(1), values], axis=1)
dataframe.columns = ['t-1', 't+1']
dataframe = dataframe.drop(dataframe.index[0])
array = dataframe.values



# Split values into training and test sets
train_size = floor(len(array) * 0.66)
train = array[0:train_size]
test = array[train_size:]

# Split sets into X,Y and Reshape
train_X, train_Y = train[:,0], train[:,1]
test_X, test_Y = test[:,0], test[:,1]

train_X = train_X.reshape(-1,1)
train_Y = train_Y.reshape(-1,1)
test_X = test_X.reshape(-1,1)
test_Y = test_Y.reshape(-1,1)

# Test each data preprocessing technique against each model and keep best model

results = []
names = []
process_version = []

for i in range(4):
	# normalized data
	p_X = processing(train_X, i)
	p_Y = processing(train_Y, i)
	# setup KFold for split testing
	seed = 3
	kfold = KFold(n_splits=3, random_state=seed)
	# for each data preprocessing test each model and keep best score
	scoring = 'neg_mean_squared_error'
	score = [0]
	model_name = 'nothing yet'
	models = regress_models()
	for name, model in models.items():
		try:
			score = cross_val_score(model, p_X, p_Y, scoring=scoring, cv=kfold, n_jobs=-1)
		except DataConversionWarning:
			score = cross_val_score(model, p_X, p_Y.ravel(), scoring=scoring, cv=kfold, n_jobs=-1)
		results.append(score.mean())
		names.append(name)
		process_version.append(i)
	print('processing type: ' + str(i))

# View results and best result

all_results = list(zip(names, results, process_version))

for i in all_results:
	if i[1] == max(results):
		print(i)

# The best result came with MinMaxScaler(). SVR(C=0.8, gamma='auto')
# Best Score: -0.02857

# Tune parameters of the SVR() model
p_X = processing(test_X, 1)
p_Y = processing(test_Y, 1)

grid_params = {
'gamma': ['scale', 'auto'],
'tol': [1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001],
'C': [1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.01, 0.001],
'epsilon': [1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.01, 0.001]
}

model = SVR()
grid = GridSearchCV(estimator=model, param_grid=grid_params, cv=kfold)

grid.fit(p_X, p_Y)
print(grid)
grid.best_params_
grid.best_score_

print("error: %s and RMSE: %s" % (score.mean(), sqrt(abs(score.mean()))))

'''
First Tuning Results:
{'C': 0.2, 'epsilon': 0.01, 'gamma': 'scale', 'tol': 0.1}
Best Score: -0.03658139020669816
error: -0.5339045413082792 and RMSE: 0.7306877180494272
'''

# Apply settings to whole dataset to see which tuned parameters work best

X = processing(array[:,0].reshape(-1,1), 1)
Y = processing(array[:,1].reshape(-1,1), 1)


model = SVR(C=0.2, epsilon=0.01, tol=0.1)
score = cross_val_score(model, X, Y.ravel(), scoring=scoring, cv=kfold, n_jobs=-1)

print("error: %s and RMSE: %s" % (score.mean(), sqrt(abs(score.mean()))))

model = SVR(C=0.2, epsilon=0.01)
score = cross_val_score(model, X, Y.ravel(), scoring=scoring, cv=kfold, n_jobs=-1)

print("error: %s and RMSE: %s" % (score.mean(), sqrt(abs(score.mean()))))

model = SVR()
score = cross_val_score(model, X, Y.ravel(), scoring=scoring, cv=kfold, n_jobs=-1)

print("error: %s and RMSE: %s" % (score.mean(), sqrt(abs(score.mean()))))

'''
For some reason tol degrades the score so I'm leaving it out. 
best score: error: -0.022548929927679883 and RMSE: 0.15016301118344652
improvement of no tuning: error: -0.02281134813757439 and RMSE: 0.15103426146929178
'''

# Second round of tuning

grid_params = {
'C': [0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29],
'epsilon': [
0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,
0.01, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.02,
0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09
]
}

model = SVR()
grid = GridSearchCV(estimator=model, param_grid=grid_params, cv=kfold)

grid.fit(p_X, p_Y)
print(grid)
grid.best_params_
grid.best_score_

'''
Best parameters: 'C': 0.23, 'epsilon': 0.05
Best score: -0.05310076712877218
When applied to whole dataset, the refinement did not improve the score.
error: -0.023076876367218713 and RMSE: 0.1519107513220138
Sticking with first round of tuning.
'''

# Make predictions and graph against naive data

# Rescale dataset
scaler = MinMaxScaler(feature_range=(0,1))
fit = scaler.fit(values)

predict_train = fit.transform(train_X)
predict_test = fit.transform(test_X)
X = fit.transform(array[:,0].reshape(-1,1))
Y = fit.transform(array[:,1].reshape(-1,1))

# fit the whole dataset with tuned parameters

model = SVR(C=0.2, epsilon=0.01)
model.fit(X,Y.ravel())

# make predictions

train_new = model.predict(predict_train)
test_new = model.predict(predict_test)

# inverse predictions

inv_train = fit.inverse_transform(train_new.reshape(-1,1))
inv_test = fit.inverse_transform(test_new.reshape(-1,1))

# Graph the results with first shifting the test predictions

test_plot = zeros(len(values))
test_plot[:] = nan 

test_plot[len(inv_train)+1:] = inv_test.reshape(len(inv_test))


plt.plot(values)
plt.plot(inv_train)
plt.plot(test_plot)

plt.show()

# Save model
filename = 'births_SVR_model.sav'
pickle.dump(model, open(filename, 'wb'))





