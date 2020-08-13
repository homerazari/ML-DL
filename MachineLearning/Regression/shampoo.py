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
from numpy import mean, std, array, sqrt, nan, zeros
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


# load data
data = read_csv('shampoo.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)

# view histogram
data.hist()
# view Density plot
data.plot(kind='density', subplots=True, sharex=False)
data.plot(kind='box', subplots=True, sharex=False,sharey=False)

plt.show()
# raw data is not Gaussian


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

# The best result came from the RANSACRegressor() model with a MinMaxScaler()

# Tune parameters of the SGDRegressor() model
p_X = processing(test_X, 1)
p_Y = processing(test_Y, 1)

grid_params = {
'max_trials': [1, 10, 50, 100, 500, 1000, 5000, 10000],
'max_skips': [0, 1, 10, 50, 100],
'loss': ['absolute_loss', 'squared_loss'],
'random_state': [0,1,2,4,8,16,32,64,128,256,None]
}
model = RANSACRegressor()
grid = GridSearchCV(estimator=model, param_grid=grid_params, cv=kfold)

grid.fit(p_X, p_Y)
print(grid)
grid.best_params_
grid.best_score_

print("error: %s and RMSE: %s" % (score.mean(), sqrt(abs(score.mean()))))

'''
First Tuning Results
{'loss': 'absolute_loss', 'max_skips': 0, 'max_trials': 10, 'random_state': 64}
Best score: -1.7085570919466646
error: error: -1.5108454451183704 and RMSE: 1.229164531345731
'''

# Apply settings to whole dataset to see which tuned parameters work best

X = processing(array[:,0].reshape(-1,1), 1)
Y = processing(array[:,1].reshape(-1,1), 1)

model = model = RANSACRegressor() 

score = cross_val_score(model, X, Y.ravel(), scoring=scoring, cv=kfold, n_jobs=-1)

print("error: %s and RMSE: %s" % (score.mean(), sqrt(abs(score.mean()))))

# error: -0.09196275720993678 and RMSE: 0.3032536186262858

model = model = RANSACRegressor(
loss = 'absolute_loss',
max_skips = 0,
max_trials = 10, 
random_state=64
) 

score = cross_val_score(model, X, Y.ravel(), scoring=scoring, cv=kfold, n_jobs=-1)

print("error: %s and RMSE: %s" % (score.mean(), sqrt(abs(score.mean()))))

# error: -0.11710174133380254 and RMSE: 0.3422013169667857

model = model = RANSACRegressor(
loss = 'absolute_loss',
max_skips = 0,
max_trials = 10
) 

score = cross_val_score(model, X, Y.ravel(), scoring=scoring, cv=kfold, n_jobs=-1)

print("error: %s and RMSE: %s" % (score.mean(), sqrt(abs(score.mean()))))

# error: -0.10204416078825791 and RMSE: 0.3194435173677154

model = model = RANSACRegressor(
loss = 'absolute_loss',
max_skips = 0
) 

score = cross_val_score(model, X, Y.ravel(), scoring=scoring, cv=kfold, n_jobs=-1)

print("error: %s and RMSE: %s" % (score.mean(), sqrt(abs(score.mean()))))

# error: -0.1386968499670426 and RMSE: 0.3724202598772556

model = model = RANSACRegressor(
loss = 'absolute_loss'
) 

score = cross_val_score(model, X, Y.ravel(), scoring=scoring, cv=kfold, n_jobs=-1)

print("error: %s and RMSE: %s" % (score.mean(), sqrt(abs(score.mean()))))

# error: -0.10988491773605347 and RMSE: 0.33148894059388084

model = model = RANSACRegressor(
loss = 'absolute_loss',
max_trials = 10
) 

score = cross_val_score(model, X, Y.ravel(), scoring=scoring, cv=kfold, n_jobs=-1)

print("error: %s and RMSE: %s" % (score.mean(), sqrt(abs(score.mean()))))

# error: -0.09936415341993173 and RMSE: 0.3152208010584513

'''
The best results on the whole dataset came from the default paramters. 
No tuning will be applied.
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

model = RANSACRegressor()
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

test_plot[len(inv_train):-1] = inv_test.reshape(len(inv_test))


plt.plot(values)
plt.plot(inv_train)
plt.plot(test_plot)

plt.show()

# save model 

model.fit(X,Y)
filename = 'shampoo_RSCR_model.sav'
pickle.dump(model, open(filename, 'wb'))

	




