import sklearn
from sklearn.svm import	NuSVR
from sklearn.svm import	SVR
import numpy as np
import csv
import os.path
import math;
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sys import stderr
from sys import exit
from sklearn.decomposition import PCA
import random
#Training for arousal.
# 2.csv to 1745.csv

maxFileIndex = 3000 + 1;

def readRows(filename):
	csv_rows = [];
	with open(filename) as csvfile:
		data = csv.reader(csvfile, delimiter=' ', quotechar='|')
		for row in data:
			# parts = row.split(',');
			csv_rows.append(row);

	return csv_rows;

def readYWithProvidedFileName(filename):
	csv_rows = readRows(filename);

	csv_rows = csv_rows[1:];
	#print(len(csv_rows));
	#print(csv_rows[0]);

	y = []
	for row in csv_rows:
		# print(row)
		# print(row[0])
		# parts = row.split(',');
		row = row[0].split(',')
		# print(row)
		y.append(float(row[1]));
		# print(row[3])

	return y;

def readY():
	#Replace with static_annotations_averaged_songs_1_2000.csv file location.
	filename = "../nemo/DEAM_dataset/annotations/annotations_averaged_per_song/song_level/static_annotations_averaged_songs_1_2000.csv"
	y = readYWithProvidedFileName(filename);
	
	#Replace with static_annotations_averaged_songs_2000_2058.csv file location.
	filename = "../nemo/DEAM_dataset/annotations/annotations_averaged_per_song/song_level/static_annotations_averaged_songs_2000_2058.csv"
	z = readYWithProvidedFileName(filename);
	
	for zz in z:
		y.append(zz);

	return y;

#####

#This was made for experimenting with average of the time intervals
def getAveragedFeatures(n_columns, feature_rows):
	average_rows = [0] * n_columns;
	for i in range(len(feature_rows)):
		for j in range(len(feature_rows[i])):
			average_rows[j] += feature_rows[i][j];

	for i in range(n_columns):
		average_rows[i] /= len(feature_rows)

	return average_rows

#This function uses top K values of features for variosu time intervals.
def getMaxFeatures(n_columns, feature_rows):
	max_rows = [-1e9] * n_columns
	for i in range(len(feature_rows)):
		for j in range(len(feature_rows[i])):
			max_rows[j] = max(max_rows[j], feature_rows[i][j]);

	for x in max_rows:
		assert(x > -1e8)

	return max_rows	

#top_k_feature_values = 100
#This was implemented in order to experiment with the idea suggested by Tanaya Guha to try out max pooling of the features.
#This strategy doesn't provide much change, but some sort of experimentation over this might be useful.
def getTopKValuesOfFeatures(n_columns, feature_rows):	
	average_rows = [0] * n_columns;

	#top_k_feature_values = int(len(feature_rows) / 20)
	top_k_feature_values = len(feature_rows)

	for j in range(n_columns):
		vals = []
		for i in range(len(feature_rows)):
			vals.append(feature_rows[i][j])
		vals.sort(reverse=True)
	
		assert(len(vals) >= top_k_feature_values)
		vals = vals[0:top_k_feature_values]
		
		average_rows[j] = sum(vals) / len(vals);

	return average_rows	

def read_data():
	valid_feature_file_indices = np.load("valid_feature_file_indices.npy")
	print(valid_feature_file_indices)
	print('total=', len(valid_feature_file_indices))

	#Replace this with the features directory
	features_dir = "../nemo/DEAM_dataset/features/"
	#Replace this with "annotations averaged per song/dynamic (per second annotations)/" directory location path.
	annotations_dir = "../nemo/DEAM_dataset/annotations/annotations_averaged_per_song/dynamic_(per_second_annotations)/"
	#for arousal, change similarly for valence.
	annotations_file = annotations_dir + "valence.csv"
	R = readRows(annotations_file)
	# print(R)
	R = R[1:]

	X = []
	y = []

	for r in R:
		t = 15.0
		file_number = r[0].split(',')
		file_number = file_number[0]
		print('file_number', file_number)
		feature_file_name = features_dir + str(file_number) + ".csv"
		x = readRows(feature_file_name)
		#print(x, type(x))
		# for i in range(len(x)):
		# 	x[i] = x[i].split(';')
		#exit()

		start = 31;
		for idx in range(len(r) - 1):
			arousal_val = float(r[idx + 1])
			if (start == len(x)):
				print("something unexpected happened.")
				break;
			# print(arousal_val, type(arousal_val))
			# print(type(x[start]), x[start])
			row_x = []
			for i in range(1, len(x[start]), 1):
				row_x.append(float(x[start][i]))
			X.append(row_x);
			y.append(arousal_val)
			t += 0.5
			start += 1
		
		# print(X[1])
		# print(y[1])
		# break

	#print(X);
	#print(y);
	np.save("features.txt", X);
	np.save("y_values.txt", y);

	# for x in R:
	# 	print(len(x), end=' ')

	# for file_number in valid_feature_file_indices:
	# 	filename = features_dir + str(file_number) + ".csv";
	# 	assert(os.path.isfile(filename))

################
# One of the very useful mehtods regarding extracting the X parameter (feature parameter). 
#This method might be useful for you to know how to read from a csv file, how to process and extract the data.
# 
################
def readX():
	features_dir = "../nemo/DEAM_dataset/features/"
	X = []
	n_columns = -1;
	for file_number in range(2, maxFileIndex):
		filename = features_dir + str(file_number) + ".csv";
			
		if (not os.path.isfile(filename)):
			continue;

		print("Processing file", filename);

		row_data = readRows(filename);
		row_data = row_data[1:];

		#row_data contains various features values over different time frames.
		#We take average of each parameter to get the final average value of the feature. 
		
		#print(row_data);

		feature_rows = [];
		for i in range(0, len(row_data)):
			#print(row_data[i][0]);
			row_str = row_data[i][0].split(';');
			row = [];
			for s in row_str:
				row.append(float(s));
			
			row = row[1:];
			#print('row = ', row);
			
			if (n_columns == -1):
				n_columns = len(row);
			elif n_columns != len(row):
				raise ValueError("Number of features are variable across csv files.");
			
			feature_rows.append(row);

		#print(average_rows);
		#average_rows = getAveragedFeatures(n_columns, feature_rows)
		# X.append(average_rows);
		#max_rows = getMaxFeatures(n_columns, feature_rows)
		#X.append(max_rows)

		# global top_k_feature_values
		# top_k_feature_values = len(feature_rows)

		top = getTopKValuesOfFeatures(n_columns, feature_rows)
		X.append(top)

		# print(top)
		# print(average_rows)
		# exit();

		# if file_number >= 10:
		# 	break;

	#print('Len of X', len(X));
	#print(X);
	return X;

np.set_printoptions(threshold=np.inf)

def save_valid_feature_file_indices():
	#Replaces this with the features directory.
	features_dir = "../nemo/DEAM_dataset/features/"
	valid_feature_file_indices = []
	for file_number in range(2, maxFileIndex):
		filename = features_dir + str(file_number) + ".csv";
			
		if os.path.isfile(filename):
			valid_feature_file_indices.append(file_number)
	np.save("valid_feature_file_indices", valid_feature_file_indices)

def readAndSave():
	data_X = readX();
	
	#Add i at the end of X.
	# for i in range(len(data_X)):
	# 	data_X[i].append(i + 1)

	print(len(data_X), data_X[0])

	np.save("features.txt", data_X);
	data_y = readY()
	#Just make y[i] = i + 1.
	# for i in range(len(data_y)):
	# 	data_y[i] = i + 1
	np.save("y_values.txt", data_y);

def readSavedData():
	X = np.load("features.txt.npy");
	y = np.load("y_values.txt.npy");
	return (X, y)



def myAssert(x):
	if not x:
		raise ValueError('x is not true')

def get_test_data():
	files = os.listdir("/home/ankit/Studies/6th_semester/UGP/feature_test")	
	#Replace this with the features directory
	features_dir = "/home/ankit/Studies/6th_semester/UGP/feature_test/"
	n_columns = -1;
	X = []

	for i in range(len(files)):
		feature_rows = [];
		avg_row = np.zeros(260)

		feature_file_name = features_dir + files[i]
		x = np.loadtxt(feature_file_name, delimiter=",")
		for j in range(len(x)):
			temp = x[j]	
			
			row = temp[1:];
			feature_rows.append(row);
			# print(len(row))
		# top = getTopKValuesOfFeatures(n_columns, feature_rows)
		# X.append(top)
		for k in range(len(x)):
			avg_row += feature_rows[k]
		avg_row	= avg_row/len(x)
		X.append(avg_row)
	return X, files	

		




# Y = readY()
# exit()

#Uncomment these lines to read the features and save into the corresponding files.
# save_valid_feature_file_indices()
# read_data();

# readAndSave()
# exit()

X_test, test_files = get_test_data()
# for i in range(10):
# 	print(X_test[i])
# for i in range(5850):
# 	print(test_files[i/30]+"-"+str(i%30)+", ")
# exit()

#saveFeatureFileNames()

(X, y) = readSavedData()
# print(len(X))
# print(len(X_test))
# exit()
#Check whether the data indeed makes sense.
# print(y[0:2])

#Total denotes the total number of data points over which you want to train.
#If you want to train over the entire data set X, then brace yourself, it will take a lot of time.
# print("fine till here")
total = len(X)
inds = [0] * len(X)
# print(len(inds))
for i in range(len(inds)):
	inds[i] = i
random.shuffle(inds)

#nX, nY are some <em>total</em> of the data points selected out of X, Y. 
#We are selecting them randomly from X, Y.
#For better rmse error values, train over the entire X, Y.
#Remember, It will take a lot of time.
nX = []
ny = []
for i in range(total):
	idx = inds[i]
	nX.append(X[idx])
	ny.append(y[idx])

X_train = nX
y_train = ny

# print(len(X), len(y))

#print(X[0:1], 'Printed single row of x')
# print('without scaling', X[1:2]);
#X = preprocessing.scale(X);
#print(X.mean(axis=0), X.std(axis=0))
#print(X[0:1]);
#exit()	

#print(X[1], y[1]);


# ---------------------------------------------------------------------------------------------------------
#Split into train and test data set.
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20);


# ---------------------------------------------------------------------------------------------------------


#####
#Uncomment the below lines to check whether the data set was properly splitted.
####
# print('print train stuff')
# i = 0
# for row_data in X_train:
# 	(xx, yy) = row_data[len(row_data) - 1], y_train[i]
# 	i += 1
# 	myAssert(int(xx) == yy)
# 	print(xx, yy)

# i = 0
# print('print test stuff')
# for row_data in X_test:
# 	(xx, yy) = row_data[len(row_data) - 1], y_test[i]
# 	i += 1
# 	myAssert(int(xx) == yy)
# 	print(xx, yy)

# print(len(X_train), len(X_test), len(y_train), len(y_test))
# exit()


#########################################
#This method is where we apply SVR. We pass various parameters and tune various parameter to ensure a good rmse 
#value is obtained. The current parameters passed in this ensure that a good arousal rmse value is obtained. 
#You can try the same to find a good valence value in the same way.
#########################################
def applySVR(X_train, X_test, y_train, n_components, gamma):
	print('n_components=', n_components, 'gamma=', gamma)

	"""To apply PCA to reduce time. I experimented with quite a values of this.
	Around 150 is the number of features/components that seem to work good for this problem.
	Anyways, a better idea would be check it up again manually by experimenting."""

	# pca = PCA(n_components=n_components).fit(X_train)

	# X_train = pca.transform(X_train)
	# X_test = pca.transform(X_test)

	# clf = NuSVR(C=100.0, cache_size=200, coef0=0.0, degree=3, gamma=gamma,
	#    kernel='rbf', max_iter=-1, nu=0.5, shrinking=True, tol=0.001,
	#    verbose=False)

	clf = NuSVR(C=100, cache_size=200, coef0=0.0, degree=3, gamma=gamma,
	   kernel='rbf', max_iter=-1, nu=0.5, shrinking=True, tol=0.001,
	   verbose=False)

	clf.fit(X_train, y_train) 
	np.set_printoptions(threshold=np.inf)
	#print(len(clf.support_), clf.support_)	

	print('number of test data', len(X_test));
	y_rbf = clf.predict(X_test);
	print('\n\npredictions\n\n')
	# print(y_rbf)
	for i in range(len(y_rbf)):
		# print(X_test[i])
		print(test_files[i]+", "+str(y_rbf[i]))

	# print('predictions made are as follows.')
	# for i in range(len(y_rbf)):
	# 	print(y_rbf[i], y_test[i])

	#for y in y_rbf:
	#	print(y, end=' ')
# 
	"""These are the set of methods which are useful metrics. The paper used rmse value as one of the metrics.
	Here is the one of the ways to find that.
	"""
	# mse = metrics.mean_squared_error(y_test, y_rbf);
	# var = metrics.explained_variance_score(y_test, y_rbf)
	# r2score = metrics.r2_score(y_test, y_rbf)

	# print('mse = ', mse, 'rmse = ', math.sqrt(mse), 'var = ', var, 'r2_score = ', r2score)
	# print('number of features', len(X[0]))
	# print('number of support vectors ', len(clf.support_))
	# print('number of points in training data set ', len(X_train))
	# print('\n', '-----', '\n')


#baseVal = 0.0125

"""These lines are basically for experimenting to find the right set of parameters rather than using the exact 
parameters from the paper mentioned. It's important to run this program and check the corresponding rmse values.
It could take a bit of time to get a good rmse value. The set of values encoded in the program work properly.
It's advised to either use them or some slight tweaking in them could potentially provide a better result"""
baseVal = 0.0125
#Perform the scaling operation after doing PCA.

X_train = preprocessing.scale(X_train);
# for i in range(10):
# 	print(X_train[i])
# print(len(X_test))
# print(len(X_test[0]))
# print(len(X_test[0][0]))
# for i in range(len(X_test)):
# 	print(X_test[i])
X_test = preprocessing.scale(X_test)
# for i in range(len(X_test)):
# 	print(X_test[i])

applySVR(X_train, X_test, y_train, 260, baseVal * 1.0/i)
# print('Number of features after PCA', len(X_train[0]), len(X_test[0]))
# for n_components in range(10, 261, 20):
# 	n_components = 260
# 	for i in range(1, 20, 1):
# 		applySVR(X_train, X_test, y_train, n_components, baseVal * 1.0/i)


"""These are the set of values I experimented with. Please make sure to play with this and experiment over it.
It'll be a good learning experience. In general, some of these work decent. But none is better than manually trying up 
to search a good parameter.
Uncomment the below lines to check for that.
"""
#clf = SVR(kernel='rbf', C=1000, gamma=0.125)
#clf = SVR(kernel='rbf', C=1, epsilon=0.1)
#clf = SVR(kernel='rbf', C=1, epsilon=0.1)
#clf = NuSVR(kernel='rbf', C=2, gamma=0.125, nu=0.1)
#clf = SVR(kernel='rbf', gamma='0.125', C=2);

#clf = NuSVR(C=100.0, cache_size=200, gamma=0.0125,
#   kernel='rbf', max_iter=-1, nu=0.5, shrinking=True, tol=0.001,
#   verbose=False)

#clf = NuSVR(gamma=0.125, C=2000, nu=0.9, kernel='rbf', shrinking=True, verbose=True)


#exit()

#print(clf)	

# print('training data')
# for y in y_train:
# 	print(y)
# print('---')

# print('training set size = ', len(X_train), len(y_train))

# L = len(X[0]);
# print(L);

# temp_test = np.random.rand(1, L)
# print(temp_test)
# y_rbf = clf.predict(temp_test);
# for y in y_rbf:
# 	print(y)

# temp_test = np.random.rand(2, L)
# for i in range(2):
# 	for j in range(L):
# 		temp_test[i][j] = 10.0;
# print(temp_test)
# t_rbf = clf.predict(temp_test);

# for y in t_rbf:
# 	print(y)

#X_test = X_train
#y_test = y_train

#print(stderr, s)

#scores = cross_val_score(clf, X_test, y_test, scoring='mean_squared_error', cv=10);
#print(scores);

#print(scores.mean())

#predicted = cross_val_predict(clf, X, y, cv=10)
#print(metrics.accuracy_score(y, predicted))
# prediction = clf.predict(X);
# print("prediction", prediction);

# ans = 0;
# for i in range(len(y)):
# 	ans += (prediction[i] - y[i]) * (prediction[i] - y[i]) 
# print(math.sqrt(ans) / len(y));


"""Uncomment the below lines if you want to see the plot of how SVR is performed in it
"""
# print('data', len(X), len(y), len(y_rbf));

# lw = 2
# plt.scatter(X[:, [1]], y, color='darkorange', label='data')
# plt.scatter(X[:, [1]], y_rbf, color='navy', label='RBF model')
# plt.xlabel('data')
# plt.ylabel('target')
# plt.title('Support Vector Regression')
# plt.legend()
# plt.show()