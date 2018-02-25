import os
import sys
import csv
import pandas as pd

def get_im_data(file1, file2):
	col_names = ['a', 'b']
	df = pd.read_csv(file1, sep = ',', names = col_names)
	dff = pd.read_csv(file2, sep = ',', names = col_names)
	data = []
	data2 = []
	for i in range(len(df['a'])):
		if(df['a'][i] == dff['a'][i]):
			data.append(df['a'][i])
			data2.append([float(df['b'][i]), float(dff['b'][i])])

	return data, data2		