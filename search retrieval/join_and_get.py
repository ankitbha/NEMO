import os
import sys
import csv
import pandas as pd


def get_data(file1, file2):
	col_names = ['song_id', 'valence_mean', 'valence_std', 'arousal_mean', 'arousal_std']

	df = pd.read_csv(file1, names = col_names)
	column1 = df['song_id']

	col_names = ['song_id', 'valence_mean', 'a', 'b', 'c', 'd', 'e', 'arousal_mean', 'a1', 'b1', 'c1', 'd1', 'e1']
	dff = pd.read_csv(file2, names = col_names)
	column = dff['song_id']
	column1 = column1.append(column, ignore_index = True)
	data2 = []
	for j in range(1,len(column1)):
		if(j != 1745):
			data2.append(column1[j]+"s")


	cv1 = df['valence_mean']
	cv2 = df['arousal_mean']
	cvv1 = dff['valence_mean']
	cvv2 = dff['arousal_mean']
	cv1 = cv1.append(cvv1, ignore_index = True)
	cv2 = cv2.append(cvv2, ignore_index = True)

	data1 = []

	for i in range(1,len(cv1)):
		if(i != 1745):
			temp = [float(cv1[i]), float(cv2[i])]
			data1.append(temp)


	return data2, data1
