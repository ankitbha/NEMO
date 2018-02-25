import os
import sys
import csv
import pandas as pd
from get_from_image import get_im_data
from join_and_get import get_data
from annoy import AnnoyIndex

# run like this : python  finalcode.py ./im_arousal.txt ./im_valence.txt ./static_annotations_averaged_songs_1_2000.csv ./static_annotations_averaged_songs_2000_2058.csv

file1 = sys.argv[1]
file2 = sys.argv[2]
file3 = sys.argv[3]
file4 = sys.argv[4]

im_item, im_vector = get_im_data(file1, file2)
au_item, au_vector = get_data(file3, file4)

f=2
t = AnnoyIndex(f)
for i in range(len(im_item)):
	t.add_item(i, im_vector[i])

t.build(10)
t.save('test.ann')

tt = AnnoyIndex(f)
for j in range(len(au_item)):
	tt.add_item(j, au_vector[j])

tt.build(10)
tt.save('testt.ann')

#-------------------------------------------------------------------------------------------------------

k = 10 # this is the knn k that we need
u = AnnoyIndex(f)
u.load('test.ann') # super fast, will just mmap the file
items_image = u.get_nns_by_item(0, k, include_distances = True)[0] # will find the 1000 nearest neighbors
# print(len(u.get_nns_by_item(0,1000)))
distances_image = u.get_nns_by_item(0, k, include_distances = True)[1]
uu = AnnoyIndex(f)
uu.load('testt.ann')
items_audio = uu.get_nns_by_item(0, k, include_distances = True)[0]
distances_audio = uu.get_nns_by_item(0, k, include_distances = True)[1]
knn = []
i = 0
a = 0
im = 1
au = 1
for it in range(k):
	if ((distances_image[i]*im) <= (distances_audio[a]*au)):
		knn.append(items_image[i])
		print(im_item[items_image[i]])
		i = i+1
		im = im*2
	else:
		knn.append(items_audio[a])
		print(au_item[items_audio[a]])
		a = a+1
		au = au*2			