import os
import sys
import numpy as np

aud_a_file = sys.argv[1]
aud_v_file = sys.argv[2]
im_a_file = sys.argv[3]
im_v_file = sys.argv[4]

aud_vecs = np.zeros((195,2))
im_vecs = np.zeros((195,2))
aud_dict = {}
im_dict = {}

with open(aud_a_file, 'r') as f:
	lines=f.readlines()
	for line in lines:
		name, aval = line.strip().split(',')
		aud_dict[name] = aval

with open(aud_v_file, 'r') as f:
	lines=f.readlines()
	count = 0
	for line in lines:
		name, vval = line.strip().split(',')
		aud_vecs[count][0] = float(aud_dict[name])
		aud_vecs[count][1] = float(vval)
		count += 1

with open(im_a_file, 'r') as f:
	lines=f.readlines()
	for line in lines:
		name, aval = line.strip().split(',')
		im_dict[name] = aval

with open(im_v_file, 'r') as f:
	lines=f.readlines()
	c = 0
	for line in lines:
		name, vval = line.strip().split(',')
		im_vecs[c][0] = float(im_dict[name])
		im_vecs[c][1] = float(vval)
		c += 1


np.save("audio_vecs.npy", aud_vecs)
np.save("image_vecs.npy", im_vecs)