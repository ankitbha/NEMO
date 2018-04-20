import sys
import cPickle as pkl
import numpy as np 
from scipy.spatial import procrustes

# with open("vid_av.pkl", 'r') as f:
#     vid_mat = pkl.load(f)

# with open("aud_av.pkl", 'r') as f:
#     aud_mat = pkl.load(f)

aud_mat = np.load("audio_vecs.npy")
vid_mat = np.load('image_vecs.npy')

mtx_a, mtx_v, disp = procrustes(aud_mat, vid_mat)

print mtx_a
print mtx_v
print disp