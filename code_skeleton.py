from annoy import AnnoyIndex
import random

f = 2
#----------------------------- Here till ----------------------------------------------------
t = AnnoyIndex(f)  # Length of item vector that will be indexed
for i in xrange(1000):
    v = [random.gauss(0, 1) for z in xrange(f)]
    t.add_item(i, v)

t.build(10) # 10 trees
t.save('test.ann')

tt = AnnoyIndex(f)  # Length of item vector that will be indexed
for i in xrange(1000):
    v = [random.gauss(0, 1) for z in xrange(f)]
    tt.add_item(i, v)

tt.build(10) # 10 trees
tt.save('testt.ann')

# ... ------------------------------------- Here will be changed according to datasets--------------------




#------------------- from here is the algorithm for cross modal retrieval ---------------------------------

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
		print("image")
		i = i+1
		im = im*2
	else:
		knn.append(items_audio[a])
		print("audio")
		a = a+1
		au = au*2	
