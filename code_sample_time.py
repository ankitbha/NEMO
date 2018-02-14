from annoy import AnnoyIndex
import random

f = 2
t = AnnoyIndex(f)  # Length of item vector that will be indexed
for i in xrange(1000):
    v = [random.gauss(0, 1) for z in xrange(f)]
    t.add_item(i, v)

t.build(10) # 10 trees
t.save('test.ann')

# ...

u = AnnoyIndex(f)
u.load('test.ann') # super fast, will just mmap the file
# print(u.get_nns_by_item(0, 1)) # will find the 1000 nearest neighbors
# print(len(u.get_item_vector(0)))
print(u.get_nns_by_vector(u.get_item_vector(0), 10))