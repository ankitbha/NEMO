import numpy as np
import csv
import math
import sys

filename = sys.argv[1]
file = open(filename, "r")
filename2 = sys.argv[2]
count = 0
count2 = 0
k = 0
sumvec = np.zeros(131)
stddev = np.zeros(130)
# writevec = []
# for i in range(261):
# 	writevec.append(0.0)
finalvec = np.zeros((30,261))
writevec = np.zeros(261)
collecvec = np.zeros((50, 131))
for line in file.readlines():
	if line != '':
		line = line.split(";")
		line = line[1:]
		# print(line)
		if(len(line)!=0):
			a = line[-1]
			a = a[:-1]
			line = line[:-1]
			line.append(a)
			vector = np.zeros(131)
			# if(count == 0):
			# 	Line = []
			# 	for i in range(len(line)):
			# 		if(i==0):
			# 			Line.append(line[i])
			# 		else:	
			# 			Line.append(line[i] + '_amean')
			# 			Line.append(line[i] + '_stddev')	
			# 	myData = ','.join(Line)
			# 	myFile = open(filename2, 'w')
				
			# 	    # writer = csv.writer(myFile)
			# 	    # print(myData)
			# 	myFile.write(myData)
			# 	# myFile.close()    
			if(count != 0):
				i = 0
				for obj in line:
					obj = float(obj)
					vector[i] = obj
					i += 1
			# if(count == 4):
			# 	print(vector)
				sumvec += vector
				collecvec[count2-1] = vector
			count += 1
			count2 = (count2+1)%50
			if(count2 == 0):
				sumvec = sumvec/50
				sumvec[0] = count/100.0
				for j in range(131):
					if(j!=0):
						for i in range(50):
							writevec[2*j] = float(writevec[2*j]) + (float(collecvec[i][j-1])-float(sumvec[j]))*(float(collecvec[i][j-1])-float(sumvec[j]))
						writevec[2*j] = writevec[2*j]/49
						writevec[2*j] = math.sqrt(writevec[2*j])
						writevec[2*j-1] = sumvec[j]
							# print(collecvec[i][j-1])
						# print(sumvec[j])	 
				# print(sumvec)
				writevec[0] = 0.0
				writevec[0] += float(sumvec[0])
				finalvec[k] = writevec
				k += 1
				sumvec = np.zeros(131)
				writevec = np.zeros(261)
myfile = open(filename2, 'wb')
np.savetxt(myfile, finalvec,  fmt='%.4e', delimiter=",")		
