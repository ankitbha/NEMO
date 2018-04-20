with open("./valence_predictions.txt", 'r') as f:
	lines = f.readlines()
	count = 0
	sumval = 0.0
	name = ''
	pname = ''
	for line in lines:
		count += 1
		listline = line.split(',')
		value = float(listline[1])
		pname = name
		name = listline[0][:-8]+".jpg"
		if((name != pname) and (pname != '')):
			sumval = sumval/(count-1)
			with open("./valenceimage", 'a') as file:
				file.write(pname+","+str(sumval)+'\n')
			sumval = 0.0
			count = 1	
		sumval += value	