import re

with open("./arousal_au", 'r') as f:
	lines = f.readlines()
	reg = r",\ "
	for line in lines:
		line = re.sub(reg, r",", line, 0, re.MULTILINE)
		with open("./arousalaudio", 'a') as file:
			file.write(line)

with open("./valence_au", 'r') as f:
	lines = f.readlines()
	reg = r",\ "
	for line in lines:
		line = re.sub(reg, r",", line, 0, re.MULTILINE)
		with open("./valenceaudio", 'a') as file:
			file.write(line)