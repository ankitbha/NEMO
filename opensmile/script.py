import os
import subprocess
import sys

files = os.listdir("/home/ankit/Studies/6th_semester/UGP/test_audios")
directory = "/home/ankit/Studies/6th_semester/UGP/test_audios/"
for i in range(len(files)):
	subprocess.call(["SMILExtract", "-C", "config/IS13_ComParE.conf", "--ccmDHELP", "-I", directory+files[i], "-O", "output.csv"])
	subprocess.call(["python3", "R.py", "output.csv", "../feature_test/"+files[i][:-4]+".csv"])