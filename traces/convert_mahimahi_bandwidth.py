import numpy as np
import matplotlib.pyplot as plt
import os

PACKET_SIZE = 1500.0  # bytes
BITS_IN_BYTE = 8.0
MBITS_IN_BITS = 1000000.0
MILLISECONDS_IN_SECONDS = 1000.0
MILLISECONDS_IN_SLOT = 100.0
N = 100
FILE_PATH = './mahimahi/'
OUTPUT_PATH = './bandwidth/'
files = os.listdir(FILE_PATH)

index=0
for trace_file in files:
	time_all = []
	packet_sent_all = []
	last_time_stamp = 0
	packet_sent = 0
	with open(FILE_PATH+trace_file, 'rb') as f,open(OUTPUT_PATH + str(index)+".log", 'wb') as mf:
		for line in f:
			time_stamp = int(line.split()[0])
			if time_stamp <= last_time_stamp+MILLISECONDS_IN_SLOT:
				packet_sent += 1
				continue
			else:
				time_all.append(last_time_stamp)
				packet_sent_all.append(packet_sent)
				packet_sent = 1
				last_time_stamp +=MILLISECONDS_IN_SLOT
	#time_window = np.array(time_all[1:]) - np.array(time_all[:-1])
	# for i,j in zip(time_window,range(len(time_window))):
	# 	if i!=1: print(j)
	# throuput_all = PACKET_SIZE * \
	# 			   BITS_IN_BYTE * \
	# 			   np.array(packet_sent_all[1:]) / \
	# 			   time_window * \
	# 			   MILLISECONDS_IN_SECONDS / \
	# 			   MBITS_IN_BITS
		throuput_all = PACKET_SIZE * \
				   BITS_IN_BYTE * \
				   np.array(packet_sent_all)*\
				   MILLISECONDS_IN_SECONDS/\
				   MILLISECONDS_IN_SLOT/\
				   MBITS_IN_BITS
		if len(throuput_all)>=2000:
			for i in range(len(throuput_all)):
				mf.write(bytes(str(throuput_all[i]) + '\n',encoding='utf-8'))
			index+=1