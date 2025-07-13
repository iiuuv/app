#!/usr/bin python3

import sys
import signal
import os
import time


# 导入python串口库
import serial

import struct

def signal_handler(signal, frame):
    sys.exit(0)

def sending_data(B,C,D,E,G):

	ser = serial.Serial("/dev/ttyS1", 115200, timeout=1)

	data_s = struct.pack("<BBBBB",int(B),int(C),int(D),int(E),int(G))

	ser.write(bytes(data_s))
	
	ser.write(data_s)
	print(bytes(data_s))

	
    
	# time.sleep(5)  # 等待数据发送完成

	# lichao = ser.read_all()
     
	# print("Received data:", lichao)
    
   
if __name__ == '__main__':
	ser = serial.Serial("/dev/ttyS1", 115200, timeout=1)
	
	# while True:
	# 	# sending_data(1,2,3,4,5)
	# 	if ser.read_all():
	# 		ser.write('R'.encode('UTF-8'))
	# 		print("Success to connect to the serial device.")
	# 	time.sleep(0.5)	
	# 	# sending_data(1,2,3,4,5)

	x=b'T\x02\x00\x07ET\x00\x03*ET\x01\x00\x0eET\x00\x01cET\x01\x00\x0eET\x00\x00.ET\x02\x00\x12ET\x00\x01\'ET\x02\x00\x17ET\x00\x00pE'
	y=b'T\x01\x00\xb4ET\x00\x00\x00ET\x00\x00\x00ET\x00\x00\x00ET\x00\x00\x00ET\x00\x00\x00ET\x00\x00\x00ET\x00\x00\x00ET\x00\x00\x00ET\x00\x00\x00E'
	z=b'T\x01\x00\x93ET\x00\x07\x12ET\x02\x00\x0eET\x00\x05\xeeET\x02\x00\nET\x00\x00\xbcE'
	a=b'T\x01\x00 ET\x00\x04\x9fET\x01\x00\x16ET\x00\x00\xb2ET\x01\x00\rET\x00\x01\x07ET\x01\x00\x07ET\x00\x02\xdcET\x02\x00\x78E'
	
	b=b'T\x01\x00 ET\x00\x04\x9fE'

	c=b'T\x03\x13\x88E'
	
	print(len(b))
	ser.write(b)
	print(len(c))
	ser.write(c)

