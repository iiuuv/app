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

	ser.write(data_s)

	print(bytes(data_s))
    
	# time.sleep(5)  # 等待数据发送完成

	# lichao = ser.read_all()
     
	# print("Received data:", lichao)
    
   
if __name__ == '__main__':
    sending_data(1,2,3,4,5) 
    
	
