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
    
   

def wait_for_complete(ser:serial.Serial,command:bytes):
	buf = []
	# ser.write(command)
	count = len(command)/5
	while True:
		x = ser.readall()
		if (x):
			buf += x
		if len(buf) >= count:
			break
		time.sleep(0.1)

def take_off_and_go(ser:serial.Serial):
    take_off = [84, 0, 0, 0, 220, 69]
    right_move = [84, 1, 0, 90, 220, 69]
    ser.write(bytes(take_off))
    time.sleep(5)
    ser.write(bytes(right_move))
    # time.sleep(5)

def turn_back_and_land(ser:serial.Serial):
	left_move = [84, 1, 1, 14, 220, 69]
	land = [84, 2, 0, 0, 0, 69]
	ser.write(bytes(left_move))
	time.sleep(7)
	ser.write(bytes(land))
	while 1:
		time.sleep(5)

def wait_for_signal(ser:serial.Serial):
	buf = "a"
	while True:
		buf = ser.readall()
		if len(buf) == 1 and buf[0] == 73:
			break

def wait_for_fire(ser:serial.Serial):
	buf = "a"
	while True:
		buf = ser.readall()
		if len(buf) == 1 and buf[0] == 87:
			break		

def text(ser:serial.Serial):
	take_off = [84, 0, 0, 0, 210, 69]
	land = [84, 2, 0, 0, 0, 69]
	
	ser.write(bytes(take_off))
	time.sleep(20)
	ser.write(bytes(land))
	time.sleep(5)

	

if __name__ == '__main__':
	ser = serial.Serial("/dev/ttyS1", 115200, timeout=1)

	# take_off_and_go(ser)
	
	# time.sleep(5)

	# turn_back_and_land(ser)
	# 	# sending_data(1,2,3,4,5)
	# 	if ser.read_all():
	# 		ser.write('R'.encode('UTF-8'))
	# 		print("Success to connect to the serial device.")
	# 	time.sleep(0.5)	
	# 	# sending_data(1,2,3,4,5)

	# wait_for_signal(ser)

	# text(ser)

	x=b'T\x02\x00\x07ET\x00\x03*ET\x01\x00\x0eET\x00\x01cET\x01\x00\x0eET\x00\x00.ET\x02\x00\x12ET\x00\x01\'ET\x02\x00\x17ET\x00\x00pE'
	y=b'T\x01\x00\xb4ET\x00\x00\x00ET\x00\x00\x00ET\x00\x00\x00ET\x00\x00\x00ET\x00\x00\x00ET\x00\x00\x00ET\x00\x00\x00ET\x00\x00\x00ET\x00\x00\x00E'
	z=b'T\x01\x00\x93ET\x00\x07\x12ET\x02\x00\x0eET\x00\x05\xeeET\x02\x00\nET\x00\x00\xbcE'
	a=b'T\x02\x00\x1eET\x00\x02TET\x02\x00\x13ET\x00\x00\x96ET\x01\x00\x19ET\x00\x00\xd8ET\x01\x007ET\x00\x01\x7fET\x02\x00\x10ET\x00\x004ET\x02\x00\x08ET\x00\x00JET\x02\x00\x0eET\x00\x027ET\x01\x00\x13ET\x00\x01QET\x02\x00]ET\x00\x00\x00E'
	
	b=b'T\x00\x01\x99E'

	c=b'T\x01\x13\x88E'
	
	print(len(a))
	# wait_for_complete(ser,b)
	# print("cx")
	ser.write(b)

