import socket
import threading
import struct
import json
import cv2
import sys
import serial
import time

ser = serial.Serial('/dev/ttyS1', 115200, timeout=1) # 1s timeout
PORT = 8888         
PC_IP = "192.168.1.253"
has_received_enable = False

def take_off_and_go(ser:serial.Serial):
    take_off = [84, 0, 0, 0,0, 110, 69]
    right_move = [84, 1, 0, 90,0, 50, 69]
    print("Taking off and moving right")
    ser.write(bytes(take_off))
    time.sleep(5)
    ser.write(bytes(right_move))
    # time.sleep(5)

def execute_command(json_data):
    global has_received_enable  # 引用全局变量
    
    # 先检查当前命令是否为"enable"，如果是则更新状态
    current_type = json_data.get("type")
    if current_type == "enable":
        has_received_enable = True
        return [0x00] * 7  # 这里返回全0，也可根据实际需求修改
    
    if current_type == "auto":
        take_off_and_go(ser)
        return [0x00] * 7  # 这里返回全0，也可根据实际需求修改
    
    # 如果从未接收过"enable"，直接返回全0x00
    if not has_received_enable:
        return [0x00] * 7
    
    # 以下是原有逻辑（仅当has_received_enable为True时执行）
    command_type = current_type
    direction = int(json_data.get("direction", 0))  # 增加默认值，避免KeyError
    distance = int(json_data.get("distance", 0))    # 增加默认值，避免KeyError

    # 初始化命令列表
    command_list = [
        0x54, 
        0, 
        int(direction / 256) % 256, 
        int(direction % 256), 
        int(distance / 256) % 256, 
        int(distance % 256), 
        0x45
    ]

    # 根据命令类型设置第二个字节
    if command_type == 'move':
        command_list[1] = 0x01
    elif command_type == 'takeoff':
        command_list[1] = 0x00
    elif command_type == 'land':
        command_list[1] = 0x02


    return command_list

def receive_commands(client_socket, client_id):
    """接收服务器发送的命令"""
    while True:
        try:
            # 接收命令长度
            buf = ser.readall()
            packed_msg_size = client_socket.recv(struct.calcsize("I"))
            if not packed_msg_size:
                break

            msg_size = struct.unpack("I", packed_msg_size)[0]

            # 接收命令数据
            command_data = b''
            while len(command_data) < msg_size:
                packet = client_socket.recv(msg_size - len(command_data))
                if not packet:
                    break
                command_data += packet
            
            if not command_data:
                break
            
            # 解码命令
            command = command_data.decode('utf-8')
            
            # 尝试解析为JSON
            try:
                json_data = json.loads(command)
                print(f"客户端 {client_id} 接收到JSON: {json.dumps(json_data, indent=2, ensure_ascii=False)}")
                serdata=bytes(execute_command(json_data))
                print(serdata)
                ser.write(serdata)
            except json.JSONDecodeError:
                # 不是JSON格式，直接打印命令
                print(f"客户端 {client_id} 接收到命令: {command}")
                
        except Exception as e:
            print(f"客户端 {client_id} 接收命令时出错: {e}")
            break

if __name__ == "__main__":
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((PC_IP, PORT))  # 连接PC端
        print(f"已连接到PC: {PC_IP}:{PORT}")
    except Exception as e:
        print(f"Socket连接失败: {e}")
        sys.exit(-1)
    client_socket.sendall('1'.encode('utf-8'))    
    
    receive_thread = threading.Thread(target=receive_commands, args=(client_socket, 2), daemon=True)
    receive_thread.start()
    receive_thread.join()
    # receive_commands(client_socket, 2)
