#!/user/bin/env python

# Copyright (c) 2024，WuChao D-Robotics.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 注意: 此程序在RDK板端端运行
# Attention: This program runs on RDK board.
import serial
import cv2
import numpy as np
from scipy.special import softmax
# from scipy.special import expit as sigmoid
from hobot_dnn import pyeasy_dnn as dnn  # BSP Python API
from pathfinder import search_path
from time import time
import argparse
import logging
import math
from serial1 import sending_data
from zoom1 import process
from firstBSPrealtime import BPU_Detect
import mmap
import os
import subprocess
import ctypes
import struct
import json
# import time

EFD_SEMAPHORE = 1
EFD_NONBLOCK = 0o4000
EFD_CLOEXEC = 0o2000000
# import rclpy
# from rclpy.node import Node
# from std_msgs.msg import BoolMultiArray
# from std_msgs.msg import MultiArrayDimension

# class BoolArrayPublisher(Node):
#     def __init__(self):
#         super().__init__('bool_array_publisher')
#         self.publisher_ = self.create_publisher(BoolMultiArray, '/bool_array', 10)

#     def publish_array(self, array_2d):
#         msg = BoolMultiArray()
#         height, width = array_2d.shape
#         msg.data = array_2d.flatten().tolist()

#         msg.layout.dim.append(MultiArrayDimension())
#         msg.layout.dim[0].label = "height"
#         msg.layout.dim[0].size = height
#         msg.layout.dim[0].stride = height * width

#         msg.layout.dim.append(MultiArrayDimension())
#         msg.layout.dim[1].label = "width"
#         msg.layout.dim[1].size = width
#         msg.layout.dim[1].stride = width

#         self.publisher_.publish(msg)
#         self.get_logger().info('已发送布尔数组')
# 日志模块配置
# logging configs
logging.basicConfig(
    level = logging.DEBUG,
    format = '[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S')
logger = logging.getLogger("RDK_YOLO")


def pixelate_mask(mask, target_size=100, keep_aspect_ratio=True):
    """
    对掩码进行像素化处理，保持原始宽高比（不裁剪为正方形），缩小像素尺寸
    
    参数:
    mask: 原始掩码图 (numpy数组)
    target_size: 目标短边尺寸（控制缩放程度，短边会缩放到此值）
    keep_aspect_ratio: 是否保持原始宽高比（固定为True，确保比例不变）
    
    返回:
    像素化后的掩码图 (numpy数组，保持原始宽高比，短边为target_size)
    """
    # 确保输入是二维或三维数组
    if len(mask.shape) == 2:
        h, w = mask.shape
        is_gray = True
    elif len(mask.shape) == 3:
        h, w, c = mask.shape
        is_gray = c == 1
    else:
        raise ValueError("输入掩码必须是二维或三维数组")
    
    # 强制保持宽高比（忽略传入的False，确保比例不变）
    keep_aspect_ratio = True  # 固定为True，确保不拉伸图像
    
    if keep_aspect_ratio:
        # 保持原始宽高比，将短边缩放到target_size（核心修改：不裁剪为正方形）
        scale = target_size / min(h, w)
        target_h, target_w = int(h * scale), int(w * scale)
        
        # 确保目标尺寸至少为1像素
        target_h = max(1, target_h)
        target_w = max(1, target_w)
        
        # 第一步：降采样到目标尺寸（生成像素化效果）
        # 直接缩放到目标尺寸（短边=target_size，长边按比例缩放）
        pixelated = cv2.resize(
            mask, 
            (target_w, target_h), 
            interpolation=cv2.INTER_NEAREST  # 最近邻插值，保持硬边缘
        )
        
        # 核心修改：移除裁剪和填充正方形的逻辑，直接保留按比例缩放的结果
        result = pixelated
    
    else:
        # 即使传入False，仍按保持比例处理（避免拉伸，与需求一致）
        scale = target_size / min(h, w)
        target_h, target_w = int(h * scale), int(w * scale)
        target_h = max(1, target_h)
        target_w = max(1, target_w)
        result = cv2.resize(
            mask, 
            (target_w, target_h), 
            interpolation=cv2.INTER_NEAREST
        )
    
    return result.astype(np.uint8)

target_colors = [(236, 24, 0), (29, 178, 255),(199, 55, 255)]

def binarize_image_to_array(image, target_colors, specific_colors, tolerance=30, black_expand_size=1, color_expand_size=0):
    """
    将图片二值化并转换为二维数组，分别扩大黑色区域和两种特定颜色区域
    
    参数:
    image (str/numpy.ndarray): 图片路径或图片数组
    target_colors (list): 目标颜色列表，每个颜色格式为(B,G,R)
    specific_colors (list): 需要额外扩大的两种特定颜色，格式为[(B1,G1,R1), (B2,G2,R2)]
    tolerance (int): 颜色匹配容忍度，值越大匹配范围越广
    black_expand_size (int): 黑色区域向外扩展的像素数
    color_expand_size (int): 特定颜色区域向外扩展的像素数
    
    返回:
    numpy.ndarray: 二值化后的二维数组，黑色区域为0，特定颜色区域及其扩展为1，其他区域为0
    """
    # 确保有两种特定颜色
    if len(specific_colors) != 2:
        raise ValueError("specific_colors参数必须包含两种颜色")
    
    # 判断输入是路径还是数组
    if isinstance(image, str):
        # 如果是路径，则读取图片
        img = cv2.imread(image)
        if img is None:
            raise ValueError(f"无法读取图片: {image}")
    else:
        # 如果是数组，直接使用
        img = image
    
    # 创建黑色区域掩码
    black_mask = cv2.inRange(img, (0, 0, 0), (50, 50, 50))
    
    # 初始化目标颜色掩码（这些颜色最终会被转换为黑色）
    target_mask = np.zeros_like(black_mask)
    
    # 为每种目标颜色创建掩码并合并
    for color in target_colors:
        lower_bound = np.array([max(0, c - tolerance) for c in color])
        upper_bound = np.array([min(255, c + tolerance) for c in color])
        target_mask = cv2.bitwise_or(target_mask, cv2.inRange(img, lower_bound, upper_bound))
    
    # 分离两种特定颜色的掩码
    color1, color2 = specific_colors
    
    # 创建第一种特定颜色的掩码
    lower1 = np.array([max(0, c - tolerance) for c in color1])
    upper1 = np.array([min(255, c + tolerance) for c in color1])
    specific_mask1 = cv2.inRange(img, lower1, upper1)
    
    # 创建第二种特定颜色的掩码
    lower2 = np.array([max(0, c - tolerance) for c in color2])
    upper2 = np.array([min(255, c + tolerance) for c in color2])
    specific_mask2 = cv2.inRange(img, lower2, upper2)
    
    # 分别扩大黑色区域、第一种特定颜色区域和第二种特定颜色区域
    if black_expand_size > 0:
        kernel_black = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (black_expand_size*2+1, black_expand_size*2+1))
        expanded_black = cv2.dilate(black_mask, kernel_black, iterations=1)
    else:
        expanded_black = black_mask.copy()
    
    if color_expand_size > 0:
        kernel_color = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (color_expand_size*2+1, color_expand_size*2+1))
        expanded_color1 = cv2.dilate(specific_mask1, kernel_color, iterations=1)
        expanded_color2 = cv2.dilate(specific_mask2, kernel_color, iterations=1)
    else:
        expanded_color1 = specific_mask1.copy()
        expanded_color2 = specific_mask2.copy()
    
    # 合并所有需要变为黑色的区域（黑色区域 + 目标颜色区域）
    black_regions = cv2.bitwise_or(expanded_black, target_mask)
    
    # 合并两种特定颜色的扩展区域
    expanded_specific_regions = cv2.bitwise_or(expanded_color1, expanded_color2)
    
    # 创建最终掩码：
    # 1. 先将特定颜色的扩展区域设为白色(255)
    # 2. 再将黑色区域设为黑色(0)，但只覆盖那些不在特定颜色扩展区域内的部分
    final_mask = np.zeros_like(black_regions)
    final_mask[expanded_specific_regions > 0] = 255  # 特定颜色扩展区域设为白色
    
    # 只在特定颜色扩展区域之外的地方应用黑色区域
    non_specific_mask = np.ones_like(black_regions, dtype=bool)
    non_specific_mask[expanded_specific_regions > 0] = False
    final_mask[np.logical_and(black_regions > 0, non_specific_mask)] = 0
    
    # 将掩码转换为0和1的二维数组
    binary_array = (final_mask > 0).astype(np.uint8).tolist()
    
    return binary_array

# def binarize_image_to_array(image, target_colors, tolerance=30, expand_size=2):
#     """
#     将图片二值化并转换为二维数组，同时扩大目标颜色区域
    
#     参数:
#     image (str/numpy.ndarray): 图片路径或图片数组
#     target_colors (list): 目标颜色列表，每个颜色格式为(B,G,R)
#     tolerance (int): 颜色匹配容忍度，值越大匹配范围越广
#     expand_size (int): 目标区域向外扩展的像素数
    
#     返回:
#     numpy.ndarray: 二值化后的二维数组，黑色为0，白色为1
#     """
#     # 判断输入是路径还是数组
#     if isinstance(image, str):
#         # 如果是路径，则读取图片
#         img = cv2.imread(image)
#         if img is None:
#             raise ValueError(f"无法读取图片: {image}")
#     else:
#         # 如果是数组，直接使用
#         img = image
    
#     # 创建二值化掩码
#     # 黑色区域
#     black_mask = cv2.inRange(img, (0, 0, 0), (50, 50, 50))
    
#     # 初始化颜色掩码
#     color_mask = np.zeros_like(black_mask)
    
#     # 为每种目标颜色创建掩码并合并
#     for color in target_colors:
#         lower_bound = np.array([max(0, c - tolerance) for c in color])
#         upper_bound = np.array([min(255, c + tolerance) for c in color])
#         color_mask = cv2.bitwise_or(color_mask, cv2.inRange(img, lower_bound, upper_bound))
    
#     # 合并所有掩码（黑色和所有目标颜色都变为黑色(0)）
#     combined_mask = cv2.bitwise_or(black_mask, color_mask)
    
#     # 扩大黑色区域（目标颜色及其周边）
#     if expand_size > 0:
#         # 创建膨胀核（可以是矩形、椭圆等）
#         kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand_size*2+1, expand_size*2+1))
#         # 应用膨胀操作
#         combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)
    
#     # 将合并后的掩码转换为0和1的二维数组
#     # 黑色区域(0)保持为0，其他区域(255)变为1
#     binary_array = (combined_mask == 0).astype(np.uint8).tolist()
    
#     return binary_array

def visualize_binary_array(binary_array, output_path, scale=1):
    """
    将二值化后的二维数组保存为黑白图像
    
    参数:
    binary_array (list): 二值化后的二维数组，0表示黑色，1表示白色
    output_path (str): 图像保存路径，例如 "result.png"
    scale (float): 图像缩放比例
    
    返回:
    None
    """
    # 将二维数组转换为NumPy数组
    binary_array = np.array(binary_array, dtype=np.uint8)
    
    # 创建黑白图像（0=黑色，255=白色）
    binary_image = binary_array * 255
    
    # 调整图像尺寸
    if scale != 1:
        height, width = binary_image.shape
        binary_image = cv2.resize(binary_image, (int(width*scale), int(height*scale)))
    
    # 保存图像
    try:
        cv2.imwrite(output_path, binary_image)
        print(f"二值化图像已成功保存到: {output_path}")
    except Exception as e:
        print(f"保存图像时出错: {e}")

def eventfd(initval=0, flags=0):
    libc = ctypes.CDLL("libc.so.6")
    fd = libc.eventfd(initval, flags)
    if fd < 0:
        raise OSError("eventfd failed")
    return fd

# 创建共享内存并映射
def create_shm(name, size):
    shm_fd = os.open(name, os.O_CREAT | os.O_TRUNC | os.O_RDWR, 0o666)
    os.ftruncate(shm_fd, size)
    return mmap.mmap(shm_fd, size), shm_fd

def create_shared_memory(data, size=1024):
    # 创建临时文件用于共享内存
    temp_file = '/tmp/shared_memory'
    with open(temp_file, 'wb') as f:
        # 调整文件大小
        f.write(b'\0' * size)
        # 写入数据
        f.seek(0)
        f.write(data.encode())

    return temp_file

def main():
    # 初始化共享内存
    SHM_MAP = "/dev/shm/shm_map"
    SHM_DATA = "/dev/shm/shm_json"
    SHM_RESULT = "/dev/shm/shm_result"

    # 各段大小
    SHM_MAP_SIZE = 133 * 100  # 地图数据
    SHM_DATA_SIZE = 200     # JSON 字符串最大长度
    SHM_RESULT_SIZE = 2048    # 示例：返回 4 个整数
    CPP_PROGRAM = "/app/my_project/path/build/pathCXX2"  # C++ 可执行文件路径

    shm_map, fd_map = create_shm(SHM_MAP, SHM_MAP_SIZE)
    shm_data, fd_data = create_shm(SHM_DATA, SHM_DATA_SIZE)
    shm_result, fd_result = create_shm(SHM_RESULT, SHM_RESULT_SIZE)

    # 启动 C++ 子任务
    efd = eventfd(0,0)
    print(f"启动 C++ 子进程，eventfd = {efd}")
    # cpp_process = subprocess.Popen([CPP_PROGRAM, str(efd)], pass_fds=(efd,))

    rangle=0
    rdistance=0
    zoomm=5.77#576/200

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='/app/my_project/bin/converted_model_modified5.bin', 
                        help="""Path to BPU Quantized *.bin Model.
                                RDK X3(Module): Bernoulli2.
                                RDK Ultra: Bayes.
                                RDK X5(Module): Bayes-e.
                                RDK S100: Nash-e.
                                RDK S100P: Nash-m.""") 
    # parser.add_argument('--test-img', type=str, default='../../../../resource/datasets/COCO2017/assets/bus.jpg', help='Path to Load Test Image.')
    parser.add_argument('--test-img', type=str, default='/app/my_project/photo/fire_screenshot_19.07.2025.png', help='Path to Load Test Image.')
    parser.add_argument('--mask-save-path', type=str, default='/app/my_project/v8seg-photo/mask.jpg', help='Path to Save Mask Image.')
    # parser.add_argument('--mask2-save-path', type=str, default='/app/my_project/v8seg-photo/mask2.jpg', help='Path to Save Mask Image.')
    parser.add_argument('--img-save-path', type=str, default='/app/my_project/v8seg-photo/result.jpg', help='Path to Load Test Image.')
    parser.add_argument('--classes-num', type=int, default=5, help='Classes Num to Detect.')
    parser.add_argument('--nms-thres', type=float, default=0.7, help='IoU threshold.')
    parser.add_argument('--score-thres', type=float, default=0.86, help='confidence threshold.')
    parser.add_argument('--reg', type=int, default=16, help='DFL reg layer.')
    parser.add_argument('--mc', type=int, default=32, help='Mask Coefficients')
    parser.add_argument('--is-open', type=bool, default=True, help='Ture: morphologyEx')
    parser.add_argument('--is-point', type=bool, default=True, help='Ture: Draw edge points')
    opt = parser.parse_args()
    logger.info(opt)
    Buildmap=0
    fire_radius=0
    fire_center=(0,0)
    ser = serial.Serial('/dev/ttyS1', 115200, timeout=1) # 1s timeout
    # if ser.readall():
    #     print("Success to connect to the serial device.")
    #     # aaaaaa=ser.readall()
    #     ser.write("r")

    #飞机飞行指令
    # drone_command(ser, drone_command_type.TAKE_OFF, 0, 0)
    coconame2 = ["fire","smoke"]
    models1 = "/app/my_project/bin/converted_model2.bin"
    infer = BPU_Detect(models1,coconame2,conf=0.5,iou=0.3,mode = True)


    # 实例化
    model = YOLO11_Seg(opt)
    # 读图
    img = cv2.imread(opt.test_img)

    infer.detect(img)
    for class_id, score, bbox in zip(infer.ids, infer.scores, infer.bboxes):
        x11, y11, x22, y22 = bbox
        fire_radius = math.sqrt((x11-x22)**2+(y11-y22)**2)*0.5
        fire_center = ((x11+x22)//2,(y11+y22)//2)
        infer.draw_detection(img, (x11, y11, x22, y22), score, class_id, infer.labelname)
    print(fire_center,fire_radius,(fire_center[0]/zoomm,fire_center[1]/zoomm),fire_radius/zoomm)

    # 准备输入数据
    input_tensor = model.preprocess_yuv420sp(img)
    # 推理
    outputs = model.c2numpy(model.forward(input_tensor))
    # 后处理
    results = model.postProcess(outputs)
    # 渲染
    logger.info("\033[1;32m" + "Draw Results: " + "\033[0m")
    # 绘制
    draw_img = img.copy()
    zeros = np.zeros((img.shape[0],img.shape[1],3), dtype=np.uint8)
    #掩码处理部分
    for class_id, score, x1, y1, x2, y2, mask in results:
        # 计算车的实时方向
        if class_id == 3:
            tou = ((x1 + x2) // 2, (y1 + y2) // 2)
        if class_id == 4:
            wei = ((x1 + x2) // 2, (y1 + y2) // 2)

        try:
            dx = tou[0]-wei[0]
            dy = tou[1]-wei[1]
            car_center=((tou[0]+wei[0])//2,(tou[1]+wei[1])//2)
            radius=abs(dy-dx)*0.75
            print('radius:',radius)

            # 计算弧度角，注意：atan2 参数是 (y, x)
            angle_rad = math.atan2(dy, dx)

            # 转换为角度，并调整范围到 [0, 360)
            angle_deg = 180-math.degrees(angle_rad)
            if angle_deg < 0:
                angle_deg += 360
            if angle_deg > 360:
                angle_deg -= 360
            print(f"Direction Angle: {angle_deg:.2f}°")
        except NameError as e:
            print("缺少车头或车尾位置信息，无法计算方向。")

        # Detect
        print("(%d, %d, %d, %d) -> %s: %.2f"%(x1,y1,x2,y2, coco_names[class_id], score))
        draw_detection(draw_img, (x1, y1, x2, y2), score, class_id)
        # Instance Segment
        if mask.size == 0:
            continue
        mask = cv2.resize(mask, (int(x2-x1), int(y2-y1)), interpolation=cv2.INTER_LINEAR)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, model.kernel_for_morphologyEx, 1) if opt.is_open else mask      
        zeros[y1:y2,x1:x2, :][mask == 1] = rdk_colors[(class_id-1)%20]
        # points
        if not opt.is_point:
            continue
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # 手动连接轮廓
            contour = np.vstack((contours[0], np.array([contours[0][0]])))
            for i in range(1, len(contours)):
                contour = np.vstack((contour, contours[i], np.array([contours[i][0]])))
            # 轮廓投射回原来的图像大小
            merged_points = contour[:,0,:]
            merged_points[:,0] = merged_points[:,0] + x1
            merged_points[:,1] = merged_points[:,1] + y1
            points = np.array([[[int(x), int(y)] for x, y in merged_points]], dtype=np.int32)
            # 绘制轮廓
            cv2.polylines(draw_img, points, isClosed=True, color=rdk_colors[(class_id-1)%20], thickness=4)

    # 可视化, 这里采用直接相加的方式，实际应用中对Mask一般不需要Resize这些操作
    add_result = np.clip(draw_img + 0.3*zeros, 0, 255).astype(np.uint8)
    # 缩放起始点
    if car_center[0]<384:
        start_x= car_center[0]+0.05*abs(car_center[0]-384)
    else:
        start_x= car_center[0]-0.05*abs(car_center[0]-384)
    if car_center[1]<288:
        start_x= car_center[1]+0.05*abs(car_center[1]-384)
    else:
        start_x= car_center[1]-0.05*abs(car_center[1]-384)
    # start_x, start_y = car_center[0]+0.05*abs(car_center[0]-384), car_center[1]-0.05*(car_center[1]-288)
    # start_x, start_y = 120,120
    #透视变换
    zeros=process(zeros)
    
    t0 = time()
    # 低像素化
    # map2=pixelate_mask(zeros,100)
    map2=pixelate_mask(zeros,100)
    cv2.imwrite(opt.mask_save_path, map2)
    # 二值化
    map2=binarize_image_to_array(map2,[(236, 24, 0), (255, 56, 132),(199, 55, 255)], [(56, 56, 255), (151, 157, 255)],tolerance=30)
    
    # print(map2)
    visualize_binary_array(map2,"/app/my_project/v8seg-photo/mask2.jpg", scale=1)

    # Step 1: 转换为 numpy 数组，并指定数据类型为 uint8
    # map3 = [1, 5, 9]
    map_array = np.array(map2, dtype=np.uint8)

    # 坐标数据
    coords = np.array([int(start_x/zoomm),int(start_y/zoomm), int(fire_center[0]/zoomm), int(fire_center[1]/zoomm),12,9], dtype=np.int32)
    print('coords:',coords)

    data ={
        "width": 133,
        "height": 100,
        "start_point": {
            "x": int(start_x/zoomm),
            "y": int(start_y/zoomm)
        },
        "end_point": {
            "x": int(fire_center[0]/zoomm),
            "y": int(fire_center[1]/zoomm)
        },
        "car_radius": 7,
        "fire_radius": 8,
    }
    json_str = json.dumps(data).encode('utf-8')

    # Step 2: 写入共享内存
    shm_map.seek(0)
    shm_map.write(map_array.tobytes())
    shm_data.seek(0)
    shm_data.write(json_str)

    # 通知 C++ 开始处理
    cpp_process = subprocess.Popen([CPP_PROGRAM])
    cpp_process.wait()
    os.write(efd, struct.pack('Q', 1))  # 发送 64-bit 整数 1
    
    #读取验证
    shm_data.seek(0)
    read_bytes = shm_data.read(SHM_DATA_SIZE)
    print(read_bytes)
    shm_result.seek(0)
    read_paths = shm_result.read(SHM_RESULT_SIZE)
    print("读取的 JSON 数据:", read_paths.decode('utf-8'))

    data_list = json.loads(read_paths.strip(b'\x00'))
    
    all_data=[]
    print(data_list)

    print('car_center',int(start_x/zoomm),int(start_y/zoomm))
    # result = search_path(map2,(int(start_x/zoomm),int(start_y/zoomm)),(int(fire_center[0]/zoomm), int(fire_center[1]/zoomm)),12,9)

    # print(result.angle,result.distance)
    end_y=0
    end_x=0
    
    cv2.circle(zeros, (int(start_x), int(start_y)), 15, (0, 255, 0), -1)
    print(start_x,start_y)

    for json_obj in data_list:
        rdistance=abs(json_obj["Distance"])
        rangle=360-abs(json_obj["Direction"])
        if rangle==360:
            rangle=0
        print(int(rangle),int(rdistance))

        target_angle=abs(int(angle_deg-rangle))
        if target_angle==360:
            target_angle=0
        # 比例！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！    
        distance=int(rdistance*23.5)#16.5,26.15,23.5
        #如果车的角度比要前进的角度大，则先负转弯，再前进
        if angle_deg > rangle:
            # sending_data(0x54,2,target_angle/255,target_angle%255,0x45)
            # sending_data(0x54,0,distance/255,distance%255,0x45)
            packet1 = bytes([0x54, 1, int(target_angle/255.0) % 255, target_angle%255, 0x45])
            packet2 = bytes([0x54, 0, int(distance/255) % 255, distance%255, 0x45])
            all_data.append(packet1)
            all_data.append(packet2)
            angle_deg=angle_deg-target_angle
            if angle_deg < 0:
                angle_deg += 360
        #如果车的角度比要前进的角度小，则先正转弯，再前进
        else :
            # sending_data(0x54,1,target_angle/255,target_angle%255,0x45)
            # sending_data(0x54,0,distance/255,distance%255,0x45)
            packet1 = bytes([0x54, 2, int(target_angle/255.0) % 255, target_angle%255, 0x45])
            packet2 = bytes([0x54, 0, int(distance/255) % 255, distance%255, 0x45])
            all_data.append(packet1) 
            all_data.append(packet2)
            angle_deg=angle_deg+target_angle
            if angle_deg >= 360:
                angle_deg -= 360

        angle_rad = math.radians(angle_deg)
        # 比例！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！    
        end_x = start_x + distance * math.cos(angle_rad)/23.5*zoomm#前是实际与低像素图比例，后是图片像素比
        end_y = start_y - distance * math.sin(angle_rad)/23.5*zoomm
        #画路径
        cv2.line(zeros, (int(start_x), int(start_y)), (int(end_x), int(end_y)), (0, 255, 0), 2)
        start_x, start_y = end_x, end_y

    # 通知 C++ 开始处理
    os.write(efd, struct.pack('Q', 1))  # 发送 64-bit 整数 1
    # os.close(efd)
    

    

    # for r in result:
    #     if r.angle==360:
    #         r.angle=0
    #     print(r.angle,r.distance)

    #     target_angle=abs(int(angle_deg-r.angle))
    #     if target_angle==360:
    #         target_angle=0
    #     # 比例！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！    
    #     distance=int(r.distance*23.5)#16.5,26.15,
    #     #如果车的角度比要前进的角度大，则先负转弯，再前进
    #     if angle_deg > r.angle:
    #         # sending_data(0x54,2,target_angle/255,target_angle%255,0x45) 
    #         # sending_data(0x54,0,distance/255,distance%255,0x45)
    #         packet1 = bytes([0x54, 1, int(target_angle/255.0) % 255, target_angle%255, 0x45])
    #         packet2 = bytes([0x54, 0, int(distance/255) % 255, distance%255, 0x45])
    #         all_data.append(packet1)
    #         all_data.append(packet2)
    #         angle_deg=angle_deg-target_angle
    #         if angle_deg < 0:
    #             angle_deg += 360
    #     #如果车的角度比要前进的角度小，则先正转弯，再前进    
    #     else :
    #         # sending_data(0x54,1,target_angle/255,target_angle%255,0x45)
    #         # sending_data(0x54,0,distance/255,distance%255,0x45)
    #         packet1 = bytes([0x54, 2, int(target_angle/255.0) % 255, target_angle%255, 0x45])
    #         packet2 = bytes([0x54, 0, int(distance/255) % 255, distance%255, 0x45])
    #         all_data.append(packet1) 
    #         all_data.append(packet2)
    #         angle_deg=angle_deg+target_angle
    #         if angle_deg >= 360:
    #             angle_deg -= 360

    #     angle_rad = math.radians(angle_deg)
    #     # 比例！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！    
    #     end_x = start_x + distance * math.cos(angle_rad)/23.5*zoomm#前是实际与低像素图比例，后是图片像素比
    #     end_y = start_y - distance * math.sin(angle_rad)/23.5*zoomm
    #     #画路径
    #     cv2.line(zeros, (int(start_x), int(start_y)), (int(end_x), int(end_y)), (0, 255, 0), 2)
    #     start_x, start_y = end_x, end_y
    packet3 = bytes([0x54, 2, 0x00, 0x64, 0x45])
    all_data.append(packet3)
    ser.write(b''.join(all_data))

    print(all_data)
    #画终点
    cv2.circle(zeros, (int(start_x), int(start_y)), 15, (255, 0, 0), -1)
    #画火焰点
    cv2.circle(zeros, (int(fire_center[0]), int(fire_center[1])), 15, (0, 0, 255), -1)
    t1 = time()
    print("forward time is :", (t1 - t0))
    
    cv2.imwrite(opt.img_save_path, np.hstack((draw_img, zeros, add_result)))
    # 单独保存掩码图
    # cv2.imwrite(opt.mask_save_path, zeros)
    
    logger.info("\033[1;32m" + f"saved combined result in path: \"./{opt.img_save_path}\"" + "\033[0m")
    logger.info("\033[1;32m" + f"saved mask result in path: \"./{opt.mask_save_path}\"" + "\033[0m")

    cpp_process.terminate()
    
    os.close(efd)
    # # 在main()函数中，后处理完成后添加以下代码
    # class_mask = generate_class_mask(results, img.shape)

    # # 保存为图像（0-背景，1-第一类，2-第二类，依此类推）
    # # cv2.imwrite("class_mask.png", class_mask)

    # # 保存为numpy数组，便于后续处理
    # np.save("class_mask.npy", class_mask)

    # # 保存为文本文件
    # txt_save_path = "/app/my_project/v8seg-photo/class_mask.txt"  # 修改为你想保存的路径
    # np.savetxt(txt_save_path, class_mask, fmt='%d')  # 使用%d格式保存整数

    # # 打印统计信息
    # print(f"生成的类别掩码形状: {class_mask.shape}")
    # print(f"检测到的类别数量: {len(np.unique(class_mask)) - 1}")  # 减去背景(0)
    # print(f"文本文件已保存至: {txt_save_path}")
        


class YOLO11_Seg():
    def __init__(self, opt):
        # 加载BPU的bin模型, 打印相关参数
        # Load the quantized *.bin model and print its parameters
        try:
            begin_time = time()
            self.quantize_model = dnn.load(opt.model_path)
            logger.debug("\033[1;31m" + "Load D-Robotics Quantize model time = %.2f ms"%(1000*(time() - begin_time)) + "\033[0m")
        except Exception as e:
            logger.error("❌ Failed to load model file: %s"%(opt.model_path))
            logger.error("You can download the model file from the following docs: ./models/download.md") 
            logger.error(e)
            exit(1)

        logger.info("\033[1;32m" + "-> input tensors" + "\033[0m")
        for i, quantize_input in enumerate(self.quantize_model[0].inputs):
            logger.info(f"intput[{i}], name={quantize_input.name}, type={quantize_input.properties.dtype}, shape={quantize_input.properties.shape}")

        logger.info("\033[1;32m" + "-> output tensors" + "\033[0m")
        for i, quantize_input in enumerate(self.quantize_model[0].outputs):
            logger.info(f"output[{i}], name={quantize_input.name}, type={quantize_input.properties.dtype}, shape={quantize_input.properties.shape}")

        # 将反量化系数准备好, 只需要准备一次
        # prepare the quantize scale, just need to generate once
        self.s_bboxes_scale = self.quantize_model[0].outputs[1].properties.scale_data[np.newaxis, :]
        self.m_bboxes_scale = self.quantize_model[0].outputs[4].properties.scale_data[np.newaxis, :]
        self.l_bboxes_scale = self.quantize_model[0].outputs[7].properties.scale_data[np.newaxis, :]
        logger.info(f"{self.s_bboxes_scale.shape=}, {self.m_bboxes_scale.shape=}, {self.l_bboxes_scale.shape=}")

        self.s_mces_scale = self.quantize_model[0].outputs[2].properties.scale_data[np.newaxis, :]
        self.m_mces_scale = self.quantize_model[0].outputs[5].properties.scale_data[np.newaxis, :]
        self.l_mces_scale = self.quantize_model[0].outputs[8].properties.scale_data[np.newaxis, :]
        logger.info(f"{self.s_mces_scale.shape=}, {self.m_mces_scale.shape=}, {self.l_mces_scale.shape=}")

        self.mask_scale = self.quantize_model[0].outputs[9].properties.scale_data[0]
        logger.info(f"{self.mask_scale = }")


        # DFL求期望的系数, 只需要生成一次
        # DFL calculates the expected coefficients, which only needs to be generated once.
        self.weights_static = np.array([i for i in range(16)]).astype(np.float32)[np.newaxis, np.newaxis, :]
        logger.info(f"{self.weights_static.shape = }")

        # anchors, 只需要生成一次
        # self.s_anchor = np.stack([np.tile(np.linspace(0.5, 79.5, 80), reps=80), 
        #                     np.repeat(np.arange(0.5, 80.5, 1), 80)], axis=0).transpose(1,0)
        # self.m_anchor = np.stack([np.tile(np.linspace(0.5, 39.5, 40), reps=40), 
        #                     np.repeat(np.arange(0.5, 40.5, 1), 40)], axis=0).transpose(1,0)
        # self.l_anchor = np.stack([np.tile(np.linspace(0.5, 19.5, 20), reps=20), 
        #                     np.repeat(np.arange(0.5, 20.5, 1), 20)], axis=0).transpose(1,0)

        # 小目标检测层（s_anchor）：原尺寸 80x80 -> 新尺寸 96x72
        self.s_anchor = np.stack([np.tile(np.linspace(0.5, 95.5, 96), reps=72),
                          np.repeat(np.arange(0.5, 72.5, 1), 96)], axis=0).transpose(1, 0)
        # 中目标检测层（m_anchor）：原尺寸 40x40 -> 新尺寸 48x36
        self.m_anchor = np.stack([np.tile(np.linspace(0.5, 47.5, 48), reps=36),
                          np.repeat(np.arange(0.5, 36.5, 1), 48)], axis=0).transpose(1, 0)
        # 大目标检测层（l_anchor）：原尺寸 20x20 -> 新尺寸 24x18
        self.l_anchor = np.stack([np.tile(np.linspace(0.5, 23.5, 24), reps=18),
                          np.repeat(np.arange(0.5, 18.5, 1), 24)], axis=0).transpose(1, 0)

        logger.info(f"{self.s_anchor.shape = }, {self.m_anchor.shape = }, {self.l_anchor.shape = }")

        # 输入图像大小, 一些阈值, 提前计算好
        self.input_image_size = (576, 768)
        self.SCORE_THRESHOLD = opt.score_thres
        self.NMS_THRESHOLD = opt.nms_thres
        self.CONF_THRES_RAW = -np.log(1/self.SCORE_THRESHOLD - 1)
        logger.info("SCORE_THRESHOLD  = %.2f, NMS_THRESHOLD = %.2f"%(self.SCORE_THRESHOLD, self.NMS_THRESHOLD))
        logger.info("CONF_THRES_RAW = %.2f"%self.CONF_THRES_RAW)

        self.input_H, self.input_W = self.quantize_model[0].inputs[0].properties.shape[2:4]
        logger.info(f"{self.input_H = }, {self.input_W = }")

        self.Mask_H, self.Mask_W = 144, 192
        self.x_scale_corp = self.Mask_W / self.input_W
        self.y_scale_corp = self.Mask_H / self.input_H
        logger.info(f"{self.Mask_H = }   {self.Mask_W = }")
        logger.info(f"{self.x_scale_corp = }, {self.y_scale_corp = }")

        self.REG = opt.reg
        logger.info(f"{self.REG = }")

        self.CLASSES_NUM = opt.classes_num
        logger.info(f"{self.CLASSES_NUM = }")

        self.MCES_NUM = opt.mc
        logger.info(f"{self.MCES_NUM = }")

        self.IS_OPEN = opt.is_open  # 是否对Mask进行形态学开运算
        self.kernel_for_morphologyEx = np.ones((5,5), np.uint8) 
        logger.info(f"{self.IS_OPEN = }   {self.kernel_for_morphologyEx = }")

        self.IS_POINT = opt.is_point # 是否绘制边缘点
        logger.info(f"{self.IS_POINT = }")

    def preprocess_yuv420sp(self, img):
        RESIZE_TYPE = 0
        LETTERBOX_TYPE = 1
        PREPROCESS_TYPE = LETTERBOX_TYPE
        logger.info(f"PREPROCESS_TYPE = {PREPROCESS_TYPE}")

        begin_time = time()
        self.img_h, self.img_w = img.shape[0:2]
        if PREPROCESS_TYPE == RESIZE_TYPE:
            # 利用resize的方式进行前处理, 准备nv12的输入数据
            begin_time = time()
            input_tensor = cv2.resize(img, (self.input_W, self.input_H), interpolation=cv2.INTER_NEAREST) # 利用resize重新开辟内存节约一次
            input_tensor = self.bgr2nv12(input_tensor)
            self.y_scale = 1.0 * self.input_H / self.img_h
            self.x_scale = 1.0 * self.input_W / self.img_w
            self.y_shift = 0
            self.x_shift = 0
            logger.info("\033[1;31m" + f"pre process(resize) time = {1000*(time() - begin_time):.2f} ms" + "\033[0m")
        elif PREPROCESS_TYPE == LETTERBOX_TYPE:
            # 利用 letter box 的方式进行前处理, 准备nv12的输入数据
            begin_time = time()
            self.x_scale = min(1.0 * self.input_H / self.img_h, 1.0 * self.input_W / self.img_w)
            self.y_scale = self.x_scale
            
            if self.x_scale <= 0 or self.y_scale <= 0:
                raise ValueError("Invalid scale factor.")
            
            new_w = int(self.img_w * self.x_scale)
            self.x_shift = (self.input_W - new_w) // 2
            x_other = self.input_W - new_w - self.x_shift
            
            new_h = int(self.img_h * self.y_scale)
            self.y_shift = (self.input_H - new_h) // 2
            y_other = self.input_H - new_h - self.y_shift
            
            input_tensor = cv2.resize(img, (new_w, new_h))
            input_tensor = cv2.copyMakeBorder(input_tensor, self.y_shift, y_other, self.x_shift, x_other, cv2.BORDER_CONSTANT, value=[127, 127, 127])
            input_tensor = self.bgr2nv12(input_tensor)
            logger.info("\033[1;31m" + f"pre process(letter box) time = {1000*(time() - begin_time):.2f} ms" + "\033[0m")
        else:
            logger.error(f"illegal PREPROCESS_TYPE = {PREPROCESS_TYPE}")
            exit(-1)

        logger.debug("\033[1;31m" + f"pre process time = {1000*(time() - begin_time):.2f} ms" + "\033[0m")
        logger.info(f"y_scale = {self.y_scale:.2f}, x_scale = {self.x_scale:.2f}")
        logger.info(f"y_shift = {self.y_shift:.2f}, x_shift = {self.x_shift:.2f}")
        return input_tensor

    def bgr2nv12(self, bgr_img):
        begin_time = time()
        height, width = bgr_img.shape[0], bgr_img.shape[1]
        area = height * width
        yuv420p = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YUV_I420).reshape((area * 3 // 2,))
        y = yuv420p[:area]
        uv_planar = yuv420p[area:].reshape((2, area // 4))
        uv_packed = uv_planar.transpose((1, 0)).reshape((area // 2,))
        nv12 = np.zeros_like(yuv420p)
        nv12[:height * width] = y
        nv12[height * width:] = uv_packed
        logger.debug("\033[1;31m" + f"bgr8 to nv12 time = {1000*(time() - begin_time):.2f} ms" + "\033[0m")
        return nv12

    def forward(self, input_tensor):
        begin_time = time()
        quantize_outputs = self.quantize_model[0].forward(input_tensor)
        logger.debug("\033[1;31m" + f"forward time = {1000*(time() - begin_time):.2f} ms" + "\033[0m")
        return quantize_outputs

    def c2numpy(self, outputs):
        begin_time = time()
        outputs = [dnnTensor.buffer for dnnTensor in outputs]
        logger.debug("\033[1;31m" + f"c to numpy time = {1000*(time() - begin_time):.2f} ms" + "\033[0m")
        return outputs

    def postProcess(self, outputs):
        begin_time = time()
        # reshape
        s_clses = outputs[0].reshape(-1, self.CLASSES_NUM)
        s_bboxes = outputs[1].reshape(-1, self.REG * 4)
        s_mces = outputs[2].reshape(-1, self.MCES_NUM)

        m_clses = outputs[3].reshape(-1, self.CLASSES_NUM)
        m_bboxes = outputs[4].reshape(-1, self.REG * 4)
        m_mces = outputs[5].reshape(-1, self.MCES_NUM)

        l_clses = outputs[6].reshape(-1, self.CLASSES_NUM)
        l_bboxes = outputs[7].reshape(-1, self.REG * 4)
        l_mces = outputs[8].reshape(-1, self.MCES_NUM)

        protos = outputs[9]


        # classify: 利用numpy向量化操作完成阈值筛选(优化版 2.0)
        s_max_scores = np.max(s_clses, axis=1)
        s_valid_indices = np.flatnonzero(s_max_scores >= self.CONF_THRES_RAW)  # 得到大于阈值分数的索引，此时为小数字
        s_ids = np.argmax(s_clses[s_valid_indices, : ], axis=1)
        s_scores = s_max_scores[s_valid_indices]

        m_max_scores = np.max(m_clses, axis=1)
        m_valid_indices = np.flatnonzero(m_max_scores >= self.CONF_THRES_RAW)  # 得到大于阈值分数的索引，此时为小数字
        m_ids = np.argmax(m_clses[m_valid_indices, : ], axis=1)
        m_scores = m_max_scores[m_valid_indices]

        l_max_scores = np.max(l_clses, axis=1)
        l_valid_indices = np.flatnonzero(l_max_scores >= self.CONF_THRES_RAW)  # 得到大于阈值分数的索引，此时为小数字
        l_ids = np.argmax(l_clses[l_valid_indices, : ], axis=1)
        l_scores = l_max_scores[l_valid_indices]

        # 3个Classify分类分支：Sigmoid计算
        s_scores = 1 / (1 + np.exp(-s_scores))
        m_scores = 1 / (1 + np.exp(-m_scores))
        l_scores = 1 / (1 + np.exp(-l_scores))

        # 3个Bounding Box分支：反量化
        s_bboxes_float32 = s_bboxes[s_valid_indices,:].astype(np.float32) * self.s_bboxes_scale
        m_bboxes_float32 = m_bboxes[m_valid_indices,:].astype(np.float32) * self.m_bboxes_scale
        l_bboxes_float32 = l_bboxes[l_valid_indices,:].astype(np.float32) * self.l_bboxes_scale

        # 3个Bounding Box分支：dist2bbox (ltrb2xyxy)
        s_ltrb_indices = np.sum(softmax(s_bboxes_float32.reshape(-1, 4, 16), axis=2) * self.weights_static, axis=2)
        s_anchor_indices = self.s_anchor[s_valid_indices, :]
        s_x1y1 = s_anchor_indices - s_ltrb_indices[:, 0:2]
        s_x2y2 = s_anchor_indices + s_ltrb_indices[:, 2:4]
        s_dbboxes = np.hstack([s_x1y1, s_x2y2])*8

        m_ltrb_indices = np.sum(softmax(m_bboxes_float32.reshape(-1, 4, 16), axis=2) * self.weights_static, axis=2)
        m_anchor_indices = self.m_anchor[m_valid_indices, :]
        m_x1y1 = m_anchor_indices - m_ltrb_indices[:, 0:2]
        m_x2y2 = m_anchor_indices + m_ltrb_indices[:, 2:4]
        m_dbboxes = np.hstack([m_x1y1, m_x2y2])*16

        l_ltrb_indices = np.sum(softmax(l_bboxes_float32.reshape(-1, 4, 16), axis=2) * self.weights_static, axis=2)
        l_anchor_indices = self.l_anchor[l_valid_indices,:]
        l_x1y1 = l_anchor_indices - l_ltrb_indices[:, 0:2]
        l_x2y2 = l_anchor_indices + l_ltrb_indices[:, 2:4]
        l_dbboxes = np.hstack([l_x1y1, l_x2y2])*32

        # 三个Mask Coefficients分支的反量化
        s_mces_float32 = (s_mces[s_valid_indices,:].astype(np.float32) * self.s_mces_scale)
        m_mces_float32 = (m_mces[m_valid_indices,:].astype(np.float32) * self.m_mces_scale)
        l_mces_float32 = (l_mces[l_valid_indices,:].astype(np.float32) * self.l_mces_scale)

        # Mask Proto的反量化
        protos_float32 = protos.astype(np.float32)[0] * self.mask_scale
        # print("protos.shape:", protos.shape)
        # print("protos_float32.shape:", protos_float32.shape)

        # 大中小特征层阈值筛选结果拼接
        dbboxes = np.concatenate((s_dbboxes, m_dbboxes, l_dbboxes), axis=0)
        scores = np.concatenate((s_scores, m_scores, l_scores), axis=0)
        ids = np.concatenate((s_ids, m_ids, l_ids), axis=0)
        mces = np.concatenate((s_mces_float32, m_mces_float32, l_mces_float32), axis=0)

        # xyxy 2 xyhw
        xy = (dbboxes[:,2:4] + dbboxes[:,0:2])/2.0
        hw = (dbboxes[:,2:4] - dbboxes[:,0:2])
        xyhw = np.hstack([xy, hw])

        # 分类别nms
        results = []
        for i in range(self.CLASSES_NUM):
            id_indices = ids==i
            indices = cv2.dnn.NMSBoxes(xyhw[id_indices,:], scores[id_indices], self.SCORE_THRESHOLD, self.NMS_THRESHOLD)
            if len(indices) == 0:
                continue
            for indic in indices:
                x1, y1, x2, y2 = dbboxes[id_indices,:][indic]
                # mask
                x1_corp = int(x1 * self.x_scale_corp)
                y1_corp = int(y1 * self.y_scale_corp)
                x2_corp = int(x2 * self.x_scale_corp)
                y2_corp = int(y2 * self.y_scale_corp)
                # bbox
                x1 = int((x1 - self.x_shift) / self.x_scale)
                y1 = int((y1 - self.y_shift) / self.y_scale)
                x2 = int((x2 - self.x_shift) / self.x_scale)
                y2 = int((y2 - self.y_shift) / self.y_scale)    
                # clip
                x1 = x1 if x1 > 0 else 0
                x2 = x2 if x2 > 0 else 0
                y1 = y1 if y1 > 0 else 0
                y2 = y2 if y2 > 0 else 0
                x1 = x1 if x1 < self.img_w else self.img_w
                x2 = x2 if x2 < self.img_w else self.img_w
                y1 = y1 if y1 < self.img_h else self.img_h
                y2 = y2 if y2 < self.img_h else self.img_h       
                # mask
                mc = mces[id_indices][indic]
                mask = (np.sum(mc[np.newaxis, np.newaxis, :]*protos_float32[y1_corp:y2_corp,x1_corp:x2_corp,:], axis=2) > 0.5).astype(np.uint8)
                # print(f"mces.shape={mces.shape}, mc.shape={mc.shape}")
                # print(f"protos_crop.shape={protos_float32[y1_corp:y2_corp,x1_corp:x2_corp,:].shape}")
                # append
                results.append((i, scores[id_indices][indic], x1, y1, x2, y2, mask))

        logger.debug("\033[1;31m" + f"Post Process time = {1000*(time() - begin_time):.2f} ms" + "\033[0m")

        return results

coco_names1 = ["background","person","knife","fork","cup","giraffe","plate","table","cake","fence","hat","terrain","tree","car"]
coco_names= ["background","terrain","tree","car-tou","car-wei"]

rdk_colors = [
    (56, 56, 255), (151, 157, 255), (236, 24, 0), (255, 56, 132),(49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
    (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0),(147, 69, 52), (255, 115, 100), (236, 24, 0), (255, 56, 132),
    (133, 0, 82), (255, 56, 203), (200, 149, 255), (199, 55, 255)]

def draw_detection(img, bbox, score, class_id) -> None:
    """
    Draws a detection bounding box and label on the image.

    Parameters:
        img (np.array): The input image.
        bbox (tuple[int, int, int, int]): A tuple containing the bounding box coordinates (x1, y1, x2, y2).
        score (float): The detection score of the object.
        class_id (int): The class ID of the detected object.
    """
    x1, y1, x2, y2 = bbox
    color = rdk_colors[class_id%20]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    label = f"{coco_names[class_id]}: {score:.2f}"
    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    label_x, label_y = x1, y1 - 10 if y1 - 10 > label_height else y1 + 10
    cv2.rectangle(
        img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
    )
    cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

if __name__ == "__main__":
    main()