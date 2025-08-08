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

import cv2
import numpy as np
from scipy.special import softmax
from hobot_dnn import pyeasy_dnn as dnn  # BSP Python API
import sys
from time import sleep, time
import struct
import argparse
import logging
import ctypes
import os
import serial
# 日志模块配置
logging.basicConfig(
    level = logging.DEBUG,
    format = '[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S')
logger = logging.getLogger("RDK_YOLO")

ser = serial.Serial("/dev/ttyS1", 115200, timeout=1)

print(sys.executable)

def signal_handler(signal, frame):
    print("\nExiting program")
    sys.exit(0)

def limit_display_cord(coor):
    coor[0] = max(min(1920, coor[0]), 0)
    coor[1] = max(min(1080, coor[1]), 2)
    coor[2] = max(min(1920, coor[2]), 0)
    coor[3] = max(min(1080, coor[3]), 0)
    return coor

libpostprocess = ctypes.CDLL('/usr/lib/libpostprocess.so')

def get_display_res():
    if os.path.exists("/usr/bin/get_hdmi_res") == False:
        return 1920, 1080

    import subprocess
    p = subprocess.Popen(["/usr/bin/get_hdmi_res"], stdout=subprocess.PIPE)
    result = p.communicate()
    res = result[0].split(b',')
    res[1] = max(min(int(res[1]), 1920), 0)
    res[0] = max(min(int(res[0]), 1080), 0)
    return int(res[1]), int(res[0])

def is_usb_camera(device):
    try:
        cap = cv2.VideoCapture(device)
        if not cap.isOpened():
            return False
        cap.release()
        return True
    except Exception:
        return False

def find_first_usb_camera():
    video_devices = [os.path.join('/dev', dev) for dev in os.listdir('/dev') if dev.startswith('video')]
    for dev in video_devices:
        if is_usb_camera(dev):
            return dev
    return None

def print_properties(pro):
    print("tensor type:", pro.tensor_type)
    print("data type:", pro.dtype)
    print("layout:", pro.layout)
    print("shape:", pro.shape)


# 基于IOU的轻量级跟踪器（无外部依赖）
class IOUTracker:
    def __init__(self, max_age=20, iou_threshold=0.2):
        self.max_age = max_age  # 目标消失后保留ID的最大帧数
        self.iou_threshold = iou_threshold  # 匹配IOU阈值
        self.tracks = []  # 跟踪列表，每个元素: (id, bbox, class_id, age)
        self.next_id = 1  # 下一个可用ID
        self.track_history = {}  # 记录目标历史位置 {id: [(x_center, y_center, timestamp), ...]}

    def update(self, detections):
        """
        更新跟踪器
        detections格式: [(class_id, score, x1, y1, x2, y2), ...]
        返回格式: [(track_id, class_id, score, x1, y1, x2, y2), ...]
        """
        # 1. 对现有跟踪目标的age+1（标记存活状态）
        for track in self.tracks:
            track['age'] += 1

        # 2. 计算当前检测与现有跟踪的IOU矩阵
        iou_matrix = []
        for det in detections:
            _, _, x1, y1, x2, y2 = det
            det_bbox = [x1, y1, x2, y2]
            ious = []
            for track in self.tracks:
                track_bbox = track['bbox']
                iou = self.calculate_iou(det_bbox, track_bbox)
                ious.append(iou)
            iou_matrix.append(ious)

        # 3. 基于IOU匹配检测与跟踪目标
        matched_indices = self.match_detections_to_tracks(iou_matrix)

        # 4. 更新匹配上的跟踪目标
        matched_tracks = []
        for det_idx, track_idx in matched_indices:
            self.tracks[track_idx]['bbox'] = detections[det_idx][2:6]  # 更新边界框
            self.tracks[track_idx]['class_id'] = detections[det_idx][0]  # 更新类别
            self.tracks[track_idx]['age'] = 0  # 重置age（标记为活跃）
            matched_tracks.append((det_idx, track_idx))

        # 5. 处理未匹配的检测（新目标）
        unmatched_dets = [i for i in range(len(detections)) if i not in [d for d, t in matched_tracks]]
        for det_idx in unmatched_dets:
            class_id, score, x1, y1, x2, y2 = detections[det_idx]
            track_id = self.next_id
            self.tracks.append({
                'id': track_id,
                'bbox': [x1, y1, x2, y2],
                'class_id': class_id,
                'age': 0
            })
            self.track_history[track_id] = []
            self.next_id += 1

        # 6. 过滤过期跟踪目标（age超过max_age）
        self.tracks = [t for t in self.tracks if t['age'] <= self.max_age]
        
        # 7. 更新轨迹历史
        current_time = time()
        for track in self.tracks:
            track_id = track['id']
            x1, y1, x2, y2 = track['bbox']
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            if track_id not in self.track_history:
                self.track_history[track_id] = []
            self.track_history[track_id].append((x_center, y_center, current_time))
            # 限制历史记录长度
            if len(self.track_history[track_id]) > 30:
                self.track_history[track_id].pop(0)

        # 8. 整理输出结果
        output = []
        for track in self.tracks:
            if track['age'] == 0:  # 只返回当前帧活跃的目标
                x1, y1, x2, y2 = track['bbox']
                output.append((
                    track['id'],
                    track['class_id'],
                    0.0,  # 占位，实际分数从detections取
                    x1, y1, x2, y2
                ))

        # 补充分数（根据class_id和bbox匹配）
        for i in range(len(output)):
            track_id, class_id, _, x1, y1, x2, y2 = output[i]
            for det in detections:
                d_class, d_score, d_x1, d_y1, d_x2, d_y2 = det
                if (d_class == class_id and 
                    abs(d_x1 - x1) < 5 and abs(d_y1 - y1) < 5 and
                    abs(d_x2 - x2) < 5 and abs(d_y2 - y2) < 5):
                    output[i] = (track_id, class_id, d_score, x1, y1, x2, y2)
                    break

        return output

    def calculate_iou(self, bbox1, bbox2):
        """计算两个边界框的交并比（IOU）"""
        x1, y1, x2, y2 = bbox1
        x1g, y1g, x2g, y2g = bbox2

        # 计算交集
        x1i = max(x1, x1g)
        y1i = max(y1, y1g)
        x2i = min(x2, x2g)
        y2i = min(y2, y2g)
        area_i = max(0, x2i - x1i) * max(0, y2i - y1i)

        # 计算并集
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x2g - x1g) * (y2g - y1g)
        area_u = area1 + area2 - area_i

        return area_i / area_u if area_u > 0 else 0

    def match_detections_to_tracks(self, iou_matrix):
        """基于IOU矩阵匹配检测与跟踪目标"""
        matched = []
        if not iou_matrix or len(iou_matrix[0]) == 0:
            return matched

        # 按IOU从高到低排序
        for det_idx in range(len(iou_matrix)):
            ious = iou_matrix[det_idx]
            if not ious:
                continue
            track_idx = np.argmax(ious)
            if ious[track_idx] >= self.iou_threshold:
                # 确保重复匹配
                if track_idx not in [t for d, t in matched]:
                    matched.append((det_idx, track_idx))
        return matched


# 计数区域类 - 无任何交互功能，仅通过代码设置
class CountingRegion:
    def __init__(self, x1=294, y1=192, x2=550, y2=384):
        # 默认设置为768x576分辨率下3x3网格的中心区域
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        
    def is_point_inside(self, x, y):
        """检查点是否在区域内"""
        return self.x1 < x < self.x2 and self.y1 < y < self.y2
    
    def set_region(self, x1, y1, x2, y2):
        """直接设置区域坐标"""
        # 确保坐标有效
        self.x1 = max(0, min(x1, x2))
        self.y1 = max(0, min(y1, y2))
        self.x2 = max(0, max(x1, x2))
        self.y2 = max(0, max(y1, y2))
        
        # 确保区域有最小尺寸
        min_size = 50
        if self.x2 - self.x1 < min_size:
            self.x2 = self.x1 + min_size
        if self.y2 - self.y1 < min_size:
            self.y2 = self.y1 + min_size
                
    def draw(self, img):
        """在图像上绘制计数区域"""
        # 绘制区域
        cv2.rectangle(img, (int(self.x1), int(self.y1)), (int(self.x2), int(self.y2)), (0, 255, 0), 2)
        # 绘制区域说明
        cv2.putText(img, "Counting Area", (int(self.x1), int(self.y1) - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def main_map():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='/app/DS/converted_model_modified90.bin', 
                        help="""Path to BPU Quantized *.bin Model.
                                RDK X3(Module): Bernoulli2.
                                RDK Ultra: Bayes.
                                RDK X5(Module): Bayes-e.
                                RDK S100: Nash-e.
                                RDK S100P: Nash-m.""") 
    parser.add_argument('--classes-num', type=int, default=6, help='Classes Num to Detect.')
    parser.add_argument('--nms-thres', type=float, default=0.7, help='IoU threshold.')
    parser.add_argument('--score-thres', type=float, default=0.5, help='confidence threshold.')
    parser.add_argument('--reg', type=int, default=16, help='DFL reg layer.')
    parser.add_argument('--mc', type=int, default=32, help='Mask Coefficients')
    parser.add_argument('--tracker-max-age', type=int, default=13, help='Max frames to keep lost target ID')
    parser.add_argument('--tracker-iou', type=float, default=0.2, help='Tracker IOU threshold')
    
    # 添加计数区域的命令行参数
    parser.add_argument('--region-x1', type=int, default=294, help='Counting region top-left x coordinate')
    parser.add_argument('--region-y1', type=int, default=183, help='Counting region top-left y coordinate')
    parser.add_argument('--region-x2', type=int, default=550, help='Counting region bottom-right x coordinate')
    parser.add_argument('--region-y2', type=int, default=375, help='Counting region bottom-right y coordinate')
    
    opt = parser.parse_args()
    logger.info(opt)

    # 实例化模型
    model = YOLO11_Seg(opt)

    

    # 初始化基于IOU的跟踪器
    tracker = IOUTracker(
        max_age=opt.tracker_max_age,
        iou_threshold=opt.tracker_iou
    )

    # 通过代码初始化计数区域（使用命令行参数）
    counting_region = CountingRegion(
        x1=opt.region_x1,
        y1=opt.region_y1,
        x2=opt.region_x2,
        y2=opt.region_y2
    )

    if len(sys.argv) > 1:
        video_device = sys.argv[1]
    else:
        video_device = find_first_usb_camera()

    if video_device is None:
        print("No USB camera found.")
        sys.exit(-1)

    print(f"Opening video device: {video_device}")
    cap = cv2.VideoCapture(video_device)
    if(not cap.isOpened()):
        exit(-1)
    
    print("Open usb camera successfully")
    # 设置摄像头参数
    codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    cap.set(cv2.CAP_PROP_FOURCC, codec)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)

    # 创建显示窗口
    window_name = "Result with Count"
    cv2.namedWindow(window_name)

    # 计数初始化
    class_counters = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}  # 1-5对应5类目标
    counted_ids = {1: set(), 2: set(), 3: set(), 4: set(), 5: set()}  # 每类已计数的ID集合
    # 记录目标是否在区域内的状态，用于判断进入事件
    in_region_status = {}  # {track_id: bool}

    while True:
        # 读取摄像头图像
        t1= time()
        _, frame = cap.read()
        if frame is None:
            print("Failed to get image from usb camera")
            continue

        des_dim = (768, 576)
        img = cv2.resize(frame, des_dim, interpolation=cv2.INTER_AREA)

        # 模型推理
        input_tensor = model.preprocess_yuv420sp(img)
        outputs = model.c2numpy(model.forward(input_tensor))
        results = model.postProcess(outputs)

        # 准备跟踪器输入：(class_id, score, x1, y1, x2, y2)
        # 只提取检测框信息，忽略mask
        detections = []
        for class_id, score, x1, y1, x2, y2, _ in results:  # 忽略最后一个mask参数
            detections.append((class_id, score, x1, y1, x2, y2))

        # 更新跟踪器，获取带ID的目标
        tracks = tracker.update(detections)

        # 渲染初始化
        logger.info("\033[1;32m" + "Draw Results: " + "\033[0m")
        draw_img = img.copy()

        # 处理跟踪结果并计数
        for track in tracks:
            track_id, class_id, score, x1, y1, x2, y2 = track
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # 计算目标中心点
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # 检查目标是否在计数区域内
            in_region = counting_region.is_point_inside(center_x, center_y)
            
            # 更新目标区域状态
            if track_id not in in_region_status:
                in_region_status[track_id] = False
            
            # 检测进入区域事件（从外部进入内部）
            if in_region and not in_region_status[track_id]:
                # 只对新进入区域的目标计数
                if class_id in counted_ids and track_id not in counted_ids[class_id]:
                    class_counters[class_id] += 1
                    if class_counters[class_id]>=3:
                        class_counters[class_id]=3
                    counted_ids[class_id].add(track_id)  # 记录已计数ID
                    logger.info(f"Counted {coco_names[class_id]} with ID {track_id}")
            
            # 更新状态
            in_region_status[track_id] = in_region

            # ser.write(bytes([84,1,class_counters[1],round(center_x),round(center_y),69]))
            # sleep(0.01)
            # ser.write(bytes([84,2,class_counters[2],round(center_x),round(center_y),69]))
            # sleep(0.01)
            # ser.write(bytes([84,3,class_counters[3],round(center_x),round(center_y),69]))
            # sleep(0.01)
            # ser.write(bytes([84,4,class_counters[4],round(center_x),round(center_y),69]))
            # sleep(0.01)
            # ser.write(bytes([84,5,class_counters[5],round(center_x),round(center_y),69]))

            ser.write(bytes([84,class_counters[1],class_counters[2],class_counters[3],class_counters[4],class_counters[5],69]))
            sleep(0.001)
            # ser.write(bytes([84,2,class_counters[2],69]))
            # sleep(0.001)
            # ser.write(bytes([84,3,class_counters[3],69]))
            # sleep(0.001)
            # ser.write(bytes([84,4,class_counters[4],69]))
            # sleep(0.001)
            # ser.write(bytes([84,5,class_counters[5],69]))


            # 绘制检测框和ID
            print(f"ID:{track_id} ({x1,y1,x2,y2}) -> {coco_names[class_id]}: {score:.2f}")
            draw_detection(draw_img, (x1, y1, x2, y2), score, class_id, track_id, in_region)

        # 绘制计数区域
        counting_region.draw(draw_img)

        # 在图像上显示计数结果
        count_text = "Counts: "
        for cls_id in class_counters:
            count_text += f"{coco_names[cls_id]}={class_counters[cls_id]} "
        cv2.putText(
            draw_img, count_text, (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1
        )

        

        # 显示结果
        # cv2.imshow(window_name, draw_img)
        t2= time()
        logger.info(f"Frame processing time: {1000 * (t2 - t1):.2f} ms")

        # 仅保留退出功能
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # ESC或q退出
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    

class YOLO11_Seg():
    def __init__(self, opt):
        # 加载BPU模型
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

        # 反量化系数
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

        # DFL系数
        self.weights_static = np.array([i for i in range(16)]).astype(np.float32)[np.newaxis, np.newaxis, :]
        logger.info(f"{self.weights_static.shape = }")

        # anchors
        self.s_anchor = np.stack([np.tile(np.linspace(0.5, 95.5, 96), reps=72),
                          np.repeat(np.arange(0.5, 72.5, 1), 96)], axis=0).transpose(1, 0)
        self.m_anchor = np.stack([np.tile(np.linspace(0.5, 47.5, 48), reps=36),
                          np.repeat(np.arange(0.5, 36.5, 1), 48)], axis=0).transpose(1, 0)
        self.l_anchor = np.stack([np.tile(np.linspace(0.5, 23.5, 24), reps=18),
                          np.repeat(np.arange(0.5, 18.5, 1), 24)], axis=0).transpose(1, 0)
        logger.info(f"{self.s_anchor.shape = }, {self.m_anchor.shape = }, {self.l_anchor.shape = }")

        # 输入参数
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

    def preprocess_yuv420sp(self, img):
        RESIZE_TYPE = 0
        LETTERBOX_TYPE = 1
        PREPROCESS_TYPE = LETTERBOX_TYPE
        logger.info(f"PREPROCESS_TYPE = {PREPROCESS_TYPE}")

        begin_time = time()
        self.img_h, self.img_w = img.shape[0:2]
        if PREPROCESS_TYPE == RESIZE_TYPE:
            input_tensor = cv2.resize(img, (self.input_W, self.input_H), interpolation=cv2.INTER_NEAREST)
            input_tensor = self.bgr2nv12(input_tensor)
            self.y_scale = 1.0 * self.input_H / self.img_h
            self.x_scale = 1.0 * self.input_W / self.img_w
            self.y_shift = 0
            self.x_shift = 0
            logger.info("\033[1;31m" + f"pre process(resize) time = {1000*(time() - begin_time):.2f} ms" + "\033[0m")
        elif PREPROCESS_TYPE == LETTERBOX_TYPE:
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

        # classify筛选
        s_max_scores = np.max(s_clses, axis=1)
        s_valid_indices = np.flatnonzero(s_max_scores >= self.CONF_THRES_RAW)
        s_ids = np.argmax(s_clses[s_valid_indices, : ], axis=1)
        s_scores = s_max_scores[s_valid_indices]

        m_max_scores = np.max(m_clses, axis=1)
        m_valid_indices = np.flatnonzero(m_max_scores >= self.CONF_THRES_RAW)
        m_ids = np.argmax(m_clses[m_valid_indices, : ], axis=1)
        m_scores = m_max_scores[m_valid_indices]

        l_max_scores = np.max(l_clses, axis=1)
        l_valid_indices = np.flatnonzero(l_max_scores >= self.CONF_THRES_RAW)
        l_ids = np.argmax(l_clses[l_valid_indices, : ], axis=1)
        l_scores = l_max_scores[l_valid_indices]

        # Sigmoid计算
        s_scores = 1 / (1 + np.exp(-s_scores))
        m_scores = 1 / (1 + np.exp(-m_scores))
        l_scores = 1 / (1 + np.exp(-l_scores))

        # Bounding Box反量化
        s_bboxes_float32 = s_bboxes[s_valid_indices,:].astype(np.float32) * self.s_bboxes_scale
        m_bboxes_float32 = m_bboxes[m_valid_indices,:].astype(np.float32) * self.m_bboxes_scale
        l_bboxes_float32 = l_bboxes[l_valid_indices,:].astype(np.float32) * self.l_bboxes_scale

        # dist2bbox
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

        # Mask Coefficients反量化
        s_mces_float32 = (s_mces[s_valid_indices,:].astype(np.float32) * self.s_mces_scale)
        m_mces_float32 = (m_mces[m_valid_indices,:].astype(np.float32) * self.m_mces_scale)
        l_mces_float32 = (l_mces[l_valid_indices,:].astype(np.float32) * self.l_mces_scale)

        # Mask Proto反量化
        protos_float32 = protos.astype(np.float32)[0] * self.mask_scale

        # 拼接结果
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
                # 坐标转换
                x1 = int((x1 - self.x_shift) / self.x_scale)
                y1 = int((y1 - self.y_shift) / self.y_scale)
                x2 = int((x2 - self.x_shift) / self.x_scale)
                y2 = int((y2 - self.y_shift) / self.y_scale)    
                # 边界裁剪
                x1 = max(0, min(x1, self.img_w))
                x2 = max(0, min(x2, self.img_w))
                y1 = max(0, min(y1, self.img_h))
                y2 = max(0, min(y2, self.img_h))       
                # 返回空掩码
                results.append((i, scores[id_indices][indic], x1, y1, x2, y2, np.array([])))

        logger.debug("\033[1;31m" + f"Post Process time = {1000*(time() - begin_time):.2f} ms" + "\033[0m")
        return results

# 类别名称（1-5对应5类目标）
coco_names = ["background", "elephant", "peacock", "monkey", "tiger", "wolf"]
# 颜色配置
rdk_colors = [(56, 56, 255), (10, 249, 72), (255, 194, 0), (236, 24, 0), (199, 55, 255), (147, 69, 52)]

# 绘制函数，显示边界框、ID和区域内状态
def draw_detection(img, bbox, score, class_id, track_id, in_region) -> None:
    x1, y1, x2, y2 = bbox
    color = rdk_colors[class_id%20]
    # 如果在计数区域内，使用不同颜色
    if in_region:
        color = (0, 0, 255)  # 红色表示在区域内
    # 绘制边界框
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    # 标签添加ID信息
    label = f"ID:{track_id} {coco_names[class_id]}: {score:.2f}"
    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    label_x, label_y = x1, y1 - 10 if y1 - 10 > label_height else y1 + 10
    # 绘制标签背景
    cv2.rectangle(
        img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height),
        color, cv2.FILLED
    )
    # 绘制标签文本
    cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

if __name__ == "__main__":
    main_map()
