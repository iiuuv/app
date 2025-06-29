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
# from scipy.special import expit as sigmoid
from hobot_dnn import pyeasy_dnn as dnn  # BSP Python API
from pathfinder import search_path
from time import time
import argparse
import logging

# 日志模块配置
# logging configs
logging.basicConfig(
    level = logging.DEBUG,
    format = '[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S')
logger = logging.getLogger("RDK_YOLO")

def pixelate_mask(mask, target_size=100, quality=2.0):
    """
    对掩码进行低像素化处理并直接输出小尺寸图片
    
    参数:
    mask: 原始掩码图 (numpy数组)
    target_size: 目标尺寸的最小边长 (像素数)
    quality: 质量因子，控制插值和滤波强度
    
    返回:
    低像素化后的小尺寸掩码图 (numpy数组)
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
    
    # 计算降采样比例
    scale = min(target_size / h, target_size / w)
    target_h, target_w = int(h * scale), int(w * scale)
    
    # 如果目标尺寸已大于原图，则不处理
    if target_h >= h and target_w >= w:
        return mask.astype(np.uint8)
    
    # 降采样（保持宽高比）
    if is_gray:
        downsampled = cv2.resize(mask, (target_w, target_h), 
                                interpolation=cv2.INTER_AREA)
    else:
        downsampled = cv2.resize(mask, (target_w, target_h), 
                                interpolation=cv2.INTER_AREA)
    
    # 计算高斯核大小（确保为正奇数）
    kernel_size = max(1, int(3 * quality))
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # 应用抗锯齿滤波
    if is_gray:
        blurred = cv2.GaussianBlur(downsampled, (kernel_size, kernel_size), quality/2)
    else:
        blurred = cv2.GaussianBlur(downsampled, (kernel_size, kernel_size), quality/2)
    
    # 二值化处理（仅对单通道掩码）
    if is_gray:
        _, pixelated = cv2.threshold(blurred, 0.5, 255, cv2.THRESH_BINARY)
        return pixelated.astype(np.uint8)
    else:
        return blurred.astype(np.uint8)

def binarize_image_to_array(image, target_color, tolerance=30):
    """
    将图片二值化并转换为二维数组
    
    参数:
    image (str/numpy.ndarray): 图片路径或图片数组
    target_color (tuple): 目标颜色，格式为(B,G,R)
    tolerance (int): 颜色匹配容忍度，值越大匹配范围越广
    
    返回:
    numpy.ndarray: 二值化后的二维数组，黑色为0，白色为1
    """
    # 判断输入是路径还是数组
    if isinstance(image, str):
        # 如果是路径，则读取图片
        img = cv2.imread(image)
        if img is None:
            raise ValueError(f"无法读取图片: {image}")
    else:
        # 如果是数组，直接使用
        img = image
    
    # 创建二值化掩码
    # 黑色区域
    black_mask = cv2.inRange(img, (0, 0, 0), (50, 50, 50))
    
    # 目标颜色区域
    lower_bound = np.array([max(0, c - tolerance) for c in target_color])
    upper_bound = np.array([min(255, c + tolerance) for c in target_color])
    color_mask = cv2.inRange(img, lower_bound, upper_bound)
    
    # 合并掩码（黑色和目标颜色都变为黑色(0)）
    combined_mask = cv2.bitwise_or(black_mask, color_mask)
    
    # 将合并后的掩码转换为0和1的二维数组
    # 黑色区域(0)保持为0，其他区域(255)变为1
    binary_array = (combined_mask == 0).tolist()
    
    return binary_array

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='/app/my_project/bin/converted_model_modified3.bin', 
                        help="""Path to BPU Quantized *.bin Model.
                                RDK X3(Module): Bernoulli2.
                                RDK Ultra: Bayes.
                                RDK X5(Module): Bayes-e.
                                RDK S100: Nash-e.
                                RDK S100P: Nash-m.""") 
    # parser.add_argument('--test-img', type=str, default='../../../../resource/datasets/COCO2017/assets/bus.jpg', help='Path to Load Test Image.')
    parser.add_argument('--test-img', type=str, default='/app/my_project/v8seg-photo/微信图片_20250615124202.jpg', help='Path to Load Test Image.')
    parser.add_argument('--mask-save-path', type=str, default='/app/my_project/v8seg-photo/mask.jpg', help='Path to Save Mask Image.')
    parser.add_argument('--img-save-path', type=str, default='/app/my_project/v8seg-photo/result.jpg', help='Path to Load Test Image.')
    parser.add_argument('--classes-num', type=int, default=14, help='Classes Num to Detect.')
    parser.add_argument('--nms-thres', type=float, default=0.7, help='IoU threshold.')
    parser.add_argument('--score-thres', type=float, default=0.25, help='confidence threshold.')
    parser.add_argument('--reg', type=int, default=16, help='DFL reg layer.')
    parser.add_argument('--mc', type=int, default=32, help='Mask Coefficients')
    parser.add_argument('--is-open', type=bool, default=True, help='Ture: morphologyEx')
    parser.add_argument('--is-point', type=bool, default=True, help='Ture: Draw edge points')
    opt = parser.parse_args()
    logger.info(opt)
    
    # 实例化
    model = YOLO11_Seg(opt)
    # 读图
    img = cv2.imread(opt.test_img)
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
    for class_id, score, x1, y1, x2, y2, mask in results:
        # Detect
        print("(%d, %d, %d, %d) -> %s: %.2f"%(x1,y1,x2,y2, coco_names[class_id], score))
        draw_detection(draw_img, (x1, y1, x2, y2), score, class_id)
        # Instance Segment
        if mask.size == 0:
            continue
        mask = cv2.resize(mask, (int(x2-x1), int(y2-y1)), interpolation=cv2.INTER_LANCZOS4)
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
    # 保存结果

    map2=pixelate_mask(zeros,150)
    cv2.imwrite(opt.mask_save_path, map2)

    map2=binarize_image_to_array(map2,(199, 55, 255))
    # print(map1)
    result = search_path(map2,(8, 8),(142, 142),5,5)
    print(result)
    cv2.imwrite(opt.img_save_path, np.hstack((draw_img, zeros, add_result)))
    # 单独保存掩码图
    
    logger.info("\033[1;32m" + f"saved combined result in path: \"./{opt.img_save_path}\"" + "\033[0m")
    logger.info("\033[1;32m" + f"saved mask result in path: \"./{opt.mask_save_path}\"" + "\033[0m")
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
        self.s_anchor = np.stack([np.tile(np.linspace(0.5, 79.5, 80), reps=80), 
                            np.repeat(np.arange(0.5, 80.5, 1), 80)], axis=0).transpose(1,0)
        self.m_anchor = np.stack([np.tile(np.linspace(0.5, 39.5, 40), reps=40), 
                            np.repeat(np.arange(0.5, 40.5, 1), 40)], axis=0).transpose(1,0)
        self.l_anchor = np.stack([np.tile(np.linspace(0.5, 19.5, 20), reps=20), 
                            np.repeat(np.arange(0.5, 20.5, 1), 20)], axis=0).transpose(1,0)
        logger.info(f"{self.s_anchor.shape = }, {self.m_anchor.shape = }, {self.l_anchor.shape = }")

        # 输入图像大小, 一些阈值, 提前计算好
        self.input_image_size = 640
        self.SCORE_THRESHOLD = opt.score_thres
        self.NMS_THRESHOLD = opt.nms_thres
        self.CONF_THRES_RAW = -np.log(1/self.SCORE_THRESHOLD - 1)
        logger.info("SCORE_THRESHOLD  = %.2f, NMS_THRESHOLD = %.2f"%(self.SCORE_THRESHOLD, self.NMS_THRESHOLD))
        logger.info("CONF_THRES_RAW = %.2f"%self.CONF_THRES_RAW)

        self.input_H, self.input_W = self.quantize_model[0].inputs[0].properties.shape[2:4]
        logger.info(f"{self.input_H = }, {self.input_W = }")

        self.Mask_H, self.Mask_W = 160, 160
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
                # append
                results.append((i, scores[id_indices][indic], x1, y1, x2, y2, mask))

        logger.debug("\033[1;31m" + f"Post Process time = {1000*(time() - begin_time):.2f} ms" + "\033[0m")

        return results

coco_names = ["background","person","knife","fork","cup","giraffe","plate","table","cake","fence","hat","terrain","tree","car"]

rdk_colors = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),(49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
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