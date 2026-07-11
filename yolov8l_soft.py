"""
yolov8l_soft.py — 基于 YOLOv8l + SORT 的车辆检测与计数方案

【与 yolov8l_ByteTrack.py 的核心区别】
  跟踪器方案: 使用 SORT（Sort类，由 kalman 模块提供）
    - 先调用 model(frame, ...) 做裸检测，再手动组装 dets 数组传给 tracker.update()
    - 使用卡尔曼滤波 + 匈牙利算法进行跟踪
    - 对低置信度检测框直接丢弃（hard threshold），不进行二次匹配
    - 依赖: from kalman import * 和 import imutils
  适用场景: 实现简单直观，适合低遮挡、车辆稀疏的场景
  性能特点: 对遮挡和密集场景的鲁棒性不如 ByteTrack 方案；ID Switch 概率较高
"""

from kalman import *
import imutils
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

line = [(0, 100), (2560, 100)]
# 车辆总数
counter = 0
# 正向车道的车辆数据
counter_up = 0
# 逆向车道的车辆数据
counter_down = 0

# 创建跟踪器对象
tracker = Sort()
memory = {}


# 线与线的碰撞检测：叉乘的方法判断两条线是否相交
# 计算叉乘符号
def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


# 检测AB和CD两条直线是否相交
def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


# 利用yoloV8l模型进行目标检测（PyTorch + GPU）
model = YOLO('./yolov8l.pt') 

# 生成多种不同的颜色
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(200, 3), dtype='uint8')

# 读取图像
# frame = cv2.imread('./images/car2.jpg')
# (W,H) = (None,None)
# (H,W) = frame.shape[:2]
# 视频
vs = cv2.VideoCapture('./input/test_1.mp4')
(W, H) = (None, None)
writer = None
try:
    # 从视频对象中读取总帧数
    total = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频总帧数: {total}")
except:
    # 某些视频源（如摄像头实时流）不支持读取帧数，静默跳过
    pass

# 调试用：帧计数器
frame_count = 0
# 遍历每一帧图像
while True:
    (grabed, frame) = vs.read()
    if not grabed:
        break
    frame_count += 1
    if W is None or H is None:
        (H,W) = frame.shape[:2]
    # 使用YOLOv8l + GPU进行推理
    start = time.time()
    results = model(frame, conf=0.3, iou=0.4, classes=[2, 5, 7])
    end = time.time()

    # 解析检测结果
    # ultralytics返回list，每帧一个Results对象
    dets = []
    if len(results) > 0 and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            dets.append([x1, y1, x2, y2, conf])

    # 类型设置
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    dets = np.asarray(dets)

    # # 显示
    # plt.imshow(frame[:,:,::-1])
    # plt.show()

    # SORT目标跟踪（始终更新，避免跳过帧导致memory丢失）
    tracks = tracker.update(dets) if np.size(dets) > 0 else np.empty((0, 5))
    # 跟踪框
    boxes = []
    # 车辆ID（SORT分配的唯一ID）
    indexIDs = []
    # previous 是上一帧的所有跟踪结果，格式是字典：{track_id: [x1, y1, x2, y2]}，memory是当前帧的跟踪结果，格式相同
    previous = memory.copy()
    memory = {}
    #每一个track:[x1, y1, x2, y2, id]
    for track in tracks:
        boxes.append([track[0], track[1], track[2], track[3]])
        indexIDs.append(int(track[4]))
        memory[indexIDs[-1]] = boxes[-1]
    # # 调试：打印跟踪结果
    # print(f"[帧{frame_count}] 跟踪目标数:{len(tracks)}, 跟踪ID:{indexIDs}, 前一帧ID:{list(previous.keys())}")

    # 碰撞检测
    if len(boxes) > 0:
        i = int(0)
        # 遍历跟踪框
        for box in boxes:
            (x, y) = (int(box[0]), int(box[1]))
            (w, h) = (int(box[2]), int(box[3]))
            color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
            cv2.rectangle(frame, (x, y), (w, h), color, 2)

            # 根据在上一帧和当前帧的检测结果，利用虚拟线圈完成车辆计数
            if indexIDs[i] in previous:
                previous_box = previous[indexIDs[i]]
                (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                p1 = (int(x2 + (w2 - x2) / 2), int(y2 + (h2 - y2) / 2))
                p0 = (int(x + (w - x) / 2), int(y + (h - y) / 2))

                # # 调试：打印每辆车的位置信息
                # print(f"  车辆ID:{indexIDs[i]}, 上一帧中心:({p1[0]},{p1[1]}), 当前帧中心:({p0[0]},{p0[1]}), 计线y={line[0][1]}")

                # 利用p0,p1与line进行碰撞检测
                if intersect(p0, p1, line[0], line[1]):
                    counter += 1
                    # 判断行进方向
                    if y2 > y:
                        counter_down += 1
                        # print(f"  >>> 车辆ID:{indexIDs[i]} 穿过计数线（下行），当前计数:{counter}")
                    else:
                        counter_up += 1
                        # print(f"  >>> 车辆ID:{indexIDs[i]} 穿过计数线（上行），当前计数:{counter}")
            # else:
            #     # 调试：打印没有前一帧数据的车辆
            #     print(f"  车辆ID:{indexIDs[i]} 无前一帧数据，无法计数")
            i += 1

    # 将车辆计数的相关结果放在视频上
    cv2.line(frame, line[0], line[1], (0, 255, 0), 3)
    cv2.putText(frame, str(counter), (30, 80), cv2.FONT_HERSHEY_DUPLEX, 3.0, (255, 0, 0), 3)
    cv2.putText(frame, str(counter_up), (130, 80), cv2.FONT_HERSHEY_DUPLEX, 3.0, (0, 255, 0), 3)
    cv2.putText(frame, str(counter_down), (230, 80), cv2.FONT_HERSHEY_DUPLEX, 3.0, (0, 0, 255), 3)

    # 将检测结果保存在视频
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter("./output/output_soft.mp4", fourcc, 30, (frame.shape[1], frame.shape[0]), True)
    writer.write(frame)
    # 无图形界面环境，不调用imshow，结果已保存到output.mp4
    # cv2.imshow("", frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

#"释放资源"
writer.release()
vs.release()
# cv2.destroyAllWindows()  # 无图形界面环境不需要
