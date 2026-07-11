"""
yolov8l_ByteTrack.py — 基于 YOLOv8l + ByteTrack 的车辆检测与计数方案

【与 yolov8l_soft.py 的核心区别】
  跟踪器方案: 使用 ByteTrack（集成在 Ultralytics 的 model.track() 中）
    - 直接调用 model.track(frame, ..., persist=True, tracker="bytetrack.yaml")
    - 基于卡尔曼滤波 + 匈牙利算法，但引入"二次匹配"策略：
      ① 先用高置信度框匹配；② 再用低置信度框与未匹配轨迹匹配
    - 无需额外导入跟踪器模块和 imutils/matplotlib 等依赖
  适用场景: 对遮挡、运动模糊、密集车流场景更鲁棒，ID Switch 更少
  性能特点: 利用低置信度检测框信息，跟踪连续性更好，但检测+跟踪整体计算量略高
"""

import time
import cv2
import numpy as np
from ultralytics import YOLO

line = [(0, 100), (2560, 100)]
# 车辆总数
counter = 0
# 正向车道的车辆数据
counter_up = 0
# 逆向车道的车辆数据
counter_down = 0

# 跨帧车辆位置记忆（用于计数判断）
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
    results = model.track(frame, conf=0.3, iou=0.4, classes=[2, 5, 7], persist=True, tracker="bytetrack.yaml")
    end = time.time()

    # 解析ByteTrack检测+跟踪结果（track_id由ByteTrack自动跨帧分配）
    boxes = []
    indexIDs = []
    if len(results) > 0 and results[0].boxes is not None and results[0].boxes.id is not None:
        for i in range(len(results[0].boxes)):
            box = results[0].boxes[i]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            track_id = int(box.id[0].item())
            boxes.append([x1, y1, x2, y2])
            indexIDs.append(track_id)

    # previous 是上一帧的所有跟踪结果，格式：{track_id: [x1, y1, x2, y2]}
    previous = memory.copy()
    memory = {}
    for i, track_id in enumerate(indexIDs):
        memory[track_id] = boxes[i]
    # # 调试：打印跟踪结果
    # print(f"[帧{frame_count}] 跟踪目标数:{len(indexIDs)}, 跟踪ID:{indexIDs}, 前一帧ID:{list(previous.keys())}")

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
        writer = cv2.VideoWriter("./output/output_ByteTrack.mp4", fourcc, 30, (frame.shape[1], frame.shape[0]), True)
    writer.write(frame)
    # 无图形界面环境，不调用imshow，结果已保存到output_ByteTrack.mp4
    # cv2.imshow("", frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

#"释放资源"
writer.release()
vs.release()
# cv2.destroyAllWindows()  # 无图形界面环境不需要
