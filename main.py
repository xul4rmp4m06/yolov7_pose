import torch
import cv2
import numpy as np
from models.experimental import attempt_load  # 載入模型的輔助函數
from utils.torch_utils import select_device
from utils.general import non_max_suppression_kpt, strip_optimizer, scale_coords
from utils.plots import output_to_keypoint, plot_skeleton_kpts,plot_one_box
from utils.datasets import letterbox


# --- 參數設定 ---
video_path = '1009-1.mp4'  # 替換成您的影片路徑
weights_path = 'yolov7-w6-pose.pt' # 替換成您的模型權重路徑
device_str = '0'  # 'cpu' 或 'cuda:0' (如果支援 CUDA)
image_size = 1920  # 模型輸入的圖片大小
conf_thres = 0.25   # 信心度閾值
iou_thres = 0.45    # IOU 閾值

# --- 1. 初始化 ---

# 選擇設備
device = select_device(device_str)
half = device.type != 'cpu'  # half precision only supported on CUDA

# 載入模型

model = attempt_load(weights_path, map_location=device)  # 載入 FP32 模型
model.eval() # 設置為評估模式
stride = int(model.stride.max())  # 獲取模型 stride

if half:
    model.half()  # 轉換為 FP16

# --- 2. 載入影片 ---
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: 無法開啟影片 {video_path}")
    exit()

# 獲取影片的寬高
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# (可選) 設置影片寫入器，以儲存結果
# out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

print("開始處理影片... 按 'q' 結束。")

# --- 3. 逐幀處理 ---
cv2.namedWindow('YOLOv7 Pose Estimation', cv2.WINDOW_NORMAL)
cv2.resizeWindow('YOLOv7 Pose Estimation', 800, 600)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # --- 4. 影像前處理 ---
    # 調整大小並填充，使其符合模型的輸入尺寸 (image_size)
    img, ratio, (dw, dh) = letterbox(frame, image_size, stride=stride)
    
    # 轉換
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
    img = np.ascontiguousarray(img)
    
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)  # 增加 batch 維度

    # --- 5. 執行推理 ---
    with torch.no_grad():
        pred = model(img)[0]

    # --- 6. NMS (非極大值抑制) ---
    # 'kpt_shape' (17, 3) 表示 17 個關鍵點，每個點有 (x, y, confidence)
    pred = non_max_suppression_kpt(pred, conf_thres, iou_thres, kpt_label=True)

    # --- 7. 處理偵測結果 ---
    # pred 是一個 list，每個元素對應一張圖片的偵測結果
    for i, det in enumerate(pred):  # 這裡 batch size 為 1，所以只會循環一次
        if len(det):
            # 將偵測框的座標從 img_size 縮放回原始 frame 的大小
            # --- 7. 處理偵測結果 ---
            for i, det in enumerate(pred):
                if len(det):
                    # --- 手動還原座標 (移除 padding 並縮放) ---

                    # 1. 還原偵測框 (Boxes) [x1, y1, x2, y2]
                    det[:, 0] = (det[:, 0] - dw) / ratio[0]  # x1
                    det[:, 1] = (det[:, 1] - dh) / ratio[1]  # y1
                    det[:, 2] = (det[:, 2] - dw) / ratio[0]  # x2
                    det[:, 3] = (det[:, 3] - dh) / ratio[1]  # y2

                    # 2. 還原關鍵點 (Keypoints) [x, y, conf]
                    # det[:, 6::3] 指的是從索引 6 開始，每 3 個取 1 個 (即所有 x 座標)
                    det[:, 6::3] = (det[:, 6::3] - dw) / ratio[0] # 所有 kpt 的 x
                    # det[:, 7::3] 指的是從索引 7 開始，每 3 個取 1 個 (即所有 y 座標)
                    det[:, 7::3] = (det[:, 7::3] - dh) / ratio[1] # 所有 kpt 的 y

                    # --- 8. 繪製結果 ---
                    # (您原有的繪圖迴圈)
            

                    # --- 8. 繪製結果 ---
                    # 繪製偵測框和關鍵點
                    for det_row in reversed(det):  # 遍歷每一條偵測結果
                        # 手動解包每一行
                        xyxy = det_row[:4]   # 前 4 個值是偵測框
                        conf = det_row[4]    # 第 5 個值是信心度
                        cls = det_row[5]     # 第 6 個值是類別
                        kpts = det_row[6:]   # 從第 6 個索引之後的所有值都是關鍵點

                        # 繪製骨架
                        # kpts 現在是一個 1D 張量 (shape [51])，len() 可以正常運作
                        plot_skeleton_kpts(frame, kpts, 3) 

                        # (可選) 繪製偵測框
                        label = f'{int(cls)} {conf:.2f}'
                        plot_one_box(xyxy, frame, label=label, color=(255, 0, 0))


    # --- 9. 顯示結果 ---
    cv2.imshow('YOLOv7 Pose Estimation', frame)
    
    # (可選) 寫入影格到輸出影片
    # out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 10. 清理 ---
cap.release()
# out.release()
cv2.destroyAllWindows()
print("處理完成。")


# --- 輔助函數 (Letterbox) ---
# 這個函數通常在 utils.datasets 中，但為求範例完整，將其複製於此
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)