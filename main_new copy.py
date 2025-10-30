#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[YOLOv7-Pose 整合版 v3 - 無 Ultralytics 依賴]
Multi-RTSP/Video real-time pose tracking:
  • 每個來源各開一個線程
  • 每個線程各自載入 YOLOv7-Pose 模型 (attempt_load)
  • 每個線程各自載入一個 *自訂的* SimpleCentroidTracker 追蹤器
  • 保留所有跌倒偵測、靜止偵測、Discord 功能
  • 為每個偵測到的 ID 動態開啟 *獨立的* 即時除錯圖表視窗
"""

import os, cv2, time, threading, numpy as np
import math
import logging
from datetime import datetime, timezone, timedelta
from collections import deque
import requests,json
from pathlib import Path

# --- 新增：YOLOv7-Pose 相關 imports ---
import torch
from models.experimental import attempt_load
from utils.torch_utils import select_device
from utils.general import non_max_suppression_kpt
# 假設 plot_skeleton_kpts 和 plot_one_box 存在於您的 utils.plots
try:
    from utils.plots import plot_skeleton_kpts, plot_one_box
except ImportError:
    print("警告：無法從 utils.plots 導入 plot_skeleton_kpts 或 plot_one_box。")
    # 如果導入失敗，您可能需要從 Wjdepp/yolov7-pose 倉庫中複製這些函式

# --- [新增] 
def get_color_for_tid(tid, s=1.0, v=1.0):
    """
    根據 Track ID 產生一個獨特且固定的 BGR 顏色
    """
    # 使用 HASH 值來確保顏色分佈均勻
    hash_val = hash(str(tid))
    # 將 hash 值映射到 0-179 的 H (Hue) 範圍
    # (OpenCV 的 H 範圍是 0-179)
    hue = int(abs(hash_val) % 180)
    # 固定的 Saturation (飽和度) 和 Value (亮度)
    saturation = int(255 * s)
    value = int(255 * v)
    
    # 建立 HSV 顏色
    hsv_color = np.array([[[hue, saturation, value]]], dtype=np.uint8)
    
    # 轉換為 BGR
    bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
    
    # 返回 (B, G, R) 元組
    return (int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2]))

# --- [新增] 即時繪圖輔助函式 ---
def _draw_curve(img, data_deque, label, color, min_val, max_val, baselines=None):
    """
    在 Numpy 影像上繪製一條曲線 (並可選地繪製基準線)
    [v2 - 修正標籤位置]
    """
    H, W, _ = img.shape
    
    # --- 1. 繪製基準線 (如果有的話) ---
    if baselines is not None:
        for baseline_val, baseline_label in baselines:
            # 1a. 計算 Y 座標 (使用與曲線相同的 min/max)
            val_norm = (baseline_val - min_val) / max(1e-6, (max_val - min_val))
            val_norm = np.clip(val_norm, 0.0, 1.0)
            y = int( (1.0 - val_norm) * (H - 40) ) + 20 
            
            # 1b. 畫線 (黃色虛線)
            baseline_color = (0, 255, 255) # Yellow
            line_x = 0
            for i in range(W) :
                if i >= W or line_x >= W:
                    break
                else:
                    if i % 2 == 0 :
                        line_x = line_x + 30
                        cv2.line(img, (line_x, y), (line_x + 10, y), color, 2, cv2.LINE_AA)
            # 1c. 畫標籤 (在線的右上方) [使用相對位置 W]
            label_x = W - 250 # 距離右側 250px
            label_y = y - 7   # 線的上方 7px
            cv2.putText(img, f"{baseline_label}: {baseline_val}", (label_x, label_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, baseline_color, 1)

    # --- 2. 繪製數據曲線 ---
    if len(data_deque) < 2:
        return # 點不夠，無法畫線

    points = []
    data_len = len(data_deque)
    for i in range(data_len):
        val = data_deque[i]
        x = int( (i / (data_deque.maxlen - 1)) * (W - 20) ) + 10 
        val_norm = (val - min_val) / max(1e-6, (max_val - min_val))
        val_norm = np.clip(val_norm, 0.0, 1.0)
        y = int( (1.0 - val_norm) * (H - 40) ) + 20
        points.append((x, y))

    pts_np = np.array(points, dtype=np.int32)
    cv2.polylines(img, [pts_np], isClosed=False, color=color, thickness=2)
    
    # --- 3. 繪製曲線標籤 [使用相對位置 W 和 H] ---
    lable_Y = H - 15 # 統一放在底部 (H - 15)
    
    if "Angle" in label:
        lable_X = int(W * 0.05) # 放在左邊 5% 處
    elif "V/H Ratio" in label:
        lable_X = int(W * 0.35) # 放在中間 35% 處
    elif "Leg Ratio" in label:
        lable_X = int(W * 0.65) # 放在右邊 65% 處
    else: # Fall Like
        lable_X = int(W * 0.05) # 放在左上角
        lable_Y = 40
        
    # 繪製標籤和 Y 軸範圍
    cv2.putText(img, f"{label}: {data_deque[-1]:.2f}", (lable_X, lable_Y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(img, f"{max_val:.1f}", (10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(img, f"{min_val:.1f}", (10, H - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)# ───────────────────── 0. 低延遲 FFmpeg 參數 ─────────────────────



os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;udp|"
    "fflags;nobuffer|"
    "flags;low_delay|"
    "probesize;32|analyzeduration;0"
)
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "quiet"

# ───────────────────── 1. 來源清單 ─────────────────────
USE_RTSP   = False
USE_VIDEOS = True

RTSP_URLS = [
    #r'rtsp://root:Asc%231432320@172.16.83.128:554/axis-media/media.amp?videocodec=h264&resolution=1920x1080&fps=10',
    #r'rtsp://root:Asc%231432320@172.16.83.171:554/axis-media/media.amp?videocodec=h264&resolution=1920x1080&fps=10',
    r'rtsp://admin:Asc%231432320@172.16.83.47:554/stream1',
    r'rtsp://root:Asc%231432320@172.16.83.42:554/axis-media/media.amp?videocodec=h264&resolution=1920x1080&fps=10',
]

VIDEO_FILES = [
    #r"C:\Users\User\SynologyDrive\py\aaa\test2-1.avi",
    #r"C:\Users\User\SynologyDrive\py\aaa\test2.avi",
    r"C:\Users\User\SynologyDrive\py\aaa\sssss.mp4",
    #r"C:\Users\User\SynologyDrive\py\yoloV7_pose\1009-1.mp4",
]


# ───────────────────── 2. 背景取流 (保留不變) ─────────────────────
class LatestFrameRTSP:
    def __init__(self, url: str, name: str):
        self.name = name
        self.cap  = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            raise RuntimeError(f"❌ 無法開啟 {url}")
        self.frame, self.lock = None, threading.Lock()
        self.running = True
        self.thread  = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()
    def _reader(self):
        while self.running:
            ok, frm = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = frm
            else:
                time.sleep(0.1)
        self.cap.release()
    def get(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()
    def stop(self):
        self.running = False
        self.thread.join()

class LatestFrameVideo:
    def __init__(self, path: str, name: str):
        self.name = name
        self.cap  = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"[{name}] 無法開啟影片：{path}")
        self.fps   = self.cap.get(cv2.CAP_PROP_FPS) or 0.0
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    def get(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame
    def release(self):
        try:
            self.cap.release()
        except Exception:
            pass

# ───────────────────── Discord Webhooks (保留不變) ─────────────────────
Alarm_bot = "https://discord.com/api/webhooks/1428747809250742343/7Y64YJRWq5SILy0eUmLVS5URYISgfE8ZnSBh8CMLLErv4qrTPegv5oyCMFWBT-k33ndD"
fail_like_bot = "https://discord.com/api/webhooks/1428748089719787550/0ib9j9yrokoadKq9LxRflU1DbW3dSuCY67MNQvCccJToTfTcCKxscrLEn_Vd4CDPXmOq"
img_bot = "https://discord.com/api/webhooks/1428748333488410624/tgJWFn_yYc0NdecNurixfJLktwFAU0cuXIDwYDBg2QdkC9JkS3YDagggDv_Vyx0lwj28"
systemlog_bot = "https://discord.com/api/webhooks/1428748579069235300/qm8x8uzbcWWkcOW4Vc9LTdcqf-rsQUQEZ17iDtLG82Fcy_-OSKvcGCkeTrvFp5__i6fa"

def Discord_message(bot,text: str,at=False):
    USER_ID = "447408102900498433"  # 要@的那個人
    if at == False:
        payload = {"content": f"{text}","allowed_mentions": { "users": [USER_ID] }}
    else:
        payload = {"content": f"<@{USER_ID}> ，{text}","allowed_mentions": { "users": [USER_ID] }}
    r = requests.post(bot, json=payload)
    print(r.status_code, r.text)

def Discord_send_image(bot,frame, text):
    USER_ID = "447408102900498433"  # 要@的那個人
    at = f"<@{USER_ID}>" if USER_ID else ""
    content = f"{text}".strip()
    payload = {"content": content,"allowed_mentions": {"users": [USER_ID]} if USER_ID else {"users": []}}
    ok, buf = cv2.imencode(".png", frame)
    if not ok: raise RuntimeError("影像編碼 PNG 失敗")
    files = {"file": ("frame.png", buf.tobytes(), "image/png")}
    data = {"payload_json": json.dumps(payload, ensure_ascii=False)}
    try:
        resp = requests.post(bot, data=data, files=files, timeout=20)
        return resp
    finally:
        return 

# ───────────────────── 3. 參數設定 (YOLOv7 + 原有邏輯) ─────────────────────
MODEL_DIR         = "yolov7-w6-pose.pt"
RAWIMGSZ          = 1920
PLAYBACK_FPS      = 20.0
DELAY_MS          = int(1000 / PLAYBACK_FPS)
MOVE_THRESH_PX    = 4                         
STATIC_TIME_S     = 240.0                       
CONF_THR          = 0.60
IOU_THR           = 0.45
# --- (您可以調整下面這幾項來改變「精準度」) ---
BOX_CONF_THR      = 0.45  # (已放寬) 偵測框信心度 > 25%
KP_CONF_THR       = 0.1   # (已放寬) 關鍵點信心度 > 10%
MIN_KP_OK         = 6     # (已放寬) 至少 2 個可靠關鍵點
# ---
MAX_DET           = 800
SKELETON = [
    (15,13),(13,11),(16,14),(14,12),(11,12),
    (5,11),(6,12),(5,6),(5,7),(7,9),
    (6,8),(8,10),(1,2),(0,1),(0,2),
    (1,3),(2,4),(3,5),(4,6)
]

stop_event = threading.Event()

def _get_dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

# ───────────────────── 4. 偵測/濾波 相關類別與函式 (保留不變) ─────────────────────

class FallDetector:
    def __init__(
        self,
        fps=10,
        drop_sigma_k=0.5,
        post_motion_eps=3.0,
        post_flat_vh=0.85,
        fall_cooldown_s=2.0,
        vertical_fall_vh_thresh=1.5, 
        vertical_angle_deg_thresh = 20,
        angle_deg_thresh=55.0,
        horizontal_fall_vh_thresh=0.80,
        leg_ratio_thresh=0.4,
        frames_for_static_fall=60, 
        frames_to_alert=60,
        conf_thr=0.5,
        debug=True
    ):
        # 儲存所有參數
        self.fps = fps
        self.drop_sigma_k = drop_sigma_k
        self.post_motion_eps = post_motion_eps
        self.post_flat_vh = post_flat_vh
        self.fall_cooldown_s = fall_cooldown_s
        self.vertical_fall_vh_thresh = vertical_fall_vh_thresh
        self.vertical_angle_deg_thresh = vertical_angle_deg_thresh
        self.angle_deg_thresh = angle_deg_thresh
        self.horizontal_fall_vh_thresh = horizontal_fall_vh_thresh
        self.leg_ratio_thresh = leg_ratio_thresh
        self.frames_for_static_fall = frames_for_static_fall
        self.frames_to_alert = frames_to_alert
        self.conf_thr = conf_thr
        self.debug = debug
        self.track_history = {}
        win = max(5, int(self.fps * 1.0))
        self._hip_y_hist = {}
        self._vh_hist = {}
        self._centroid_hist = {}
        self._last_fall_time = {}
        self._win = win
    
    def _update_fall_event(self, tid, pts, ts):
        P = np.asarray(pts, dtype=float)
        conf = P[:, 2]
        thr = self.conf_thr

        if tid not in self._hip_y_hist:
            self._hip_y_hist[tid] = deque(maxlen=self._win)
            self._vh_hist[tid] = deque(maxlen=self._win)
            self._centroid_hist[tid] = deque(maxlen=self._win)
            self._last_fall_time[tid] = -1.0

        if (conf[11] < thr) or (conf[12] < thr):
            return False

        hip_mid = (P[11, :2] + P[12, :2]) * 0.5
        valid = P[conf >= thr, :2]
        if valid.shape[0] < 6:
            return False

        xs, ys = valid[:, 0], valid[:, 1]
        H = max(1.0, xs.max() - xs.min())
        V = max(1.0, ys.max() - ys.min())
        vh_ratio = V / H
        centroid = valid.mean(axis=0)

        self._hip_y_hist[tid].append(float(hip_mid[1]))
        self._vh_hist[tid].append(float(vh_ratio))
        self._centroid_hist[tid].append(centroid)

        if len(self._hip_y_hist[tid]) < self._hip_y_hist[tid].maxlen:
            return False

        a = np.array(self._hip_y_hist[tid], dtype=float)
        N = len(a)
        first_half, second_half = a[:N//2], a[N//2:]
        med1, med2 = np.median(first_half), np.median(second_half)
        mad = np.median(np.abs(first_half - med1)) + 1e-6
        drop_sigma = (med2 - med1) / mad

        c = np.vstack(self._centroid_hist[tid])
        move_range = float(np.linalg.norm(c.max(axis=0) - c.min(axis=0), ord=2))
        is_static = (move_range <= self.post_motion_eps)
        is_flat = (np.median(self._vh_hist[tid]) <= self.post_flat_vh)
        
        head_hip_vert_norm = abs(P[0, 1] - hip_mid[1]) / V if conf[0] > thr else 1.0
        is_headhip_small = head_hip_vert_norm <= 0.3 # 上身壓縮

        is_drop_event = (drop_sigma >= self.drop_sigma_k)
        post_ok = is_static and (is_flat or is_headhip_small)

        if is_drop_event and post_ok:
            if (self._last_fall_time[tid] < 0) or (ts - self._last_fall_time[tid] >= self.fall_cooldown_s):
                self._last_fall_time[tid] = ts
                return True
        return False

    def _check_static_pose(self, pts):
        """
        [整合後的靜態姿勢分析]
        [修改]：回傳一個包含所有指標的 dict
        """
        P = np.asarray(pts, dtype=float)
        mask = P[:, 2] >= self.conf_thr

        # [新增] 建立一個預設的回傳值
        default_metrics = {
            'angle': 0.0,
            'vh_ratio': 1.0,
            'leg_ratio': 1.0,
            'fall_like': False,
            'valid': False # 標記此幀是否有效
        }

        if mask.sum() < 6:
            return default_metrics # [修改] 回傳預設值

        try:
            # --- 1. 計算所有需要的指標 ---
            p_sho_mid = (P[5, :2] + P[6, :2]) * 0.5
            p_hip_mid = (P[11, :2] + P[12, :2]) * 0.5
            vx, vy = p_hip_mid[0] - p_sho_mid[0], p_hip_mid[1] - p_sho_mid[1]
            n = math.hypot(vx, vy)
            angle_deg = 0.0
            if n > 1e-6:
                vy_n = np.clip(vy / n, -1.0, 1.0)
                angle_deg = math.degrees(math.acos(vy_n))
                if angle_deg > 90:
                    angle_deg = 180 - angle_deg
            
            xy = P[mask, :2]
            xs, ys = xy[:, 0], xy[:, 1]
            H_spread = max(1.0, xs.max() - xs.min())
            V_spread = max(1.0, ys.max() - ys.min())
            vh_ratio = V_spread / H_spread
            
            p_ank_mid = (P[15, :2] + P[16, :2]) * 0.5
            p_nose = P[0, :2]
            leg_vertical_span = abs(p_hip_mid[1] - p_ank_mid[1])
            body_height_approx = abs(p_nose[1] - p_ank_mid[1])
            leg_compression_ratio = leg_vertical_span / body_height_approx if body_height_approx > 1e-6 else 1.0

            # --- 2. 綜合判斷 ---
            cond_horizontal_fall = (angle_deg >= self.angle_deg_thresh and 
                                    vh_ratio <= self.horizontal_fall_vh_thresh)

            cond_vertical_fall = (angle_deg < self.angle_deg_thresh and 
                                 (vh_ratio <= self.vertical_fall_vh_thresh or leg_compression_ratio <= self.leg_ratio_thresh))
            
            fall_like = cond_horizontal_fall or cond_vertical_fall

            if self.debug and hasattr(self, '_is_debugging_tid') and self._is_debugging_tid:
                print(
                    f"    [Static Pose Check] Angle={angle_deg:.1f}°, V/H={vh_ratio:.2f}, LegRatio={leg_compression_ratio:.2f}\n"
                    f"    -> H_Fall(angle>{self.angle_deg_thresh}, vh<{self.horizontal_fall_vh_thresh})={cond_horizontal_fall} | "
                    f"V_Fall(angle<{self.angle_deg_thresh}, vh<{self.vertical_fall_vh_thresh} or leg<{self.leg_ratio_thresh})={cond_vertical_fall} | "
                    f"fall_like={fall_like}"
                )
            
            # [修改] 回傳包含所有值的 dict
            return {
                'angle': angle_deg,
                'vh_ratio': vh_ratio,
                'leg_ratio': leg_compression_ratio,
                'fall_like': fall_like,
                'valid': True
            }

        except IndexError:
            return default_metrics # [修改] 回傳預設值

    def detect(self, pts, tid, frame_num):
        """
        主偵測函式，結合事件式偵測與狀態機。
        [修改]：回傳 (is_alert, metrics_dict)
        """
        tid = int(tid)
        P = np.asarray(pts, dtype=float)
        self._is_debugging_tid = True 

        if tid not in self.track_history:
            self.track_history[tid] = {
                'state': 'NORMAL', 
                'fallen_count': 0, 
                'last_hip_y': None,
                'last_frame_num': frame_num,
                'static_fall_pose_count': 0
            }
        
        history = self.track_history[tid]
        current_state = history['state']
        
        # [修改] 建立一個預設的 metrics，以便在早期 return 時也能回傳
        default_metrics = {
            'angle': 0.0, 'vh_ratio': 1.0, 'leg_ratio': 1.0, 
            'fall_like': False, 'valid': False
        }

        try:
            hip_mid_y = (P[11, 1] + P[12, 1]) * 0.5
        except IndexError:
            history['state'] = 'NORMAL'
            history['last_hip_y'] = None
            return False, default_metrics # [修改] 回傳 tuple

        vertical_speed = 0.0
        if history['last_hip_y'] is not None:
            delta_frames = frame_num - history['last_frame_num']
            if delta_frames > 0:
                delta_y = hip_mid_y - history['last_hip_y']
                vertical_speed = delta_y / delta_frames
        
        history['last_hip_y'] = hip_mid_y
        history['last_frame_num'] = frame_num
        
        # --- [修改] 接收 metrics dict ---
        pose_metrics = self._check_static_pose(P)
        is_in_fallen_pose = pose_metrics['fall_like']
        
        # 事件式偵測作為補充
        ts = frame_num / max(1e-6, float(self.fps))
        event_trigger = self._update_fall_event(tid, P, ts)

        is_on_ground = is_in_fallen_pose or event_trigger
        
        # --- 狀態機邏輯 ---
        new_state = current_state
        if current_state == 'NORMAL':
            if vertical_speed > 15.0:
                new_state = 'FALLING'
                history['static_fall_pose_count'] = 0
            elif is_in_fallen_pose:
                history['static_fall_pose_count'] += 1
                if history['static_fall_pose_count'] >= self.frames_for_static_fall:
                    new_state = 'FALLEN'
                    history['fallen_count'] = 1
            else:
                history['static_fall_pose_count'] = 0

        elif current_state == 'FALLING':
            if is_in_fallen_pose:
                new_state = 'FALLEN'
                history['fallen_count'] = 1
            else:
                new_state = 'NORMAL'
        
        elif current_state == 'FALLEN':
            if is_in_fallen_pose:
                history['fallen_count'] += 1
                if history['fallen_count'] >= self.frames_to_alert:
                    new_state = 'ALERT'
            else:
                new_state = 'NORMAL'
                history['fallen_count'] = 0
        
        elif current_state == 'ALERT':
            if not is_in_fallen_pose:
                new_state = 'NORMAL'
                history['fallen_count'] = 0

        if new_state != 'NORMAL':
            history['static_fall_pose_count'] = 0
        history['state'] = new_state
        
        TZ = timezone(timedelta(hours=8), name="Asia/Taipei")
        now = datetime.now(TZ)
        if self.debug and pose_metrics['valid']: # [修改] 只在 metrics 有效時印出
            print(
                f"[fall][tid={tid}][frame={frame_num}] [time={now.strftime('%Y/%m/%d %H:%M:%S')}] "
                f"state: {current_state:>7s} → {new_state:>7s} | "
                f"alert_count={history['fallen_count']}/{self.frames_to_alert} | "
                f"static_pose_count={history['static_fall_pose_count']}/{self.frames_for_static_fall}"
                "\n-----------------------------------------------------------------------------------"
            )
        
        self._is_debugging_tid = False
        
        is_alert_state = (new_state == 'ALERT')
        
        if event_trigger:
            print(f"!!! EVENT TRIGGERED for tid={tid} !!!")
            # [修改] 回傳 (is_alert, metrics)
            return True, pose_metrics 
            
        # [修改] 回傳 (is_alert, metrics)
        return is_alert_state, pose_metrics

fall_detector = FallDetector()

def get_stable_center(P, conf_thr=0.6):
    ok = P[:,2] >= conf_thr
    cands = []
    if ok[11] and ok[12]:
        hip_mid = ( (P[11,0] + P[12,0]) * 0.5, (P[11,1] + P[12,1]) * 0.5 )
        cands.append(("hip", hip_mid))
    if ok[5] and ok[6]:
        sho_mid = ( (P[5,0] + P[6,0]) * 0.5, (P[5,1] + P[6,1]) * 0.5 )
        cands.append(("sho", sho_mid))

    if not cands:
        if ok.any():
            xy = P[ok,:2]
            return float(xy[:,0].mean()), float(xy[:,1].mean())
        else:
            return float(P[0,0]), float(P[0,1])

    w = {"hip":0.7, "sho":0.3}
    sx = sy = sw = 0.0
    for tag, (x,y) in cands:
        ww = w[tag]
        sx += ww*x; sy += ww*y; sw += ww
    return sx/sw, sy/sw

class OneEuro:
    def __init__(self, freq=30.0, min_cutoff=1.0, beta=0.005, d_cutoff=1.0):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def _alpha(self, cutoff):
        tau = 1.0 / (2*np.pi*cutoff)
        te  = 1.0 / max(1e-6, self.freq)
        return 1.0 / (1.0 + tau/te)

    def __call__(self, t, x):
        if self.t_prev is None:
            self.t_prev = t; self.x_prev = x
            return x
        dt = max(1e-6, t - self.t_prev)
        self.freq = 1.0 / dt
        dx = (x - self.x_prev) * self.freq
        a_d = self._alpha(self.d_cutoff)
        dx_hat = a_d*dx + (1-a_d)*self.dx_prev
        cutoff = self.min_cutoff + self.beta*abs(dx_hat)
        a = self._alpha(cutoff)
        x_hat = a*x + (1-a)*self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

def estimate_jitter(P, cx, cy, conf_thr=0.6):
    ok = P[:,2] >= conf_thr
    if ok.sum() < 4:
        return 2.0  # 缺點時給個保守值
    xy = P[ok,:2]
    r = np.hypot(xy[:,0]-cx, xy[:,1]-cy)
    med = np.median(r)
    mad = np.median(np.abs(r - med)) + 1e-6
    return float(mad)

# --- 新增：自訂的簡易中心點追蹤器 ---
class SimpleCentroidTracker:
    def __init__(self, max_age=30, max_dist_px=100):
        self.next_tid = 0
        self.tracks = {}  # 儲存: tid -> {'centroid':(cx,cy), 'kpts':(17,3), 'box':(4,), 'conf':f, 'age':0}
        self.max_age = max_age          # 消失N幀後刪除
        self.max_dist_px = max_dist_px  # 幀間最大移動距離

    def _register(self, centroid, kpts, box, conf):
        tid = self.next_tid
        self.tracks[tid] = {
            'centroid': centroid,
            'kpts': kpts,
            'box': box,
            'conf': conf,
            'age': 0
        }
        self.next_tid += 1
        return tid

    def _deregister(self, tid):
        if tid in self.tracks:
            del self.tracks[tid]

    def update(self, det_results_v7_np):
        """
        用 YOLOv7 (N, 57) 的 numpy 陣列來更新追蹤器
        回傳: list of (tid, kpts_array, box_conf, box_xyxy)
        """
        
        # 1. 為當前幀的所有偵測結果計算中心點
        new_detections = []
        for i in range(len(det_results_v7_np)):
            det_row = det_results_v7_np[i]
            kpts = det_row[6:].reshape(17, 3)
            box = det_row[:4]
            conf = det_row[4]
            
            # 使用您 v8 腳本中的 'get_stable_center'
            centroid = get_stable_center(kpts, conf_thr=KP_CONF_THR)
            new_detections.append({
                'centroid': centroid,
                'kpts': kpts,
                'box': box,
                'conf': conf
            })

        # 如果沒有已追蹤目標，全部註冊為新的
        if len(self.tracks) == 0:
            for det in new_detections:
                self._register(det['centroid'], det['kpts'], det['box'], det['conf'])
            # (第一次不回傳，等待下一幀)
            return []

        # 建立現有TID和新偵測的中心點列表
        existing_tids = list(self.tracks.keys())
        existing_centroids = [t['centroid'] for t in self.tracks.values()]
        new_centroids = [d['centroid'] for d in new_detections]

        active_tracks_output = [] # 準備回傳的列表

        if len(new_detections) > 0 and len(existing_tids) > 0:
            # 2. 計算距離矩陣 (舊 track vs 新 det)
            dist_matrix = np.zeros((len(existing_centroids), len(new_centroids)))
            for i in range(len(existing_centroids)):
                for j in range(len(new_centroids)):
                    dist_matrix[i, j] = _get_dist(existing_centroids[i], new_centroids[j])
            
            # 3. 貪婪匹配 (Greedy Matching)
            matched_new_indices = set()
            unmatched_track_tids = set(existing_tids)
            matches = []

            for t_idx, tid in enumerate(existing_tids):
                if len(new_centroids) == 0: break
                best_dist = np.inf
                best_d_idx = -1
                
                for d_idx in range(len(new_centroids)):
                    if d_idx in matched_new_indices:
                        continue
                    
                    d = dist_matrix[t_idx, d_idx]
                    if d < best_dist:
                        best_dist = d
                        best_d_idx = d_idx
                
                if best_dist < self.max_dist_px:
                    matches.append((tid, best_d_idx))
                    matched_new_indices.add(best_d_idx)
                    if tid in unmatched_track_tids:
                        unmatched_track_tids.remove(tid)

            # 4. 更新匹配上的 Track
            for tid, d_idx in matches:
                det = new_detections[d_idx]
                self.tracks[tid]['centroid'] = det['centroid']
                self.tracks[tid]['kpts'] = det['kpts']
                self.tracks[tid]['box'] = det['box']
                self.tracks[tid]['conf'] = det['conf']
                self.tracks[tid]['age'] = 0
                active_tracks_output.append((tid, det['kpts'], det['conf'], det['box']))

            # 5. 註冊未匹配上的 Detections (新目標)
            for d_idx in range(len(new_detections)):
                if d_idx not in matched_new_indices:
                    det = new_detections[d_idx]
                    new_tid = self._register(det['centroid'], det['kpts'], det['box'], det['conf'])
                    active_tracks_output.append((new_tid, det['kpts'], det['conf'], det['box']))

            # 6. 處理未匹配上的 Tracks (舊目標)
            for tid in unmatched_track_tids:
                self.tracks[tid]['age'] += 1
                if self.tracks[tid]['age'] > self.max_age:
                    self._deregister(tid)
        
        elif len(new_detections) == 0:
             for tid in existing_tids:
                self.tracks[tid]['age'] += 1
                if self.tracks[tid]['age'] > self.max_age:
                    self._deregister(tid)
        
        return active_tracks_output


# ───────────────────── 5. 每線程 worker：(修改為 YOLOv7 + 自訂追蹤器) ─────────────────────
def worker(stream, device="cuda:0"): #cuda:0
    """
    每個線程都各自載入一顆 YOLOv7 模型 + SimpleCentroidTracker 追蹤器
    """
    STATIONARY = 0
    FALLEN = 0
    FALL_LIKE = 0
    ALERT_IDs = {}
    TZ = timezone(timedelta(hours=8), name="Asia/Taipei")
    center_filters = {}
    jitter_hist = {}
    
    # --- [修改] 即時繪圖相關初始化 ---
    PLOT_MAX_LEN = 200 # 圖表顯示最近 200 幀
    plot_data_all_tids = {} # 結構: { tid: {'angle': deque, 'vh_ratio': deque, ..., 'age': 0} }
    PLOT_IMG_H, PLOT_IMG_W = 400, 1500

    def next_alert(tid: str,alert_type: str) -> dict:
        now = datetime.now(TZ)
        slot = ALERT_IDs.get(tid)
        if slot is None: slot = {"seq_next": 0, "records": []}
        seq = slot["seq_next"]; slot["seq_next"] += 1
        rec = {"tid": tid,"seq": seq,"cam_name": window_name,"alert_type":alert_type,"time":  now.strftime("%Y/%m/%d %H:%M:%S"),"report": False}
        slot["records"].append(rec); ALERT_IDs[tid] = slot
        return rec
    
    def pass_filters(box_conf: float, kp_conf: np.ndarray) -> bool:
        # (使用您在 3. 參數設定中放寬後的值)
        return (box_conf >= BOX_CONF_THR) and ((kp_conf >= KP_CONF_THR).sum() >= MIN_KP_OK)
    
    # --- 初始化 YOLOv7 模型 ---
    print(f"[{stream.name}] 正在載入 YOLOv7 模型 {MODEL_DIR} 到 {device}...")
    device_obj = select_device(device)
    model = attempt_load(MODEL_DIR, map_location=device_obj)
    stride = int(model.stride.max())
    model.eval()
    half = (device_obj.type != 'cpu')
    if half:
        model.half()
    
    def align_to_stride(x: int, s: int) -> int:
        return (x + s - 1) // s * s
    
    raw_imgsz = RAWIMGSZ 
    imgsz = align_to_stride(raw_imgsz, stride)

    # --- 初始化 自訂追蹤器 ---
    tracker = SimpleCentroidTracker(
        max_age=int(PLAYBACK_FPS * 1.5), # 消失 1.5 秒後刪除
        max_dist_px=int(imgsz * 0.15)      # 最大移動距離設為影像尺寸的 15%
    )
    
    prev_center, static_start_time = {}, {}
    window_name = str(stream.name)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)
    frame_count = 0
    
    while not stop_event.is_set():
        frame_count += 1 
        now = datetime.now(TZ)
        frame = stream.get()
        if frame is None:
            if isinstance(stream, LatestFrameVideo):
                break
            time.sleep(0.01)
            continue

        t0 = time.perf_counter()
        
        # 1. v7 Pre-processing
        img, ratio, (dw, dh) = letterbox(frame, imgsz, stride=stride)
        img_for_torch = img[:, :, ::-1].transpose(2, 0, 1)
        img_for_torch = np.ascontiguousarray(img_for_torch)
        
        img_torch = torch.from_numpy(img_for_torch).to(device_obj)
        img_torch = img_torch.half() if half else img_torch.float()
        img_torch /= 255.0
        if img_torch.ndimension() == 3:
            img_torch = img_torch.unsqueeze(0)

        # 2. v7 Inference
        with torch.no_grad():
            pred = model(img_torch)[0]

        # 3. v7 NMS
        det = non_max_suppression_kpt(pred, CONF_THR, IOU_THR, kpt_label=True)[0]
        
        proc_ms = (time.perf_counter() - t0) * 1000
        
        det_np_scaled = np.array([]) # 準備給 tracker 的 numpy 陣列

        if det is not None and len(det):
            # 4. v7 座標還原 (在傳給 tracker 之前)
            h, w, _ = frame.shape
            
            det[:, 0] = (det[:, 0] - dw) / ratio[0]  # x1
            det[:, 1] = (det[:, 1] - dh) / ratio[1]  # y1
            det[:, 2] = (det[:, 2] - dw) / ratio[0]  # x2
            det[:, 3] = (det[:, 3] - dh) / ratio[1]  # y2

            det[:, 6::3] = (det[:, 6::3] - dw) / ratio[0] # kpt x
            det[:, 7::3] = (det[:, 7::3] - dh) / ratio[1] # kpt y
            
            # --- [FIX] 修正 SyntaxError ---
            # 建立一個包含所有 'x' 座標索引的列表 (0, 2, 6, 9, 12, ...)
            # 假設有 17 個關鍵點 (17*3=51)，總共 6+51=57 欄
            all_x_indices = [0, 2] + list(range(6, 57, 3))
            # 建立一個包含所有 'y' 座標索引的列表 (1, 3, 7, 10, 13, ...)
            all_y_indices = [1, 3] + list(range(7, 57, 3))

            # 使用這個有效的列表來進行索引和 clamping
            det[:, all_x_indices] = det[:, all_x_indices].clamp(0, w)
            det[:, all_y_indices] = det[:, all_y_indices].clamp(0, h)
            # --- [FIX END] ---
            
            det_np_scaled = det.cpu().numpy()
        
        # 5. 更新 自訂追蹤器
        matched_results = tracker.update(det_np_scaled)

        # 6. 保留原有的 v8 腳本邏輯 (跌倒/靜止偵測)
        
        # --- [修改] 清理/更新 繪圖字典 (並同步銷毀視窗) ---
        tids_to_delete = []
        for tid_key, data in plot_data_all_tids.items():
            data['age'] += 1
            if data['age'] > PLOT_MAX_LEN * 2: # 消失超過 400 幀就移除
                tids_to_delete.append(tid_key)
        
        for tid_key in tids_to_delete:
            window_name_to_close = f"Debug Plot (TID: {tid_key}) - {stream.name}"
            try:
                cv2.destroyWindow(window_name_to_close)
            except cv2.error:
                pass 
            del plot_data_all_tids[tid_key]
        # --- [修改結束] ---

        if len(matched_results) > 0:
            for (tid, kpts_array, box_conf, (x1,y1,x2,y2)) in matched_results:
                
                pts = kpts_array # (17, 3) numpy array
                
                # 執行 v8 腳本的過濾器
                kp_conf = pts[:, 2] # (17,)
                # if not pass_filters(box_conf, kp_conf): # <--- 您可以取消註解這行來啟用過濾
                #     continue
                
                # --- (*** 以下是從 v8 腳本中完整複製的邏輯 ***) ---
                raw_cx, raw_cy = get_stable_center(pts, conf_thr=KP_CONF_THR)
                now_t = time.monotonic()
                flt = center_filters.get(tid)
                if flt is None:
                    flt = {
                        "fx": OneEuro(freq=PLAYBACK_FPS, min_cutoff=1.0, beta=0.01, d_cutoff=1.0),
                        "fy": OneEuro(freq=PLAYBACK_FPS, min_cutoff=1.0, beta=0.01, d_cutoff=1.0),
                    }
                    center_filters[tid] = flt
                cx = flt["fx"](now_t, raw_cx)
                cy = flt["fy"](now_t, raw_cy)

                jpx = estimate_jitter(pts, cx, cy, conf_thr=KP_CONF_THR)
                px, py  = prev_center.get(tid, (cx, cy))
                dist    = float(np.hypot(cx - px, cy - py))
                prev_center[tid] = (cx, cy)

                h = max(1.0, y2 - y1)
                thr_px = max(2.0, 2.5*jpx) * (h / 200.0)

                if dist <= thr_px:
                    if tid not in static_start_time:
                        static_start_time[tid] = time.monotonic()
                    static_sec = time.monotonic() - static_start_time[tid]
                else:
                    static_start_time.pop(tid, None)
                    static_sec = 0.0

                now_mono = time.monotonic()
                if dist <= MOVE_THRESH_PX:
                    if tid not in static_start_time:
                        static_start_time[tid] = now_mono
                    static_sec = now_mono - static_start_time[tid]
                else:
                    static_start_time.pop(tid, None)
                    static_sec = 0.0
                
                # ── 狀態決策 ────────────────────
                is_fallen, pose_metrics = fall_detector.detect(pts, tid, frame_count)
                
                if static_sec >= STATIC_TIME_S:
                    STATIONARY += 1
                    next_alert(int(tid),"STATIONARY")
                    status, color = "STATIONARY",  (0,   0, 255)   # 紅
                elif is_fallen :
                    FALL_LIKE += 1
                    next_alert(int(tid),"FALL-LIKE")
                    status, color = "FALL-LIKE",  (0,   255, 255)   #黃
                elif is_fallen and static_sec >= STATIC_TIME_S:
                    FALLEN += 1
                    next_alert(int(tid),"FALLEN")
                    status, color = "FALLEN",  (0,   0, 255)   # 紅
                else:
                    next_alert(int(tid),"STANDING")
                    status, color = "STANDING",(0, 255,   0)   # 綠
                
                # --- [修改] 更新 deque 資料 (所有 ID) ---
                if pose_metrics['valid']:
                    if tid not in plot_data_all_tids:
                        plot_data_all_tids[tid] = {
                            'angle': deque(maxlen=PLOT_MAX_LEN),
                            'vh_ratio': deque(maxlen=PLOT_MAX_LEN),
                            'leg_ratio': deque(maxlen=PLOT_MAX_LEN),
                            'fall_like': deque(maxlen=PLOT_MAX_LEN),
                            'age': 0
                        }
                    
                    plot_data_all_tids[tid]['angle'].append(pose_metrics['angle'])
                    plot_data_all_tids[tid]['vh_ratio'].append(pose_metrics['vh_ratio'])
                    plot_data_all_tids[tid]['leg_ratio'].append(pose_metrics['leg_ratio'])
                    plot_data_all_tids[tid]['fall_like'].append(1.0 if pose_metrics['fall_like'] else 0.0)
                    plot_data_all_tids[tid]['age'] = 0
                # --- [修改結束] ---


                print(f"CAM:{window_name} STATIONARY:{STATIONARY} FALL_LIKE:{FALL_LIKE} FALLEN:{FALLEN}")
                
                # ── 繪製框 / 骨架 / 關鍵點 ───────
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame,(x1,y1),(x2,y2), color, 2)
                
                if 'plot_skeleton_kpts' in globals():
                    plot_skeleton_kpts(frame, pts.reshape(-1), 3) # 傳入 1D array (51,)
                
                cv2.putText(frame, f"ID:{int(tid)} {status} {static_sec:.1f}s",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 8) #文字黑底
                cv2.putText(frame, f"ID:{int(tid)} {status} {static_sec:.1f}s",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2) #文字

        # ── 畫推論耗時 & 顯示 ──────────────────
        cv2.putText(frame, f"{window_name}  {proc_ms:.1f} ms frame_count:{frame_count} {now.strftime('%Y/%m/%d %H:%M:%S')}",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 8) #文字黑底
        cv2.putText(frame, f"{window_name}  {proc_ms:.1f} ms frame_count:{frame_count} {now.strftime('%Y/%m/%d %H:%M:%S')}",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)#文字

        cv2.imshow(window_name, frame)
        
        # --- [修改] 繪製並顯示 *多個* 即時圖表 ---
        ANGLE_RANGE = (0.0, 90.0)
        VH_RANGE    = (0.0, 3.0)   # <--- 如果您想放大, 改成 (0.5, 2.5)
        LEG_RANGE   = (0.0, 1.5)   # <--- 如果您想放大, 改成 (0.2, 1.0)
        FALL_RANGE  = (0.0, 1.0)
        for tid, data_deques in list(plot_data_all_tids.items()):
            
            # 2. 只更新本幀有出現的 ID (age == 0)
            if data_deques['age'] > 0:
                continue 

            # 3. 建立獨立的畫布
            plot_img = np.zeros((PLOT_IMG_H, PLOT_IMG_W, 3), dtype=np.uint8)
            color = get_color_for_tid(tid)
            window_name_plot = f"Debug Plot (TID: {tid}) - {stream.name}"

            _draw_curve(plot_img, data_deques['angle'], 
                        f"ID {tid} Angle", (0, 100, 255), *ANGLE_RANGE,
                        baselines=[ (55.0, "Angle Thresh") ])
            
            _draw_curve(plot_img, data_deques['vh_ratio'], 
                        f"ID {tid} V/H Ratio", (0, 255, 0), *VH_RANGE,
                        baselines=[ (1.5, "V/H (Vert)"), (0.8, "V/H (Horiz)") ])
            
            _draw_curve(plot_img, data_deques['leg_ratio'], 
                        f"ID {tid} Leg Ratio", (0, 200, 200), *LEG_RANGE,
                        baselines=[ (0.4, "Leg Thresh") ])

            # (Fall Like 狀態統一用紅色, 沒有基準線)
            _draw_curve(plot_img, data_deques['fall_like'], 
                        f"ID {tid} Fall Like", (0, 0, 255), *FALL_RANGE,
                        baselines=None) # <-- 傳遞 None


            # 5. 顯示此 ID 的專屬視窗
            cv2.imshow(window_name_plot, plot_img)
        
        if cv2.waitKey(DELAY_MS) & 0xFF == 27:   # ESC
            stop_event.set()
            break
    
    cv2.destroyWindow(window_name)
    
    # --- [修改] 關閉 *所有* 圖表視窗 ---
    for tid_key in plot_data_all_tids.keys():
        window_name_to_close = f"Debug Plot (TID: {tid_key}) - {stream.name}"
        try:
            cv2.destroyWindow(window_name_to_close)
        except cv2.error:
            pass
    # --- [修改結束] ---
    
    if hasattr(stream, "stop"): stream.stop()
    if hasattr(stream, "release"): stream.release()
    print(f"🛑 {window_name} 結束")

# ───────────────────── 6. 建立所有串流、依數量自動開線程 (保留不變) ─────────────────────
def build_streams():
    streams = []
    idx = 1
    if USE_RTSP:
        for i, url in enumerate(RTSP_URLS):
            streams.append(LatestFrameRTSP(url, f"CAM{i+1}"))
            idx += 1
    if USE_VIDEOS:
        for j, p in enumerate(VIDEO_FILES):
            name = f"VID{j+1}-{Path(p).stem}"
            streams.append(LatestFrameVideo(p, name))
            idx += 1
    if not streams:
        raise RuntimeError("沒有任何來源可用，請設定 RTSP_URLS 或 VIDEO_FILES。")
    return streams

def main():
    streams = build_streams()
    try:
        n_gpu = torch.cuda.device_count()
    except Exception:
        n_gpu = 0

    def device_for(k):
        if n_gpu >= 1:
            return f"cuda:{k % n_gpu}"
        else:
            return "cpu"

    threads = []
    for k, stream in enumerate(streams):
        dev = device_for(k)
        t = threading.Thread(target=worker, args=(stream, dev), daemon=True)
        threads.append(t)

    for t in threads: t.start()

    try:
        while True:
            if stop_event.is_set():
                break
            # 檢查是否有任何線程還在執行
            if not any(t.is_alive() for t in threads):
                print("[Main] 所有工作線程都已結束。")
                break
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[Main] 偵測到 Ctrl+C，正在要求所有線程停止...")
        stop_event.set()
    
    for t in threads: 
        t.join()

    cv2.destroyAllWindows()
    print("✅ 程式已完全結束")

# --- 新增：Letterbox 輔助函式 (來自 main.py) ---
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

if __name__ == "__main__":
    main()