#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[YOLOv7-Pose 整合版 v2 - 無 Ultralytics 依賴]
Multi-RTSP/Video real-time pose tracking:
  • 每個來源各開一個線程
  • 每個線程各自載入 YOLOv7-Pose 模型 (attempt_load)
  • 每個線程各自載入一個 *自訂的* SimpleCentroidTracker 追蹤器
  • 保留所有跌倒偵測、靜止偵測、Discord 功能
"""

import os, cv2, time, threading, numpy as np
import math
from datetime import datetime, timezone, timedelta
from collections import deque
import requests,json
from pathlib import Path

# --- 新增：YOLOv7-Pose 相關 imports ---
import torch
from models.experimental import attempt_load
from utils.torch_utils import select_device
from utils.general import non_max_suppression_kpt
try:
    from utils.plots import plot_skeleton_kpts, plot_one_box
except ImportError:
    print("警告：無法從 utils.plots 導入 plot_skeleton_kpts 或 plot_one_box。")
    # 如果導入失敗，您可能需要從 Wjdepp/yolov7-pose 倉庫中複製這些函式

# ───────────────────── 0. 低延遲 FFmpeg 參數 ─────────────────────
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
    r"C:\Users\User\SynologyDrive\py\yoloV7_pose\1009-1.mp4",
]

# ───────────────────── 2. 背景取流 (保留不變) ─────────────────────
class LatestFrameRTSP:
    """
    背景 Thread 持續 read()，只存最新 frame
    get() 隨時回傳複製後的最新 frame（None 表示尚未拿到）
    """
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
    """
    影片逐幀讀取；回傳 None 代表已播放結束
    """
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
MODEL_DIR         = "yolov7-w6-pose.pt"     # *** 修改：使用 YOLOv7 模型 ***
RAWIMGSZ          = 1920                      # *** 修改：使用 main.py 的 1920 尺寸 ***
PLAYBACK_FPS      = 20.0                      # UI 顯示節流
DELAY_MS          = int(1000 / PLAYBACK_FPS)
MOVE_THRESH_PX    = 4                         # 移動 ≤3 px 視為靜止
STATIC_TIME_S     = 240.0                     # 靜止 ≥5 秒標紅
CONF_THR          = 0.25                      # 可信度
IOU_THR           = 0.45                      # *** 修改：使用 main.py 的 0.45 ***
BOX_CONF_THR      = 0.50                      #偵測框信心度閾值
KP_CONF_THR       = 0.5                       #0.8 關鍵點可信度
MIN_KP_OK         = 6
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
    # ... (您原有的 FallDetector 類別程式碼 ... 保持不變) ...
    # ... (為了簡潔，此處省略，請貼上您原有的完整 FallDetector 類別) ...
    def __init__(
        self,
        # --- 事件式偵測參數 (偵測"過程") ---
        fps=10,
        drop_sigma_k=0.5,
        post_motion_eps=3.0,
        post_flat_vh=0.85,
        fall_cooldown_s=2.0,
        
        # --- 狀態機參數 (偵測"狀態") ---
        # 垂直跌倒/癱倒的V/H比例門檻 (站立時通常>2.5，癱坐時<2.0)
        vertical_fall_vh_thresh=1.5, 
        # 垂直跌倒的角度門檻
        vertical_angle_deg_thresh = 20,
        # 水平跌倒的角度門檻
        angle_deg_thresh=55.0,
        # 水平跌倒的V/H比例門檻
        horizontal_fall_vh_thresh=0.80,
        # 腿部壓縮比例門檻 (腿長/身高)
        leg_ratio_thresh=0.4,
        
        # 連續幀數確認
        frames_for_static_fall=60, 
        frames_to_alert=60,
        
        # --- 通用參數 ---
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

        # 事件式偵測用的滑動視窗
        win = max(5, int(self.fps * 1.0))
        self._hip_y_hist = {}
        self._vh_hist = {}
        self._centroid_hist = {}
        self._last_fall_time = {}
        self._win = win
    
    def _update_fall_event(self, tid, pts, ts):
        """
        [事件式偵測] 偵測 '站→倒' 過程 (此函式邏輯大致保留)
        回傳 True 表示觸發一次跌倒事件
        """
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
        判斷單一幀的姿勢是否像跌倒，能同時處理水平和垂直跌倒。
        """
        P = np.asarray(pts, dtype=float)
        mask = P[:, 2] >= self.conf_thr
        
        # [修改] 至少需要6個點才能進行可靠的比例計算
        if mask.sum() < 6:
            return False

        try:
            # --- 1. 計算所有需要的指標 ---
            
            # 1a. 身體主軸角度
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
            
            # 1b. V/H 比例
            xy = P[mask, :2]
            xs, ys = xy[:, 0], xy[:, 1]
            H_spread = max(1.0, xs.max() - xs.min())
            V_spread = max(1.0, ys.max() - ys.min())
            vh_ratio = V_spread / H_spread
            
            # 1c. 腿部壓縮比例
            p_ank_mid = (P[15, :2] + P[16, :2]) * 0.5
            p_nose = P[0, :2]
            leg_vertical_span = abs(p_hip_mid[1] - p_ank_mid[1])
            body_height_approx = abs(p_nose[1] - p_ank_mid[1])
            leg_compression_ratio = leg_vertical_span / body_height_approx if body_height_approx > 1e-6 else 1.0

            # --- 2. 綜合判斷 ---

            # [修改] 條件一：水平跌倒 (角度大 且 身體扁平)
            cond_horizontal_fall = (angle_deg >= self.angle_deg_thresh and 
                                    vh_ratio <= self.horizontal_fall_vh_thresh)

            # [修改] 條件二：垂直跌倒/癱倒 (角度小 但 身體被壓縮)
            # 身體被壓縮可以由 V/H比例 或 腿部比例 來判斷
            cond_vertical_fall = (angle_deg < self.angle_deg_thresh and 
                                 (vh_ratio <= self.vertical_fall_vh_thresh or leg_compression_ratio <= self.leg_ratio_thresh))
            
            cond_vertical_angle_fall = (angle_deg >= self.vertical_angle_deg_thresh and 
                                        angle_deg <= self.vertical_angle_deg_thresh +3)
            
            fall_like = cond_horizontal_fall or cond_vertical_fall #or cond_vertical_angle_fall

            if self.debug and hasattr(self, '_is_debugging_tid') and self._is_debugging_tid:
                print(
                    f"    [Static Pose Check] Angle={angle_deg:.1f}°, V/H={vh_ratio:.2f}, LegRatio={leg_compression_ratio:.2f}\n"
                    f"    -> H_Fall(angle>{self.angle_deg_thresh}, vh<{self.horizontal_fall_vh_thresh})={cond_horizontal_fall} | "
                    f"V_Fall(angle<{self.angle_deg_thresh}, vh<{self.vertical_fall_vh_thresh} or leg<{self.leg_ratio_thresh})={cond_vertical_fall} | "
                    f"fall_like={fall_like}"
                )
            return fall_like

        except IndexError:
            return False

    def detect(self, pts, tid, frame_num):
        """主偵測函式，結合事件式偵測與狀態機。"""
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
        
        try:
            hip_mid_y = (P[11, 1] + P[12, 1]) * 0.5
        except IndexError:
            history['state'] = 'NORMAL'
            history['last_hip_y'] = None
            return False

        vertical_speed = 0.0
        if history['last_hip_y'] is not None:
            delta_frames = frame_num - history['last_frame_num']
            if delta_frames > 0:
                delta_y = hip_mid_y - history['last_hip_y']
                vertical_speed = delta_y / delta_frames
        
        history['last_hip_y'] = hip_mid_y
        history['last_frame_num'] = frame_num
        
        # --- [修改] 使用整合後的姿勢判斷函式 ---
        is_in_fallen_pose = self._check_static_pose(P)
        
        # 事件式偵測作為補充
        ts = frame_num / max(1e-6, float(self.fps))
        event_trigger = self._update_fall_event(tid, P, ts)

        # 只要事件觸發 或 連續處於倒地姿勢，就認為 "is_on_ground"
        is_on_ground = is_in_fallen_pose or event_trigger
        
        # --- 狀態機邏輯 (與您上一版相同，非常穩健) ---
        new_state = current_state
        if current_state == 'NORMAL':
            if vertical_speed > 15.0: # 簡易的速度門檻，事件式偵測更可靠
                new_state = 'FALLING'
                history['static_fall_pose_count'] = 0
            elif is_in_fallen_pose: # [修改] 只用靜態姿勢來觸發計數
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
        if self.debug:
           print(
                f"[fall][tid={tid}][frame={frame_num}] [time={now.strftime('%Y/%m/%d %H:%M:%S')}] "
                f"state: {current_state:>7s} → {new_state:>7s} | "
                f"alert_count={history['fallen_count']}/{self.frames_to_alert} | "
                f"static_pose_count={history['static_fall_pose_count']}/{self.frames_for_static_fall}"
                "\n-----------------------------------------------------------------------------------"
            )
        self._is_debugging_tid = False
        # [修改] 事件觸發時，也直接回傳True (立即警報)
        if event_trigger:
            print(f"!!! EVENT TRIGGERED for tid={tid} !!!")
            return True
        return new_state == 'ALERT'
fall_detector = FallDetector()

def get_stable_center(P, conf_thr=0.6):
    # P: (17,3) [x,y,conf]
    ok = P[:,2] >= conf_thr
    # 髖中點與肩中點（有缺就退而求其次）
    cands = []
    if ok[11] and ok[12]:
        hip_mid = ( (P[11,0] + P[12,0]) * 0.5, (P[11,1] + P[12,1]) * 0.5 )
        cands.append(("hip", hip_mid))
    if ok[5] and ok[6]:
        sho_mid = ( (P[5,0] + P[6,0]) * 0.5, (P[5,1] + P[6,1]) * 0.5 )
        cands.append(("sho", sho_mid))

    if not cands:
        # 退化：用所有高信心點的質心
        if ok.any():
            xy = P[ok,:2]
            return float(xy[:,0].mean()), float(xy[:,1].mean())
        else:
            # 再退化：鼻子
            return float(P[0,0]), float(P[0,1])

    # 加權平均：髖 0.7、肩 0.3
    w = {"hip":0.7, "sho":0.3}
    sx = sy = sw = 0.0
    for tag, (x,y) in cands:
        ww = w[tag]
        sx += ww*x; sy += ww*y; sw += ww
    return sx/sw, sy/sw

class OneEuro:
    # ... (您原有的 OneEuro 類別 ... 保持不變) ...
    # 簡化版 One Euro Filter：用在單維
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
        # 更新實際頻率
        dt = max(1e-6, t - self.t_prev)
        self.freq = 1.0 / dt
        # 導數濾波
        dx = (x - self.x_prev) * self.freq
        a_d = self._alpha(self.d_cutoff)
        dx_hat = a_d*dx + (1-a_d)*self.dx_prev
        # 主濾波 cutoff 隨動態調整
        cutoff = self.min_cutoff + self.beta*abs(dx_hat)
        a = self._alpha(cutoff)
        x_hat = a*x + (1-a)*self.x_prev
        # 狀態更新
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

def estimate_jitter(P, cx, cy, conf_thr=0.6):
    ok = P[:,2] >= conf_thr
    if ok.sum() < 4:
        return 2.0  # 缺點時給個保守值
    xy = P[ok,:2]
    # 關鍵點相對中心的半徑
    r = np.hypot(xy[:,0]-cx, xy[:,1]-cy)
    med = np.median(r)
    mad = np.median(np.abs(r - med)) + 1e-6
    # 用 MAD 當做該幀的抖動基準（你也可以乘個係數）
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
            # 為每個舊 track 找到最近的新 det
            matched_new_indices = set()
            unmatched_track_tids = set(existing_tids)

            # 找出最佳匹配對 (距離最近的)
            matches = []
            for t_idx, tid in enumerate(existing_tids):
                if len(new_centroids) == 0: break # 沒有可匹配的了
                best_dist = np.inf
                best_d_idx = -1
                
                for d_idx in range(len(new_centroids)):
                    if d_idx in matched_new_indices:
                        continue # 這個 det 已經被匹配過了
                    
                    d = dist_matrix[t_idx, d_idx]
                    if d < best_dist:
                        best_dist = d
                        best_d_idx = d_idx
                
                if best_dist < self.max_dist_px:
                    matches.append((tid, best_d_idx))
                    matched_new_indices.add(best_d_idx)
                    unmatched_track_tids.remove(tid)

            # 4. 更新匹配上的 Track
            for tid, d_idx in matches:
                det = new_detections[d_idx]
                self.tracks[tid]['centroid'] = det['centroid']
                self.tracks[tid]['kpts'] = det['kpts']
                self.tracks[tid]['box'] = det['box']
                self.tracks[tid]['conf'] = det['conf']
                self.tracks[tid]['age'] = 0
                
                # 加入到回傳列表
                active_tracks_output.append((
                    tid, 
                    det['kpts'], 
                    det['conf'], 
                    det['box']
                ))

            # 5. 註冊未匹配上的 Detections (新目標)
            for d_idx in range(len(new_detections)):
                if d_idx not in matched_new_indices:
                    det = new_detections[d_idx]
                    new_tid = self._register(det['centroid'], det['kpts'], det['box'], det['conf'])
                    # (新註冊的目標也立刻回傳)
                    active_tracks_output.append((
                        new_tid, 
                        det['kpts'], 
                        det['conf'], 
                        det['box']
                    ))

            # 6. 處理未匹配上的 Tracks (舊目標)
            for tid in unmatched_track_tids:
                self.tracks[tid]['age'] += 1
                if self.tracks[tid]['age'] > self.max_age:
                    self._deregister(tid)
        
        elif len(new_detections) == 0:
             # 沒有任何偵測，所有 track 都 +1 age
             for tid in existing_tids:
                self.tracks[tid]['age'] += 1
                if self.tracks[tid]['age'] > self.max_age:
                    self._deregister(tid)
        
        return active_tracks_output


# ───────────────────── 5. 每線程 worker：(修改為 YOLOv7 + 自訂追蹤器) ─────────────────────
def worker(stream, device="cuda:0"):
    """
    每個線程都各自載入一顆 YOLOv7 模型 + SimpleCentroidTracker 追蹤器
    """
    # 線程專屬狀態
    STATIONARY = 0
    FALLEN = 0
    FALL_LIKE = 0
    ALERT_IDs = {}
    TZ = timezone(timedelta(hours=8), name="Asia/Taipei") #台北時間
    center_filters = {}   # tid -> {"fx":OneEuro(...), "fy":OneEuro(...)}
    jitter_hist = {}      # tid -> deque 最近抖動估計
    
    # --- Discord Alert 相關函式 (保留不變) ---
    def next_alert(tid: str,alert_type: str) -> dict:
        # ... (與您 v8 腳本中相同, 此處省略) ...
        now = datetime.now(TZ)
        slot = ALERT_IDs.get(tid)
        if slot is None: slot = {"seq_next": 0, "records": []}
        seq = slot["seq_next"]; slot["seq_next"] += 1
        rec = {"tid": tid,"seq": seq,"cam_name": window_name,"alert_type":alert_type,"time":  now.strftime("%Y/%m/%d %H:%M:%S"),"report": False}
        slot["records"].append(rec); ALERT_IDs[tid] = slot
        return rec
    
    # --- 過濾器 (保留不變) ---
    def pass_filters(box_conf: float, kp_conf: np.ndarray) -> bool:
        return (box_conf >= BOX_CONF_THR) and ((kp_conf >= KP_CONF_THR).sum() >= MIN_KP_OK)
    
    # --- 修改：初始化 YOLOv7 模型 ---
    print(f"[{stream.name}] 正在載入 YOLOv7 模型 {MODEL_DIR} 到 {device}...")
    device_obj = select_device(device)
    model = attempt_load(MODEL_DIR, map_location=device_obj)
    stride = int(model.stride.max())  # 對 P6 會是 64
    model.eval()
    half = (device_obj.type != 'cpu')
    if half:
        model.half()
    
    def align_to_stride(x: int, s: int) -> int:
        return (x + s - 1) // s * s
    
    raw_imgsz = RAWIMGSZ 
    imgsz = align_to_stride(raw_imgsz, stride)

    # --- 修改：初始化 自訂追蹤器 ---
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
        
        # --- 核心修改：YOLOv7 推論流程 ---
        
        # 1. v7 Pre-processing
        img, ratio, (dw, dh) = letterbox(frame, imgsz, stride=stride)
        img_for_torch = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
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
            
            # 建立一個包含所有 'x' 座標索引的列表 (0, 2, 6, 9, 12, ...)
            # 假設有 17 個關鍵點 (17*3=51)，總共 6+51=57 欄
            all_x_indices = [0, 2] + list(range(6, 57, 3))
            # 建立一個包含所有 'y' 座標索引的列表 (1, 3, 7, 10, 13, ...)
            all_y_indices = [1, 3] + list(range(7, 57, 3))

            # 使用這個有效的列表來進行索引和 clamping
            det[:, all_x_indices] = det[:, all_x_indices].clamp(0, w)
            det[:, all_y_indices] = det[:, all_y_indices].clamp(0, h)
            
            det_np_scaled = det.cpu().numpy()
        
        # 5. 更新 自訂追蹤器
        #    matched_results: list of (tid, kpts_array, box_conf, box_xyxy)
        matched_results = tracker.update(det_np_scaled)

        # 6. 保留原有的 v8 腳本邏輯 (跌倒/靜止偵測)
        if len(matched_results) > 0:
            for (tid, kpts_array, box_conf, (x1,y1,x2,y2)) in matched_results:
                
                pts = kpts_array # (17, 3) numpy array
                
                # 執行 v8 腳本的過濾器
                kp_conf = pts[:, 2] # (17,)
                if not pass_filters(box_conf, kp_conf):
                    continue
                
                # --- (*** 以下是從 v8 腳本中完整複製的邏輯 ***) ---
                
                # ── 計算移動距離 ─────────────────
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
                is_fallen = fall_detector.detect(pts, tid, frame_count)
                if static_sec >= STATIC_TIME_S:
                    STATIONARY += 1
                    next_alert(int(tid),"STATIONARY")
                    status, color = "STATIONARY",  (0,   0, 255)   # 紅 人員禁止不動 觸發警報
                elif is_fallen :
                    FALL_LIKE += 1
                    next_alert(int(tid),"FALL-LIKE")
                    status, color = "FALL-LIKE",  (0,   255, 255)   #黃 人員疑似倒地
                elif is_fallen and static_sec >= STATIC_TIME_S:
                    FALLEN += 1
                    next_alert(int(tid),"FALLEN")
                    status, color = "FALLEN",  (0,   0, 255)   # 紅 人員疑似倒地、禁止不動 觸發警報
                else:
                    next_alert(int(tid),"STANDING")
                    status, color = "STANDING",(0, 255,   0)   # 綠 人員正常移動
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
        if cv2.waitKey(DELAY_MS) & 0xFF == 27:   # ESC
            stop_event.set()
            break
    
    cv2.destroyWindow(window_name)
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
    # 建立串流
    streams = build_streams()

    # 自動偵測 GPU，輪詢分配 device
    try:
        import torch
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

    # 起線程
    for t in threads: t.start()

    # 等待全部結束
    try:
        while True:
            if stop_event.is_set():
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