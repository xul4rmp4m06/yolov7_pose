import pandas as pd
import matplotlib.pyplot as plt
import glob

# --- 設定 ---
LOG_FILE_PATTERN = "debug_log_*.csv"  # 讀取所有 debug_log_ 開頭的 CSV
TID_TO_PLOT = 0  # 您想專注分析的 ID
# --- 設定結束 ---

# 找到所有符合條件的檔案
log_files = glob.glob(LOG_FILE_PATTERN)
if not log_files:
    print(f"錯誤：在當前目錄下找不到任何 '{LOG_FILE_PATTERN}' 檔案。")
    exit()

print(f"找到 {len(log_files)} 個 檔案： {log_files}")

# 讀取並合併所有
df_list = [pd.read_csv(f) for f in log_files]
df = pd.concat(df_list)

# 過濾出您想分析的特定 ID
df_tid = df[df['tid'] == TID_TO_PLOT].copy()

if df_tid.empty:
    print(f"錯誤：在 ログ 檔案中找不到 ID = {TID_TO_PLOT} 的資料。")
    print(f"可用的 ID 有：{sorted(df['tid'].unique())}")
    exit()

print(f"正在為 ID = {TID_TO_PLOT} 繪製 {len(df_tid)} 筆資料...")

# 將 'fall_like' (True/False) 轉換為 1/0，以便繪圖
df_tid['fall_like_num'] = df_tid['fall_like'].astype(int)

# --- 繪圖 ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

# --- 圖 1：角度 (Angle) ---
ax1.plot(df_tid['frame'], df_tid['angle'], label='Angle (°)', color='blue', alpha=0.8)
ax1.set_ylabel('Angle (Degrees)')
ax1.set_title(f'Debug Metrics for TID: {TID_TO_PLOT} (File: {log_files[0]})')
ax1.legend(loc='upper left')
ax1.grid(True, linestyle='--', alpha=0.5)

# 建立第二個 Y 軸 (用於 fall_like)
ax1_twin = ax1.twinx()
ax1_twin.fill_between(df_tid['frame'], 0, df_tid['fall_like_num'], 
                      label='fall_like (True)', color='red', alpha=0.2, step='post')
ax1_twin.set_ylim(-0.1, 5) # 讓紅色區域高度不要太高
ax1_twin.set_yticks([])
ax1_twin.legend(loc='upper right')

# --- 圖 2：比例 (Ratios) ---
ax2.plot(df_tid['frame'], df_tid['vh_ratio'], label='V/H Ratio', color='green')
ax2.plot(df_tid['frame'], df_tid['leg_ratio'], label='Leg Ratio', color='orange')

# 標示出 V/H 和 LegRatio 的門檻
# (請確保這些值與您 main_new.py 中的 FallDetector 參數一致)
FALL_VH_THRESH = 1.5
LEG_RATIO_THRESH = 0.4
ax2.axhline(y=FALL_VH_THRESH, color='green', linestyle=':', label=f'V/H Thresh ({FALL_VH_THRESH})')
ax2.axhline(y=LEG_RATIO_THRESH, color='orange', linestyle=':', label=f'Leg Thresh ({LEG_RATIO_THRESH})')

ax2.set_xlabel('Frame Number')
ax2.set_ylabel('Ratio')
ax2.legend(loc='upper left')
ax2.grid(True, linestyle='--', alpha=0.5)

# 建立第二個 Y 軸 (用於 fall_like)
ax2_twin = ax2.twinx()
ax2_twin.fill_between(df_tid['frame'], 0, df_tid['fall_like_num'], 
                      label='fall_like (True)', color='red', alpha=0.2, step='post')
ax2_twin.set_ylim(-0.1, 5)
ax2_twin.set_yticks([])
ax2_twin.legend(loc='upper right')

# 顯示圖表
plt.tight_layout()
plt.show()