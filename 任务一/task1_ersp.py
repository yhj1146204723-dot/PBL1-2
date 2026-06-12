import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from dataloaders.loader import load_neuracle
from dataloaders import neo

# =================================================================
# 1. 参数设置
# =================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(script_dir, 'data', 'single-movement')

event_id = {'Left Grasp': 5, 'Right Grasp': 7}
tmin, tmax = -1.5, 3.5      
baseline = (-1.0, 0.0)      
freqs = np.arange(5, 200, 2)  # 更密集的频率采样，让谱曲线和时频图更平滑

tfr_left_all = []
tfr_right_all = []

# =================================================================
# 2. 数据循环读取与跨 Session 平均
# =================================================================
folders = [f for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f))]
print(f"开始处理数据，共发现 {len(folders)} 个数据文件夹...")

for folder in folders:
    data_path = os.path.join(root_path, folder)
    try:
        raw = load_neuracle(data_path, 'ecog')
        raw = neo.preprocessing(raw, reref_method='average')
        raw.set_channel_types({ch: 'ecog' for ch in raw.ch_names})
        events, _ = mne.events_from_annotations(raw, {'5': 5, '7': 7})
        epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax, baseline=None, preload=True)
        
        # 计算对侧左手抓握的 TFR
        if 'Left Grasp' in epochs.event_id:
            tfr_left = mne.time_frequency.tfr_multitaper(
                epochs['Left Grasp'], freqs=freqs, n_cycles=freqs/2, 
                return_itc=False, average=True, decim=2
            )
            tfr_left.apply_baseline(baseline, mode="logratio")  # 严格对标文献的对数比值法
            tfr_left_all.append(tfr_left)
            
        # 计算同侧右手抓握的 TFR
        if 'Right Grasp' in epochs.event_id:
            tfr_right = mne.time_frequency.tfr_multitaper(
                epochs['Right Grasp'], freqs=freqs, n_cycles=freqs/2, 
                return_itc=False, average=True, decim=2
            )
            tfr_right.apply_baseline(baseline, mode="logratio")
            tfr_right_all.append(tfr_right)
            
        print(f"成功处理 Session: {folder}")
    except Exception as e:
        print(f"跳过出错 Session {folder}: {e}")
        continue

# 执行跨 Session 的全局平均
print("正在计算跨 Session 平均...")
avg_tfr_left = tfr_left_all[0].copy()
avg_tfr_left.data = np.mean([tfr.data for tfr in tfr_left_all], axis=0)

avg_tfr_right = tfr_right_all[0].copy()
avg_tfr_right.data = np.mean([tfr.data for tfr in tfr_right_all], axis=0)

# 提取画图通用的时间与频率坐标轴
times = avg_tfr_left.times
freq_array = avg_tfr_left.freqs

# =================================================================
# 3. 绘图与保存: Fig_1c_Time_Frequency.jpg (8通道对侧时频图)
# =================================================================
print("正在生成图 1c (8通道对侧时频图)...")
left_data_matrix = avg_tfr_left.data  # 形状: (8, n_freqs, n_times)

fig, axes = plt.subplots(4, 2, figsize=(10, 12), sharex=True, sharey=True)
axes = axes.flatten()

for i in range(8):
    ax = axes[i]
    # 使用 pcolormesh 绘制，vmin/vmax 缩放到真实范围 -0.4~0.4 以增强对比度
    c = ax.pcolormesh(times, freq_array, left_data_matrix[i], cmap='RdBu_r', vmin=-0.4, vmax=0.4, shading='auto')
    ax.axvline(0, color='k', linestyle='--', linewidth=1)  # t=0 虚线
    ax.set_title(f"ch {i+1}")
    if i >= 6: ax.set_xlabel("Time (s)")
    if i % 2 == 0: ax.set_ylabel("Frequency (Hz)")

# 配置右侧公共颜色条
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
fig.colorbar(c, cax=cbar_ax, label="ERSP (dB)")
fig.suptitle("Left Hand MI vs Rest (Time-Frequency Maps)", fontsize=16, y=0.95)

# 保存图 1c
save_path_1c = os.path.join(script_dir, "Fig_1c_Time_Frequency.jpg")
plt.savefig(save_path_1c, dpi=300, bbox_inches='tight')
print(f"【成功】图 1c 已保存至: {save_path_1c}")

# =================================================================
# 4. 绘图与保存: Fig_1d_Left_Right_Contrast.png (左右手对比频率曲线)
# =================================================================
print("正在生成图 1d (左右手对比频率曲线)...")
plt.figure(figsize=(9, 5))

# 锁定动作维持期时间窗 (0.0s - 2.5s)
move_mask = (times >= 0.0) & (times <= 2.5)

# 严格对标文献逻辑：选择右脑核心运动皮层代表性电极 ch3 (索引为 2)
best_motor_ch = 2 

# 沿时间轴求平均，将二维降为一维随频率变化的曲线
ersp_left_freq = np.mean(avg_tfr_left.data[best_motor_ch][:, move_mask], axis=1)
ersp_right_freq = np.mean(avg_tfr_right.data[best_motor_ch][:, move_mask], axis=1)

# 绘制经典神经震荡频段阴影背景
plt.axvspan(8, 12, color='blue', alpha=0.1, label=r'$\alpha$ (8-12 Hz)')
plt.axvspan(13, 30, color='cyan', alpha=0.1, label=r'$\beta$ (13-30 Hz)')
plt.axvspan(30, 50, color='green', alpha=0.1, label=r'low $\gamma$ (30-50 Hz)')
plt.axvspan(50, 150, color='red', alpha=0.1, label=r'high $\gamma$ (50-150 Hz)')

# 绘制双线对比
plt.plot(freq_array, ersp_left_freq, color='darkorange', linewidth=2.5, label='Left Hand Grasp (Contralateral eECoG)')
plt.plot(freq_array, ersp_right_freq, color='dodgerblue', linewidth=2.5, label='Right Hand Grasp (Ipsilateral eECoG)')
plt.axhline(0, color='grey', linestyle='-', linewidth=1.5)

# 坐标轴与图例美化
plt.xlim(0, 200)
plt.ylim(-0.5, 0.5)
plt.xlabel("Frequency (Hz)")
plt.ylabel("ERSP (dB / Log-ratio)")
plt.title("ERSP vs Frequency: Contralateral vs Ipsilateral Grasp (Channel 3)")
plt.legend(loc='upper right')
plt.grid(True, alpha=0.2)

# 保存图 1d
save_path_1d = os.path.join(script_dir, "Fig_1d_Left_Right_Contrast.png")
plt.savefig(save_path_1d, dpi=300, bbox_inches='tight')
print(f"【成功】图 1d 已保存至: {save_path_1d}")

# =================================================================
# 5. 集中展现所有窗口
# =================================================================
plt.show()