import mne
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import hilbert, butter, filtfilt
from collections import defaultdict
from mne.time_frequency import tfr_morlet
import warnings
warnings.filterwarnings("ignore")

# ==================== 1. 配置区 ====================
DATA_ROOT = "data"          # 你的数据根目录
OUTPUT_DIR = "results"      # 图片输出文件夹
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 核心映射规则（根据你的描述设定）
HAND_MAP = {
    'A': {
        'gestures': ['scissor', 'six', 'grasp'],
        'b_triggers': ['1', '4', '7']   # 对应上述顺序的 trigger b
    },
    'B': {
        'gestures': ['index', 'seven', 'thumb'],
        'b_triggers': ['1', '4', '7']   # 对应上述顺序的 trigger b
    }
}
ALL_GESTURE_NAMES = ['grasp', 'scissor', 'six', 'thumb', 'index', 'seven']
# ===================================================

# 存储所有数据
all_epochs_data = {name: [] for name in ALL_GESTURE_NAMES}

print("=" * 60)
print("开始遍历文件夹并提取数据...")

# 2. 遍历所有文件夹
for folder_name in sorted(os.listdir(DATA_ROOT)):
    folder_path = os.path.join(DATA_ROOT, folder_name)
    if not os.path.isdir(folder_path):
        continue
    
    if '-handA' in folder_name:
        hand_type = 'A'
    elif '-handB' in folder_name:
        hand_type = 'B'
    else:
        continue
    
    data_file = os.path.join(folder_path, 'data.bdf')
    evt_file = os.path.join(folder_path, 'evt.bdf')
    if not (os.path.exists(data_file) and os.path.exists(evt_file)):
        continue
    
    print(f"正在处理: {folder_name} (归类为 hand{hand_type})")
    
    try:
        raw = mne.io.read_raw_bdf(data_file, preload=True, verbose=False)
        annot = mne.read_annotations(evt_file)
        raw.set_annotations(annot)
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        raw_filtered = raw.copy().filter(1, 150, method='iir', verbose=False)
        
        gestures = HAND_MAP[hand_type]['gestures']
        b_triggers = HAND_MAP[hand_type]['b_triggers']
        
        for idx, gesture_name in enumerate(gestures):
            b_str = b_triggers[idx]
            if b_str not in event_id:
                continue
            b_code = event_id[b_str]
            events_b = events[events[:, 2] == b_code]
            if len(events_b) == 0:
                continue
            
            epochs_b = mne.Epochs(raw_filtered, events_b, event_id={'b': b_code},
                                  tmin=0, tmax=8, baseline=None,
                                  preload=True, verbose=False, proj=False)
            
            for ep_data in epochs_b.get_data():
                all_epochs_data[gesture_name].append(ep_data.astype(np.float32))
                
    except Exception as e:
        print(f"  跳过 {folder_name}，原因: {e}")
        continue

# 3. 打印统计
print("\n" + "=" * 60)
print("数据收集完毕，各手势试次数量：")
for name in ALL_GESTURE_NAMES:
    print(f"  {name:>10}: {len(all_epochs_data[name])} 个试次")
print("=" * 60)

# ================================================================
# 辅助函数：计算 High Gamma 能量
# ================================================================
def compute_hg_energy(data_list, sfreq=1000):
    all_energies = []
    b, a = butter(4, [50, 100], btype='band', fs=sfreq)
    for data in data_list:
        hg_energy_ch = []
        for ch in range(data.shape[0]):
            filtered = filtfilt(b, a, data[ch])
            envelope = np.abs(hilbert(filtered)) ** 2
            hg_energy_ch.append(envelope)
        avg_energy = np.mean(hg_energy_ch, axis=0)
        all_energies.append(avg_energy)
    all_energies = np.array(all_energies)
    return np.mean(all_energies, axis=0), np.std(all_energies, axis=0)

# ================================================================
# 任务A：High Gamma 能量时序图
# ================================================================
print("\n正在绘制 High Gamma 能量时序图...")

for gesture_name in ALL_GESTURE_NAMES:
    data_list = all_epochs_data[gesture_name]
    if len(data_list) == 0:
        print(f"  跳过 {gesture_name}，无数据")
        continue
    
    n_samples = data_list[0].shape[1]
    times = np.linspace(0, 8, n_samples)
    
    mean_energy, std_energy = compute_hg_energy(data_list)
    
    plt.figure(figsize=(12, 5))
    plt.plot(times, mean_energy, 'b-', linewidth=2, label='Mean High Gamma Power')
    plt.fill_between(times, mean_energy - std_energy, mean_energy + std_energy,
                     alpha=0.3, color='b', label='±1 SD')
    
    plt.axvline(x=0, color='k', linestyle='-', linewidth=1, label='Trigger b (Start)')
    plt.axvline(x=1.5, color='r', linestyle='--', linewidth=1.5, label='Flex Start')
    plt.axvline(x=4.0, color='orange', linestyle='--', linewidth=1.5, label='Hold Start')
    plt.axvline(x=5.5, color='g', linestyle='--', linewidth=1.5, label='Extend Start')
    plt.axvspan(0, 1.5, alpha=0.08, color='gray', label='Rest')
    plt.axvspan(1.5, 4.0, alpha=0.08, color='red', label='Flex')
    plt.axvspan(4.0, 5.5, alpha=0.08, color='orange', label='Hold')
    plt.axvspan(5.5, 8.0, alpha=0.08, color='green', label='Extend')
    
    plt.xlabel('Time (s)')
    plt.ylabel('High Gamma Power (a.u.)')
    plt.title(f'{gesture_name.capitalize()} (N={len(data_list)} trials) - High Gamma Energy')
    plt.legend(loc='upper right', fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'HighGamma_{gesture_name}.png'), dpi=150)
    plt.close()
    print(f"  已保存：{gesture_name} 的 High Gamma 图")

# ================================================================
# 任务B：ERSP 时频图（【手动基线校正，兼容所有 MNE 版本】）
# ================================================================
print("\n正在绘制 ERSP 时频图...")
freqs = np.arange(2, 100, 2)
n_cycles = freqs / 2.0

for gesture_name in ALL_GESTURE_NAMES:
    data_list = all_epochs_data[gesture_name]
    if len(data_list) == 0:
        continue

    all_data = np.array(data_list)
    ch_names = ['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8']
    info = mne.create_info(ch_names=ch_names, sfreq=1000, ch_types=['eeg']*8)
    epochs_temp = mne.EpochsArray(all_data, info, tmin=0, verbose=False)

    picks = mne.pick_types(epochs_temp.info, eeg=True)

    # 1. 计算时频图（不进行基线校正）
    power = tfr_morlet(epochs_temp, freqs=freqs, n_cycles=n_cycles,
                       use_fft=True, return_itc=False, average=True,
                       picks=picks, verbose=False)

    # 2. 手动计算百分比变化的 ERSP
    # power.data 形状: (n_channels, n_freqs, n_times)
    # 提取基线时间点索引：0 ~ 1.5 秒
    baseline_mask = (power.times >= 0) & (power.times <= 1.5)
    baseline_mean = power.data[:, :, baseline_mask].mean(axis=2, keepdims=True)  # (ch, freq, 1)
    
    # 避免除以零
    baseline_mean[baseline_mean == 0] = 1e-12
    
    # 百分比变化: (值 - 基线) / 基线 * 100
    data_percent = (power.data - baseline_mean) / baseline_mean * 100.0
    data_percent = np.nan_to_num(data_percent, nan=0.0, posinf=0.0, neginf=0.0)

    # 对所有通道取平均，得到 (n_freqs, n_times)
    data_avg = data_percent.mean(axis=0)

    if np.max(np.abs(data_avg)) < 1e-6:
        print(f"  警告：{gesture_name} 的 ERSP 数据全为零，跳过绘图")
        continue

    # 3. 动态自适应颜色范围（使用 5%~95% 分位数，并强制对称）
    vmin = np.percentile(data_avg, 5)
    vmax = np.percentile(data_avg, 95)
    # 对称化
    abs_max = max(abs(vmin), abs(vmax))
    if abs_max < 0.1:
        abs_max = 50.0  # 兜底
    vmin, vmax = -abs_max, abs_max

    times = power.times
    plt.figure(figsize=(12, 6))
    im = plt.imshow(data_avg, aspect='auto', origin='lower',
                    extent=[times[0], times[-1], freqs[0], freqs[-1]],
                    cmap='RdBu_r', vmin=vmin, vmax=vmax)

    # 阶段分割线
    plt.axvline(x=1.5, color='k', linestyle='--', linewidth=1.5, label='Flex Start')
    plt.axvline(x=4.0, color='gray', linestyle='--', linewidth=1, label='Hold Start')
    plt.axvline(x=5.5, color='gray', linestyle='--', linewidth=1, label='Extend Start')

    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title(f'{gesture_name.capitalize()} ERSP (N={len(data_list)} trials)')
    cbar = plt.colorbar(im, label='Power Change (%)')
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'ERSP_{gesture_name}.png'), dpi=150)
    plt.close()
    print(f"  已保存：{gesture_name} 的 ERSP 图 (颜色范围: {vmin:.2f}% ~ {vmax:.2f}%)")

print("\n" + "=" * 60)
print("🎉 全部完成！请查看 results 文件夹中的 12 张图片。")
print("=" * 60)