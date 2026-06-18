# -*- coding: utf-8 -*-
"""
PBL 任务1：手部抓握运动神经活动分析
严格按作业 PDF：尝试抓握阶段 vs 静息阶段（Rest）计算 ERSP。

作业范式：
- trigger a：动作开始，动作尝试持续 2.5 s
- trigger b：进入休息，静息持续 2.0 s
- 左手握拳：trigger a = 5, trigger b = 6
- 右手握拳：trigger a = 7, trigger b = 8

本脚本的核心修改：
1. 不再用动作前 -1~0 s 作为 baseline。
2. 分别读取 trigger 6 / trigger 8 后 0~2 s 的 Rest 段作为静息功率。
3. ERSP 按 10*log10(P_move / P_rest) 计算，因此单位可写 dB。
"""

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
root_path = os.path.join(script_dir, "data", "single-movement")

# 作业 PDF 中的 single-movement trigger 编号
# 左手握拳：动作开始 5，进入静息 6；右手握拳：动作开始 7，进入静息 8
EVENT_CODE = {
    "Left Move": 5,
    "Left Rest": 6,
    "Right Move": 7,
    "Right Rest": 8,
}

# 为了画时频图，动作 epoch 保留动作前后较长时间窗；真正的 ERSP 分母来自 Rest epoch
move_tmin, move_tmax = -1.5, 3.5
rest_tmin, rest_tmax = 0.0, 2.0

# 动作阶段：PDF 写明动作尝试持续 2.5 s
move_window = (0.0, 2.5)

# 频率范围。任务图展示到约 200 Hz；后续 high gamma 可重点看 50-100 / 50-150 Hz
freqs = np.arange(5, 200, 2)
n_cycles = freqs / 2

decim = 2

# 如果觉得时频图边缘有卷积效应，可把 epoch 加长后再 crop；这里先按图展示窗口绘制
# color scale 可根据实际结果微调
vmin, vmax = -4.0, 4.0   # dB；若图太淡可改为 -2, 2 或 -3, 3


def _safe_logratio_db(move_power, rest_power):
    """计算 10*log10(move/rest)，避免除零。"""
    eps = np.finfo(float).eps
    return 10.0 * np.log10(np.maximum(move_power, eps) / np.maximum(rest_power, eps))


def compute_tfr_vs_rest(raw, events, move_code, rest_code, condition_name):
    """
    对一个条件计算 ERSP：
    - move_code 对应 trigger a 后的动作尝试阶段；
    - rest_code 对应 trigger b 后的静息阶段；
    - 输出的 TFR 以 move trigger 为 0 s 对齐，但数值已经是相对 Rest 的 dB。
    """
    event_codes_in_data = set(events[:, 2].tolist())
    if move_code not in event_codes_in_data or rest_code not in event_codes_in_data:
        print(f"  [跳过] {condition_name}: 缺少 trigger {move_code} 或 {rest_code}")
        return None

    epochs_move = mne.Epochs(
        raw,
        events,
        event_id={f"{condition_name} Move": move_code},
        tmin=move_tmin,
        tmax=move_tmax,
        baseline=None,
        preload=True,
        reject_by_annotation=True,
        verbose=False,
    )

    epochs_rest = mne.Epochs(
        raw,
        events,
        event_id={f"{condition_name} Rest": rest_code},
        tmin=rest_tmin,
        tmax=rest_tmax,
        baseline=None,
        preload=True,
        reject_by_annotation=True,
        verbose=False,
    )

    if len(epochs_move) == 0 or len(epochs_rest) == 0:
        print(f"  [跳过] {condition_name}: move/rest epoch 数为 0")
        return None

    # 先分别计算 move 和 rest 的平均功率谱，而不是用动作前 baseline
    tfr_move = mne.time_frequency.tfr_multitaper(
        epochs_move,
        freqs=freqs,
        n_cycles=n_cycles,
        return_itc=False,
        average=True,
        decim=decim,
        verbose=False,
    )

    tfr_rest = mne.time_frequency.tfr_multitaper(
        epochs_rest,
        freqs=freqs,
        n_cycles=n_cycles,
        return_itc=False,
        average=True,
        decim=decim,
        verbose=False,
    )

    # Rest power：对 trigger b 后 0~2 s 的静息阶段做时间平均
    # shape: (n_channels, n_freqs, 1)，便于和 move 的 (n_channels, n_freqs, n_times) 广播相除
    rest_power_mean = np.mean(tfr_rest.data, axis=2, keepdims=True)

    # ERSP = 10 * log10(P_move / P_rest)，单位 dB
    tfr_move.data = _safe_logratio_db(tfr_move.data, rest_power_mean)

    print(
        f"  [完成] {condition_name}: move trigger={move_code}, rest trigger={rest_code}, "
        f"move epochs={len(epochs_move)}, rest epochs={len(epochs_rest)}"
    )
    return tfr_move


def average_tfr_list(tfr_list, name):
    """跨 session 平均 TFR。"""
    if len(tfr_list) == 0:
        raise RuntimeError(f"没有成功得到 {name} 的 TFR，请检查数据路径和 trigger 编号。")
    avg = tfr_list[0].copy()
    avg.data = np.mean([tfr.data for tfr in tfr_list], axis=0)
    return avg


# =================================================================
# 2. 数据循环读取与跨 session 平均
# =================================================================
tfr_left_all = []
tfr_right_all = []

folders = sorted(
    [f for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f))]
)
print(f"开始处理数据，共发现 {len(folders)} 个数据文件夹...")

for folder in folders:
    data_path = os.path.join(root_path, folder)
    print(f"\n正在处理 Session: {folder}")
    try:
        raw = load_neuracle(data_path, "ecog")
        raw = neo.preprocessing(raw, reref_method="average")
        raw.set_channel_types({ch: "ecog" for ch in raw.ch_names})

        # 读取作业 PDF 中涉及的 5/6/7/8 四类 trigger
        events, _ = mne.events_from_annotations(
            raw,
            event_id={"5": 5, "6": 6, "7": 7, "8": 8},
            verbose=False,
        )

        tfr_left = compute_tfr_vs_rest(
            raw,
            events,
            move_code=EVENT_CODE["Left Move"],
            rest_code=EVENT_CODE["Left Rest"],
            condition_name="Left Grasp",
        )
        if tfr_left is not None:
            tfr_left_all.append(tfr_left)

        tfr_right = compute_tfr_vs_rest(
            raw,
            events,
            move_code=EVENT_CODE["Right Move"],
            rest_code=EVENT_CODE["Right Rest"],
            condition_name="Right Grasp",
        )
        if tfr_right is not None:
            tfr_right_all.append(tfr_right)

    except Exception as e:
        print(f"  [出错跳过] Session {folder}: {e}")
        continue

print("\n正在计算跨 Session 平均...")
avg_tfr_left = average_tfr_list(tfr_left_all, "Left Grasp")
avg_tfr_right = average_tfr_list(tfr_right_all, "Right Grasp")

times = avg_tfr_left.times
freq_array = avg_tfr_left.freqs

# =================================================================
# 3. 图 1c：左手抓握 vs Rest 的 8 通道时频图
# =================================================================
print("正在生成图 1c：Left Hand Grasp vs Rest 时频图...")
left_data_matrix = avg_tfr_left.data

fig, axes = plt.subplots(4, 2, figsize=(10, 12), sharex=True, sharey=True)
axes = axes.flatten()

for i in range(8):
    ax = axes[i]
    c = ax.pcolormesh(
        times,
        freq_array,
        left_data_matrix[i],
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        shading="auto",
    )
    ax.axvline(0, color="k", linestyle="--", linewidth=1)
    ax.axvspan(0, 2.5, color="gray", alpha=0.08)  # 动作尝试阶段
    ax.set_title(f"ch {i + 1}")
    if i >= 6:
        ax.set_xlabel("Time from movement onset (s)")
    if i % 2 == 0:
        ax.set_ylabel("Frequency (Hz)")

fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
fig.colorbar(c, cax=cbar_ax, label="ERSP (dB, Movement / Rest)")
fig.suptitle("Left Hand Grasp vs Rest", fontsize=16, y=0.95)

save_path_1c = os.path.join(script_dir, "Fig_1c_Left_Grasp_vs_Rest_PDFRequirement.png")
plt.savefig(save_path_1c, dpi=300, bbox_inches="tight")
print(f"【成功】图 1c 已保存至: {save_path_1c}")

# =================================================================
# 4. 图 1d：ch3 左手/右手抓握 vs 各自 Rest 的频率曲线
# =================================================================
print("正在生成图 1d：左手/右手抓握 vs Rest 频率曲线...")
plt.figure(figsize=(9, 5))

move_mask = (times >= move_window[0]) & (times <= move_window[1])
best_motor_ch = 2  # ch3，Python 索引为 2

ersp_left_freq = np.mean(avg_tfr_left.data[best_motor_ch][:, move_mask], axis=1)
ersp_right_freq = np.mean(avg_tfr_right.data[best_motor_ch][:, move_mask], axis=1)

plt.axvspan(8, 12, color="blue", alpha=0.1, label=r"$\alpha$ (8-12 Hz)")
plt.axvspan(13, 30, color="cyan", alpha=0.1, label=r"$\beta$ (13-30 Hz)")
plt.axvspan(30, 50, color="green", alpha=0.1, label=r"low $\gamma$ (30-50 Hz)")
plt.axvspan(50, 150, color="red", alpha=0.1, label=r"high $\gamma$ (50-150 Hz)")

plt.plot(
    freq_array,
    ersp_left_freq,
    color="darkorange",
    linewidth=2.5,
    label="Left Hand Grasp vs Left Rest (contralateral)",
)
plt.plot(
    freq_array,
    ersp_right_freq,
    color="dodgerblue",
    linewidth=2.5,
    label="Right Hand Grasp vs Right Rest (ipsilateral)",
)
plt.axhline(0, color="grey", linestyle="-", linewidth=1.5)

plt.xlim(0, 200)
plt.xlabel("Frequency (Hz)")
plt.ylabel("ERSP (dB, Movement / Rest)")
plt.title("ERSP vs Frequency: Grasp Stage Compared with Rest (Channel 3)")
plt.legend(loc="upper right")
plt.grid(True, alpha=0.2)

save_path_1d = os.path.join(script_dir, "Fig_1d_Left_Right_vs_Rest_PDFRequirement.png")
plt.savefig(save_path_1d, dpi=300, bbox_inches="tight")
print(f"【成功】图 1d 已保存至: {save_path_1d}")

# =================================================================
# 5. 输出几个简单的定量指标，方便写报告
# =================================================================
def band_mean(curve, f_low, f_high):
    mask = (freq_array >= f_low) & (freq_array <= f_high)
    return float(np.mean(curve[mask]))

print("\n===== ch3 动作阶段 0~2.5 s 平均 ERSP（单位 dB，Movement / Rest）=====")
for band_name, f_low, f_high in [
    ("alpha", 8, 12),
    ("beta", 13, 30),
    ("low_gamma", 30, 50),
    ("high_gamma_50_100", 50, 100),
    ("high_gamma_50_150", 50, 150),
]:
    left_val = band_mean(ersp_left_freq, f_low, f_high)
    right_val = band_mean(ersp_right_freq, f_low, f_high)
    print(f"{band_name:18s} | Left Grasp: {left_val: .3f} dB | Right Grasp: {right_val: .3f} dB")

plt.show()
