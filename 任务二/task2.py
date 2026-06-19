#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# PBL 任务2：手部抓握连续神经解码
# 数据加载：自动读取指定路径下所有 single-movement 文件夹的 data.bdf 和 evt.bdf，从注释中提取事件码（1、2、3）。
# 训练集与验证集划分：基于左手握拳 trials（事件码1），随机划分为 80% 训练、20% 验证。
# 因果特征提取：对每个 trial 使用 0.5s 滑动窗口（步长 0.05s）提取 50‑150Hz 高伽马能量，并拼接前 4 个历史窗口以引入时序信息，确保在线因果性。
# 模型训练与评估：训练逻辑回归分类器，打印详细分类报告和混淆矩阵。
# 长时段连续解码：选取第一个文件夹的全时段信号（数分钟），用同一模型逐点预测抓握概率，并绘制整个时段的时频图（平均通道）与概率曲线，上下图共享 x 轴，数据部分严格对齐，图例外置。

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy.signal import hilbert, butter, filtfilt, spectrogram
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

# ---------- 设置中文字体 ----------
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ---------- 配置参数 ----------
DATA_ROOT = r'.\single-movement'# 修改为实际数据路径
TARGET_HAND = 'left'# 可修改left或right
HAND_CODE = {'left': 1, 'right': 2}

WIN_LEN = 0.5
STEP = 0.05
FREQ_BAND = (50, 150)
N_HIST = 4

TF_WIN = 0.5
TF_OVERLAP = 0.45
TF_FMIN = 1
TF_FMAX = 150

VAL_RATIO = 0.2          # 验证集比例

# ---------- 辅助函数 ----------
def butter_bandpass(low, high, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return b, a

def bandpass_filter(data, low, high, fs, order=4):
    b, a = butter_bandpass(low, high, fs, order=order)
    return filtfilt(b, a, data, axis=-1)

def compute_high_gamma_power_train(raw_data, fs, win_len, step, freq_band, t_start):
    """训练时用的特征提取（支持偏移时间）"""
    n_ch, n_samples = raw_data.shape
    window_samples = int(win_len * fs)
    step_samples = int(step * fs)
    filtered = bandpass_filter(raw_data, freq_band[0], freq_band[1], fs)
    analytic = hilbert(filtered, axis=-1)
    envelope = np.abs(analytic) ** 2
    centers = []
    features = []
    max_start_idx = n_samples - window_samples
    start_idx = 0
    while start_idx <= max_start_idx:
        end_idx = start_idx + window_samples
        win_data = envelope[:, start_idx:end_idx]
        feat = np.mean(win_data, axis=-1)
        center_time = t_start + (start_idx + window_samples/2) / fs
        centers.append(center_time)
        features.append(feat)
        start_idx += step_samples
    features = np.array(features)
    centers = np.array(centers)
    # 历史信息
    hist_features = []
    for i in range(features.shape[0]):
        start_idx = max(0, i - N_HIST)
        hist_win = features[start_idx:i+1, :]
        if hist_win.shape[0] < N_HIST + 1:
            pad = np.zeros((N_HIST + 1 - hist_win.shape[0], hist_win.shape[1]))
            hist_win = np.vstack([pad, hist_win])
        hist_features.append(hist_win.flatten())
    return np.array(hist_features), centers

def compute_high_gamma_power_continuous(raw_data, fs, win_len, step, freq_band):
    """连续信号特征提取（时间从0开始）"""
    n_ch, n_samples = raw_data.shape
    window_samples = int(win_len * fs)
    step_samples = int(step * fs)
    filtered = bandpass_filter(raw_data, freq_band[0], freq_band[1], fs)
    analytic = hilbert(filtered, axis=-1)
    envelope = np.abs(analytic) ** 2
    centers = []
    features = []
    max_start_idx = n_samples - window_samples
    start_idx = 0
    while start_idx <= max_start_idx:
        end_idx = start_idx + window_samples
        win_data = envelope[:, start_idx:end_idx]
        feat = np.mean(win_data, axis=-1)
        center_time = (start_idx + window_samples/2) / fs
        centers.append(center_time)
        features.append(feat)
        start_idx += step_samples
    features = np.array(features)
    centers = np.array(centers)
    hist_features = []
    for i in range(features.shape[0]):
        start_idx = max(0, i - N_HIST)
        hist_win = features[start_idx:i+1, :]
        if hist_win.shape[0] < N_HIST + 1:
            pad = np.zeros((N_HIST + 1 - hist_win.shape[0], hist_win.shape[1]))
            hist_win = np.vstack([pad, hist_win])
        hist_features.append(hist_win.flatten())
    return np.array(hist_features), centers

def read_events_from_evt(evt_path):
    raw = mne.io.read_raw_bdf(evt_path, preload=True)
    events = []
    for annot in raw.annotations:
        onset = annot['onset']
        desc = annot['description']
        try:
            code = int(desc)
        except ValueError:
            continue
        events.append((onset, code))
    return events, raw.info['sfreq']

def load_single_folder(folder_path):
    data_bdf = os.path.join(folder_path, 'data.bdf')
    evt_bdf = os.path.join(folder_path, 'evt.bdf')
    raw_data = mne.io.read_raw_bdf(data_bdf, preload=True)
    sfreq = raw_data.info['sfreq']
    data = raw_data.get_data()[:4, :]
    events, _ = read_events_from_evt(evt_bdf)
    return data, sfreq, events

def extract_trials(data, sfreq, events, target_code, pre_dur=2.0, post_dur=4.5):
    trials = []
    target_events = [ev for ev in events if ev[1] == target_code]
    for t, code in target_events:
        start_s = t - pre_dur
        end_s = t + post_dur
        start_idx = int(start_s * sfreq)
        end_idx = int(end_s * sfreq)
        if start_idx < 0 or end_idx > data.shape[1]:
            continue
        trial_data = data[:, start_idx:end_idx]
        t_axis = np.linspace(start_s, end_s, trial_data.shape[1])
        trials.append((trial_data, t_axis))
    return trials

def print_model_report(y_true, y_pred, y_prob=None):
    """打印分类质量报告"""
    print("\n" + "="*60)
    print("模型质量报告")
    print("="*60)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"准确率 (Accuracy): {acc:.4f}")
    print(f"精确率 (Precision): {prec:.4f}")
    print(f"召回率 (Recall):    {rec:.4f}")
    print(f"F1分数 (F1-score):  {f1:.4f}")
    if y_prob is not None:
        auc = roc_auc_score(y_true, y_prob)
        print(f"ROC-AUC:            {auc:.4f}")
    print("\n分类报告:")
    print(classification_report(y_true, y_pred, target_names=['非抓握', '抓握']))
    print("混淆矩阵:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    print("="*60 + "\n")

# ---------- 主流程 ----------
def main():
    folders = glob.glob(os.path.join(DATA_ROOT, 'TT01_*single-MA'))
    if not folders:
        print("未找到数据文件夹，请检查路径")
        return

    target_code = HAND_CODE.get(TARGET_HAND)
    print(f"目标动作: {TARGET_HAND} 握拳 (事件码 {target_code})")
    print(f"找到 {len(folders)} 个文件夹")

    # 提取所有trials
    all_trials = []
    for folder in folders:
        print(f"\n处理文件夹: {folder}")
        data, sfreq, events = load_single_folder(folder)
        codes = [e[1] for e in events]
        print(f"  事件码分布: {np.unique(codes, return_counts=True)}")
        trials = extract_trials(data, sfreq, events, target_code)
        print(f"  提取到 {len(trials)} 个 trials")
        all_trials.extend(trials)

    if not all_trials:
        print("未提取到任何trial")
        return

    print(f"\n总计提取到 {len(all_trials)} 个 trials")

    # ---------- 划分训练集和验证集 ----------
    train_trials, val_trials = train_test_split(all_trials, test_size=VAL_RATIO, random_state=42)
    print(f"训练集: {len(train_trials)} trials, 验证集: {len(val_trials)} trials")

    # ---------- 提取训练集特征 ----------
    X_train, y_train = [], []
    for trial_data, _ in train_trials:
        feat, centers = compute_high_gamma_power_train(trial_data, sfreq, WIN_LEN, STEP, FREQ_BAND, -2.0)
        valid = (centers >= -2.0) & (centers <= 4.5)
        feat, centers = feat[valid], centers[valid]
        labels = np.where((centers >= 0) & (centers < 2.5), 1, 0)
        X_train.append(feat)
        y_train.append(labels)

    X_train = np.vstack(X_train)
    y_train = np.hstack(y_train)

    # ---------- 提取验证集特征 ----------
    X_val, y_val = [], []
    for trial_data, _ in val_trials:
        feat, centers = compute_high_gamma_power_train(trial_data, sfreq, WIN_LEN, STEP, FREQ_BAND, -2.0)
        valid = (centers >= -2.0) & (centers <= 4.5)
        feat, centers = feat[valid], centers[valid]
        labels = np.where((centers >= 0) & (centers < 2.5), 1, 0)
        X_val.append(feat)
        y_val.append(labels)

    X_val = np.vstack(X_val)
    y_val = np.hstack(y_val)

    # ---------- 标准化和训练 ----------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)
    clf.fit(X_train_scaled, y_train)
    print("模型训练完成")

    # ---------- 验证集预测并打印报告 ----------
    y_pred = clf.predict(X_val_scaled)
    y_prob = clf.predict_proba(X_val_scaled)[:, 1]
    print_model_report(y_val, y_pred, y_prob)

    # ---------- 测试：选择一个完整文件夹进行长时段解码 ----------
    test_folder = folders[0]
    print(f"\n长时段解码测试文件夹: {test_folder}")
    data_test, sfreq_test, events_test = load_single_folder(test_folder)
    duration = data_test.shape[1] / sfreq_test
    print(f"测试数据时长: {duration:.2f} 秒")

    # 对整个数据计算特征
    feat_all, centers_all = compute_high_gamma_power_continuous(data_test, sfreq_test, WIN_LEN, STEP, FREQ_BAND)
    X_all_scaled = scaler.transform(feat_all)
    prob_all = clf.predict_proba(X_all_scaled)[:, 1]

    # ---------- 绘制整个时段的时频图和概率曲线 ----------
    ch_data = np.mean(data_test, axis=0)
    ch_data_filt = bandpass_filter(ch_data, TF_FMIN, TF_FMAX, sfreq_test, order=4)
    nperseg = int(TF_WIN * sfreq_test)
    noverlap = int(TF_OVERLAP * sfreq_test)
    f, t_spec, Sxx = spectrogram(ch_data_filt, fs=sfreq_test, nperseg=nperseg, noverlap=noverlap,
                                 scaling='density', mode='psd')
    freq_mask = f <= TF_FMAX
    f, Sxx = f[freq_mask], Sxx[freq_mask, :]
    Sxx_db = 10 * np.log10(Sxx + 1e-12)

    vmin = np.percentile(Sxx_db, 5)
    vmax = np.percentile(Sxx_db, 95)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True,
                                   gridspec_kw={'height_ratios': [2, 1]},
                                   constrained_layout=True)

    # 时频热图
    im = ax1.pcolormesh(t_spec, f, Sxx_db, shading='gouraud', cmap='RdBu_r', vmin=vmin, vmax=vmax)
    ax1.set_ylabel('频率 (Hz)')
    ax1.set_title(f'全时段时频图 (平均通道), {TARGET_HAND} 握拳')
    cbar = fig.colorbar(im, ax=ax1)
    cbar.set_label('PSD (dB)')

    # 标记事件码1（动作起始）和3（block结束）
    for t, code in events_test:
        if code == 1:
            ax1.axvline(t, color='green', linestyle='--', linewidth=0.5, alpha=0.6, label='动作起始' if t == events_test[0][0] else "")
        elif code == 3:
            ax1.axvline(t, color='red', linestyle='--', linewidth=0.5, alpha=0.6, label='Block结束' if t == events_test[0][0] else "")
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1.02, 1))

    # 概率曲线
    ax2.plot(centers_all, prob_all, color='blue', linewidth=1.5)
    ax2.set_xlabel('时间 (秒)')
    ax2.set_ylabel('抓握概率')
    ax2.set_ylim([-0.05, 1.05])
    ax2.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('连续解码抓握概率 (全时段)')

    ax1.set_xlim(0, duration)
    plt.show()

if __name__ == '__main__':
    main()

