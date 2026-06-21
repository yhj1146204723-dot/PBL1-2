#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PBL 任务4：多手势运动连续神经解码 v2（两阶段伪在线版）

设计目标：
1. 严格围绕作业 PDF 的任务4：结合任务2的连续解码思想和任务3的多手势分析，
   从 hand-gesture 数据中伪在线连续解码手势时间序列。
2. 相比 v1 的主要改进：
   - 两阶段模型：先判断 Rest/Open vs Movement，再在 Movement 中判断 6 种手势；
   - 额外输出二分类运动检测指标，更能体现连续控制价值；
   - 按 session/folder 分组划分 train/val/test，避免同一 session 泄露；
   - 对连续原始信号做单向 sosfilt，再切 trial，减少 epoch 起始滤波伪迹；
   - 预测时间点是滑动窗口终点，窗口只使用过去 WIN_LEN 秒数据；
   - 使用因果概率平滑与滞回阈值，减少连续输出抖动；
   - 输出 7 类混淆矩阵、二分类混淆矩阵、6 类 movement-only 混淆矩阵、连续解码图、CSV 结果；
   - 自动基于 prediction.csv 生成 Fig.3a 风格五指连续轨迹图。

运行前：修改 DATA_ROOT 为你的 hand-gesture 数据目录，例如：
DATA_ROOT = r"C:\\Users\\dsmji\\Desktop\\SCNS\\1\\PBL2026_Data\\data\\hand-gesture"

目录结构可为：
DATA_ROOT/
  TT01_xxx-handA/data.bdf, evt.bdf
  TT01_xxx-handB/data.bdf, evt.bdf
也支持递归搜索包含 data.bdf + evt.bdf 的 handA/handB 文件夹。
"""

import os
import re
import csv
import glob
import argparse
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.signal import butter, sosfilt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import make_pipeline

warnings.filterwarnings("ignore")

# =========================
# 1. 配置区
# =========================
DATA_ROOT = r"C:\\Users\\dsmji\\Desktop\\SCNS\\1\\PBL2026_Data\\data\\hand-gesture"  # 改成你的 hand-gesture 路径
OUTPUT_DIR = "results_task4_v2"
FIG3A_DIR_NAME = "fig3a_v2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# handA / handB 的 trigger b 映射：根据作业图 3c 和你们任务3代码
HAND_MAP = {
    "A": {"gestures": ["scissor", "six", "grasp"], "b_triggers": ["1", "4", "7"]},
    "B": {"gestures": ["index", "seven", "thumb"], "b_triggers": ["1", "4", "7"]},
}
GESTURES = ["grasp", "scissor", "six", "thumb", "index", "seven"]
LABELS_7 = ["rest"] + GESTURES
LABEL_TO_ID_7 = {name: i for i, name in enumerate(LABELS_7)}
GESTURE_TO_ID_6 = {name: i for i, name in enumerate(GESTURES)}

# 作业 PDF 中视频跟踪阶段：trigger b 记为 0 s，0~8 s
TRIAL_TMIN = 0.0
TRIAL_TMAX = 8.0
REST_END = 1.5
FLEX_END = 4.0
HOLD_END = 5.5
# 5.5~8.0 是 Extend / Return。为了输出 Rest + 6 手势序列，这里归为 rest/open。
EXTEND_AS_REST = True

# 伪在线滑动窗口：预测时刻为窗口终点，只使用过去 WIN_LEN 秒
WIN_LEN = 0.5
STEP = 0.05
N_HIST = 4

# 多频段 + 8 通道特征
BANDS = {
    "beta_13_30": (13, 30),
    "low_gamma_30_50": (30, 50),
    "high_gamma_50_100": (50, 100),
    "broad_high_gamma_50_150": (50, 150),
}

# 数据划分。先划 test，再从剩余中划 val。全部按 folder/session 分组。
TEST_SIZE = 0.20
VAL_SIZE_WITHIN_TRAIN = 0.20
RANDOM_STATE = 42

# 两阶段阈值与后处理
# movement 阈值会在验证集上自动搜索；以下是兜底值
DEFAULT_MOVEMENT_THRESHOLD = 0.45
CAUSAL_SMOOTH_ALPHA = 0.25       # 概率指数平滑，越小越平滑
MAJORITY_SECONDS = 0.50          # 类别多数投票窗口
MIN_CONFIDENCE = 0.0             # 可设 0.35，低置信度强制 rest；默认不强制

# 模型选择
# gesture 可选 "logreg" 或 "rf"。RF 通常对非线性类别边界更友好，但慢一点。
GESTURE_MODEL = "rf"

# 可视化最多输出多少个测试 trial。优先每个手势选一个。
N_PLOT_TRIALS = 8

# 中文显示（没有 SimHei 时 matplotlib 会自动 fallback，不影响运行）
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# =========================
# 2. 数据结构
# =========================
@dataclass
class TrialRecord:
    folder: str
    hand_type: str
    gesture: str
    trigger_time: float
    sfreq: float
    times: np.ndarray                 # 每个窗口的预测时间，单位 s，0~8
    X: np.ndarray                     # raw feature, shape=(n_windows, n_features)
    y7: np.ndarray                    # 7 类标签：rest + 6 gesture
    y_bin: np.ndarray                 # 0=rest/open, 1=movement
    trial_index: int


# =========================
# 3. 文件读取辅助函数
# =========================
def infer_hand_type(folder_name: str) -> Optional[str]:
    lower = folder_name.lower()
    if "handa" in lower or "-handa" in lower:
        return "A"
    if "handb" in lower or "-handb" in lower:
        return "B"
    return None


def find_hand_gesture_folders(data_root: str) -> List[str]:
    """递归查找包含 data.bdf 和 evt.bdf 的 handA/handB 文件夹。"""
    candidate_folders = []
    for root, dirs, files in os.walk(data_root):
        files_lower = {f.lower() for f in files}
        if "data.bdf" in files_lower and "evt.bdf" in files_lower:
            if infer_hand_type(os.path.basename(root)) is not None:
                candidate_folders.append(root)
    candidate_folders = sorted(candidate_folders)
    return candidate_folders


def read_raw_and_events(folder_path: str):
    data_bdf = os.path.join(folder_path, "data.bdf")
    evt_bdf = os.path.join(folder_path, "evt.bdf")
    raw = mne.io.read_raw_bdf(data_bdf, preload=True, verbose=False)

    # 只保留前 8 个通道，按作业 NEO 8 通道电极处理
    raw.pick(raw.ch_names[:8])
    raw.set_channel_types({ch: "ecog" for ch in raw.ch_names})

    # evt.bdf 作为 annotations 读取
    try:
        annot = mne.read_annotations(evt_bdf)
        raw.set_annotations(annot)
        events, event_id = mne.events_from_annotations(raw, verbose=False)
    except Exception:
        raw_evt = mne.io.read_raw_bdf(evt_bdf, preload=True, verbose=False)
        events = []
        event_id = {}
        for ann in raw_evt.annotations:
            desc = str(ann["description"])
            if desc not in event_id:
                event_id[desc] = len(event_id) + 1
            events.append([int(round(ann["onset"] * raw.info["sfreq"])), 0, event_id[desc]])
        events = np.asarray(events, dtype=int)

    return raw, events, event_id


# =========================
# 4. 特征提取
# =========================
def causal_bandpower_full_raw(data: np.ndarray, sfreq: float) -> Dict[str, np.ndarray]:
    """
    对完整连续 raw 做单向 sosfilt，输出每个频段的瞬时功率。
    data: (n_ch, n_samples)
    return: band -> power array (n_ch, n_samples)
    """
    band_power = {}
    for band_name, (fmin, fmax) in BANDS.items():
        sos = butter(4, [fmin, fmax], btype="bandpass", fs=sfreq, output="sos")
        filtered = sosfilt(sos, data, axis=-1)
        # 为降低计算量，不做 Hilbert；用平方幅值作为 causal band power 近似
        power = filtered ** 2
        band_power[band_name] = power.astype(np.float32)
    return band_power


def label_at_time(t: float, gesture: str) -> Tuple[int, int]:
    """返回 y7, y_bin。预测时间点 t 为窗口终点。"""
    if t < REST_END:
        return LABEL_TO_ID_7["rest"], 0
    if t < HOLD_END:
        # Flex + Hold 都输出当前手势，用于连续手势序列
        return LABEL_TO_ID_7[gesture], 1
    # Extend / Return 阶段
    if EXTEND_AS_REST:
        return LABEL_TO_ID_7["rest"], 0
    # 如果后续想做 8 类，可在这里扩展 extend 类
    return LABEL_TO_ID_7["rest"], 0


def extract_trial_features_from_bandpower(
    band_power: Dict[str, np.ndarray],
    sfreq: float,
    trigger_time: float,
    gesture: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    对单个 trial 提取伪在线窗口特征。
    预测时间点为 window_end，窗口为 [window_end-WIN_LEN, window_end]。
    每个时间点再拼接过去 N_HIST 个窗口特征。
    """
    n_ch = next(iter(band_power.values())).shape[0]
    win_samp = int(round(WIN_LEN * sfreq))
    step_samp = int(round(STEP * sfreq))
    start_abs = int(round(trigger_time * sfreq))
    end_abs = int(round((trigger_time + TRIAL_TMAX) * sfreq))
    n_total = next(iter(band_power.values())).shape[1]
    if start_abs < 0 or end_abs > n_total:
        raise ValueError("trial 超出数据边界")

    # 预测时间从 WIN_LEN 开始，保证窗口完整；直到 8.0 s
    window_end_offsets = np.arange(win_samp, int(round(TRIAL_TMAX * sfreq)) + 1, step_samp)
    times = window_end_offsets / sfreq

    base_feats = []
    for offset in window_end_offsets:
        end_idx = start_abs + int(offset)
        start_idx = end_idx - win_samp
        feat = []
        for band_name in BANDS.keys():
            p = band_power[band_name][:, start_idx:end_idx]
            # log power 更稳定，避免不同 session 绝对幅值差异过大
            ch_power = np.log10(np.mean(p, axis=1) + 1e-20)
            feat.extend(ch_power.tolist())
        base_feats.append(feat)
    base_feats = np.asarray(base_feats, dtype=np.float32)

    # 历史窗口拼接：t-N_HIST ... t-0
    hist_feats = []
    for i in range(len(base_feats)):
        start_i = max(0, i - N_HIST)
        hist = base_feats[start_i : i + 1]
        if hist.shape[0] < N_HIST + 1:
            # 用最早可用窗口重复填充，比全零更不引入异常值
            pad = np.repeat(hist[0:1], N_HIST + 1 - hist.shape[0], axis=0)
            hist = np.vstack([pad, hist])
        hist_feats.append(hist.reshape(-1))
    X = np.asarray(hist_feats, dtype=np.float32)

    y7, y_bin = [], []
    for t in times:
        yy7, yyb = label_at_time(float(t), gesture)
        y7.append(yy7)
        y_bin.append(yyb)
    y7 = np.asarray(y7, dtype=int)
    y_bin = np.asarray(y_bin, dtype=int)

    # feature names
    names = []
    for hist_i in range(N_HIST, -1, -1):
        # hist_i 越大表示越久以前，这里命名为 t-4step...t-0step
        for band_name in BANDS.keys():
            for ch in range(1, n_ch + 1):
                names.append(f"t-{hist_i}step_{band_name}_ch{ch}")

    return times, X, y7, y_bin, names


def load_all_records(data_root: str) -> Tuple[List[TrialRecord], List[str]]:
    folders = find_hand_gesture_folders(data_root)
    print("=" * 80)
    print(f"开始读取 hand-gesture 数据，DATA_ROOT = {data_root}")
    print(f"递归发现 {len(folders)} 个 handA/handB 数据文件夹")
    if len(folders) == 0:
        raise RuntimeError("没有找到 handA/handB 数据文件夹。请确认 DATA_ROOT 指向 data/hand-gesture。")

    records: List[TrialRecord] = []
    feature_names: List[str] = []
    counts = {g: 0 for g in GESTURES}

    for folder_path in folders:
        folder = os.path.basename(folder_path)
        hand_type = infer_hand_type(folder)
        if hand_type not in HAND_MAP:
            continue
        print(f"\n处理文件夹: {folder}")
        try:
            raw, events, event_id = read_raw_and_events(folder_path)
            sfreq = float(raw.info["sfreq"])
            data = raw.get_data().astype(np.float32)
            band_power = causal_bandpower_full_raw(data, sfreq)

            gestures = HAND_MAP[hand_type]["gestures"]
            b_triggers = HAND_MAP[hand_type]["b_triggers"]
            event_id_inv = {v: k for k, v in event_id.items()}
            print(f"  event_id: {event_id}")

            trial_index = 0
            for gesture, trig_str in zip(gestures, b_triggers):
                if trig_str not in event_id:
                    print(f"  警告：{folder} 中缺少 trigger b={trig_str}，跳过 {gesture}")
                    continue
                trig_code = event_id[trig_str]
                trig_events = events[events[:, 2] == trig_code]
                print(f"  {gesture:>8} | trigger b={trig_str} | trials={len(trig_events)}")
                for ev in trig_events:
                    trigger_time = ev[0] / sfreq
                    try:
                        times, X, y7, y_bin, names = extract_trial_features_from_bandpower(
                            band_power, sfreq, trigger_time, gesture
                        )
                        if not feature_names:
                            feature_names = names
                        records.append(
                            TrialRecord(
                                folder=folder,
                                hand_type=hand_type,
                                gesture=gesture,
                                trigger_time=trigger_time,
                                sfreq=sfreq,
                                times=times,
                                X=X,
                                y7=y7,
                                y_bin=y_bin,
                                trial_index=trial_index,
                            )
                        )
                        counts[gesture] += 1
                        trial_index += 1
                    except Exception as e:
                        # 边界 trial 可能不足 8s，跳过
                        pass
        except Exception as e:
            print(f"  跳过 {folder}，原因：{e}")
            continue

    print("\n" + "=" * 80)
    print("数据读取完成，各手势 trial 数：")
    for g in GESTURES:
        print(f"  {g:>8}: {counts[g]}")
    print(f"总 trial 数: {len(records)}")
    print("=" * 80)
    if len(records) == 0:
        raise RuntimeError("没有成功读取任何 hand-gesture trial，请检查路径和事件标记。")
    return records, feature_names


# =========================
# 5. 划分数据与拼接矩阵
# =========================
def split_records_by_group(records: List[TrialRecord]):
    groups = np.asarray([r.folder for r in records])
    idx = np.arange(len(records))

    gss1 = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    trainval_idx, test_idx = next(gss1.split(idx, groups=groups))

    trainval_groups = groups[trainval_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=VAL_SIZE_WITHIN_TRAIN, random_state=RANDOM_STATE + 1)
    train_rel, val_rel = next(gss2.split(trainval_idx, groups=trainval_groups))
    train_idx = trainval_idx[train_rel]
    val_idx = trainval_idx[val_rel]

    train_records = [records[i] for i in train_idx]
    val_records = [records[i] for i in val_idx]
    test_records = [records[i] for i in test_idx]

    def folders_of(rs):
        return sorted(set(r.folder for r in rs))

    print("\n数据划分（按 folder/session 分组）：")
    print(f"  Train trials: {len(train_records)}, folders: {len(folders_of(train_records))}")
    print(f"  Val   trials: {len(val_records)}, folders: {len(folders_of(val_records))}")
    print(f"  Test  trials: {len(test_records)}, folders: {len(folders_of(test_records))}")
    print(f"  Test folders: {folders_of(test_records)[:10]}{' ...' if len(folders_of(test_records)) > 10 else ''}")
    return train_records, val_records, test_records


def stack_records(records: List[TrialRecord]):
    X = np.vstack([r.X for r in records])
    y7 = np.concatenate([r.y7 for r in records])
    y_bin = np.concatenate([r.y_bin for r in records])
    gesture_id = np.full_like(y7, fill_value=-1)
    for lab_name, lab_id7 in LABEL_TO_ID_7.items():
        if lab_name in GESTURE_TO_ID_6:
            gesture_id[y7 == lab_id7] = GESTURE_TO_ID_6[lab_name]
    return X, y7, y_bin, gesture_id


# =========================
# 6. 后处理与评估
# =========================
def causal_ema(proba: np.ndarray, alpha: float = CAUSAL_SMOOTH_ALPHA) -> np.ndarray:
    if proba.ndim == 1:
        out = np.zeros_like(proba, dtype=float)
        out[0] = proba[0]
        for i in range(1, len(proba)):
            out[i] = alpha * proba[i] + (1 - alpha) * out[i - 1]
        return out
    out = np.zeros_like(proba, dtype=float)
    out[0] = proba[0]
    for i in range(1, proba.shape[0]):
        out[i] = alpha * proba[i] + (1 - alpha) * out[i - 1]
    row_sum = out.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    return out / row_sum


def majority_filter(labels: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return labels.copy()
    out = labels.copy()
    for i in range(len(labels)):
        start = max(0, i - win + 1)
        vals, cnts = np.unique(labels[start : i + 1], return_counts=True)
        out[i] = vals[np.argmax(cnts)]
    return out


def tune_movement_threshold(y_true_bin: np.ndarray, p_move: np.ndarray) -> float:
    candidates = np.arange(0.20, 0.81, 0.02)
    best_thr = DEFAULT_MOVEMENT_THRESHOLD
    best_score = -np.inf
    for thr in candidates:
        pred = (p_move >= thr).astype(int)
        score = balanced_accuracy_score(y_true_bin, pred)
        if score > best_score:
            best_score = score
            best_thr = float(thr)
    print(f"验证集自动选择 movement threshold = {best_thr:.2f}, val binary BAcc = {best_score:.4f}")
    return best_thr


def combine_two_stage_predictions(
    p_move: np.ndarray,
    p_gesture6: np.ndarray,
    movement_threshold: float,
    times: Optional[np.ndarray] = None,
    apply_smoothing: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    输出 raw_pred7 和 smoothed_pred7。
    raw_pred7: 0=rest, 1~6=gesture
    """
    # 组合概率：rest = 1-pmove, gesture_k = pmove * pgesture_k
    proba7 = np.zeros((len(p_move), len(LABELS_7)), dtype=float)
    proba7[:, 0] = 1.0 - p_move
    proba7[:, 1:] = p_move[:, None] * p_gesture6

    raw = np.argmax(proba7, axis=1)
    # 强制 movement threshold：低于阈值直接 rest，高于阈值采用 6 手势分类
    best_g = np.argmax(p_gesture6, axis=1) + 1
    raw = np.where(p_move >= movement_threshold, best_g, 0)

    if not apply_smoothing:
        return raw, raw

    p_move_s = causal_ema(p_move, alpha=CAUSAL_SMOOTH_ALPHA)
    p_gesture_s = causal_ema(p_gesture6, alpha=CAUSAL_SMOOTH_ALPHA)
    best_g_s = np.argmax(p_gesture_s, axis=1) + 1
    max_g_prob = np.max(p_gesture_s, axis=1)
    smoothed = np.where(p_move_s >= movement_threshold, best_g_s, 0)
    if MIN_CONFIDENCE > 0:
        smoothed = np.where((smoothed != 0) & (max_g_prob < MIN_CONFIDENCE), 0, smoothed)

    win = max(1, int(round(MAJORITY_SECONDS / STEP)))
    smoothed = majority_filter(smoothed, win=win)
    return raw, smoothed


def print_and_save_metrics(
    out_path: str,
    y7_true: np.ndarray,
    y7_raw: np.ndarray,
    y7_smooth: np.ndarray,
    ybin_true: np.ndarray,
    p_move: np.ndarray,
    gesture_true6: np.ndarray,
    gesture_pred6: np.ndarray,
    feature_names: List[str],
    train_records: List[TrialRecord],
    val_records: List[TrialRecord],
    test_records: List[TrialRecord],
    threshold: float,
):
    ybin_pred = (p_move >= threshold).astype(int)
    move_mask_true = gesture_true6 >= 0

    lines = []
    add = lines.append
    add("任务4：多手势运动连续神经解码 v2（两阶段伪在线版）")
    add("=" * 90)
    add(f"DATA_ROOT: {DATA_ROOT}")
    add(f"Train trials: {len(train_records)}")
    add(f"Val trials:   {len(val_records)}")
    add(f"Test trials:  {len(test_records)}")
    add(f"Test windows: {len(y7_true)}")
    add(f"Feature dim:  {len(feature_names)}")
    add(f"WIN_LEN={WIN_LEN}s, STEP={STEP}s, N_HIST={N_HIST}")
    add(f"BANDS={BANDS}")
    add(f"EXTEND_AS_REST={EXTEND_AS_REST}")
    add(f"GESTURE_MODEL={GESTURE_MODEL}")
    add(f"movement_threshold={threshold:.2f}")
    add("")

    add("一、7类连续解码指标：rest + 6 gestures")
    add("- raw prediction:")
    add(f"  Accuracy:          {accuracy_score(y7_true, y7_raw):.4f}")
    add(f"  Balanced Accuracy: {balanced_accuracy_score(y7_true, y7_raw):.4f}")
    add(classification_report(y7_true, y7_raw, target_names=LABELS_7, zero_division=0))
    add("- smoothed prediction:")
    add(f"  Accuracy:          {accuracy_score(y7_true, y7_smooth):.4f}")
    add(f"  Balanced Accuracy: {balanced_accuracy_score(y7_true, y7_smooth):.4f}")
    add(classification_report(y7_true, y7_smooth, target_names=LABELS_7, zero_division=0))
    add("")

    add("二、二分类运动状态检测指标：rest/open vs movement")
    add(f"  Accuracy:          {accuracy_score(ybin_true, ybin_pred):.4f}")
    add(f"  Balanced Accuracy: {balanced_accuracy_score(ybin_true, ybin_pred):.4f}")
    add(f"  F1 movement:       {f1_score(ybin_true, ybin_pred, zero_division=0):.4f}")
    add(classification_report(ybin_true, ybin_pred, target_names=["rest/open", "movement"], zero_division=0))
    add("")

    add("三、Movement-only 6类手势识别指标（只在真实 movement 窗口统计）")
    if np.any(move_mask_true):
        add(f"  Accuracy:          {accuracy_score(gesture_true6[move_mask_true], gesture_pred6[move_mask_true]):.4f}")
        add(f"  Balanced Accuracy: {balanced_accuracy_score(gesture_true6[move_mask_true], gesture_pred6[move_mask_true]):.4f}")
        add(classification_report(gesture_true6[move_mask_true], gesture_pred6[move_mask_true], target_names=GESTURES, zero_division=0))
    else:
        add("  No movement windows in test set.")
    add("")

    add("四、7类 smoothed 混淆矩阵")
    add(str(confusion_matrix(y7_true, y7_smooth)))
    add("")
    add("五、特征名称")
    add("\n".join(feature_names))

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines[:40]))
    print(f"\n完整指标已保存：{out_path}")


def plot_confusion(cm: np.ndarray, labels: List[str], title: str, save_path: str):
    plt.figure(figsize=(10, 8))
    im = plt.imshow(cm, interpolation="nearest", cmap="viridis")
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(np.arange(len(labels)), labels)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = int(cm[i, j])
            color = "white" if val > cm.max() * 0.55 else "black"
            plt.text(j, i, str(val), ha="center", va="center", color=color, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# =========================
# 7. 可视化与 CSV 输出
# =========================
def get_current_hg_db(record: TrialRecord, feature_names: List[str]) -> np.ndarray:
    """从当前窗口 t-0step 的 high_gamma_50_100_ch1-8 提取平均 dB，用于上图展示。"""
    indices = [i for i, n in enumerate(feature_names) if n.startswith("t-0step_high_gamma_50_100")]
    if not indices:
        return np.zeros(len(record.times))
    # 特征本身是 log10(power)，转为 10*log10(power)
    return 10.0 * np.mean(record.X[:, indices], axis=1)


def plot_one_record(
    record: TrialRecord,
    feature_names: List[str],
    bin_model,
    gesture_model,
    threshold: float,
    save_prefix: str,
):
    X = record.X
    p_move = bin_model.predict_proba(X)[:, 1]
    p_gesture6 = gesture_model.predict_proba(X)
    raw, smooth = combine_two_stage_predictions(p_move, p_gesture6, threshold, record.times)

    proba7 = np.zeros((len(p_move), len(LABELS_7)))
    proba7[:, 0] = 1 - p_move
    proba7[:, 1:] = p_move[:, None] * p_gesture6
    proba7_s = causal_ema(proba7, alpha=CAUSAL_SMOOTH_ALPHA)

    times = record.times
    hg_db = get_current_hg_db(record, feature_names)

    fig, axes = plt.subplots(4, 1, figsize=(16, 10), sharex=True, gridspec_kw={"height_ratios": [1.1, 0.8, 0.9, 1.4]})

    def add_stage_bg(ax):
        ax.axvspan(0, REST_END, color="lightgray", alpha=0.25, label="Rest")
        ax.axvspan(REST_END, FLEX_END, color="lightcoral", alpha=0.18, label="Flex")
        ax.axvspan(FLEX_END, HOLD_END, color="wheat", alpha=0.25, label="Hold")
        ax.axvspan(HOLD_END, TRIAL_TMAX, color="lightgreen", alpha=0.20, label="Extend")
        for x in [REST_END, FLEX_END, HOLD_END]:
            ax.axvline(x, color="k", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.grid(alpha=0.25)

    axes[0].plot(times, hg_db, linewidth=1.8)
    axes[0].set_ylabel("HG power\n(dB)")
    axes[0].set_title(f"Task4 v2 continuous decoding | {record.folder} | true={record.gesture} | epoch={record.trial_index}")
    add_stage_bg(axes[0])
    axes[0].legend(loc="upper right", ncol=4, fontsize=8)

    axes[1].step(times, record.y7, where="post", linewidth=2)
    axes[1].set_yticks(np.arange(len(LABELS_7)))
    axes[1].set_yticklabels(LABELS_7)
    axes[1].set_ylabel("True")
    add_stage_bg(axes[1])

    axes[2].step(times, raw, where="post", alpha=0.35, linewidth=1.2, label="raw")
    axes[2].step(times, smooth, where="post", linewidth=2.2, label="smoothed")
    axes[2].set_yticks(np.arange(len(LABELS_7)))
    axes[2].set_yticklabels(LABELS_7)
    axes[2].set_ylabel("Pred")
    add_stage_bg(axes[2])
    axes[2].legend(loc="upper right")

    for i, lab in enumerate(LABELS_7):
        axes[3].plot(times, proba7_s[:, i], linewidth=1.5, label=lab)
    axes[3].axhline(threshold, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="movement threshold")
    axes[3].set_ylim(-0.03, 1.03)
    axes[3].set_ylabel("Probability")
    axes[3].set_xlabel("Time from trigger b (s)")
    add_stage_bg(axes[3])
    axes[3].legend(loc="upper right", ncol=4, fontsize=8)

    plt.tight_layout()
    fig_path = save_prefix + ".png"
    plt.savefig(fig_path, dpi=180)
    plt.close()

    csv_path = save_prefix + "_prediction.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["time_s", "true_label", "pred_raw", "pred_smooth", "p_movement"] + [f"p_{lab}" for lab in LABELS_7]
        writer.writerow(header)
        for i, t in enumerate(times):
            writer.writerow([
                f"{t:.4f}",
                LABELS_7[int(record.y7[i])],
                LABELS_7[int(raw[i])],
                LABELS_7[int(smooth[i])],
                f"{p_move[i]:.6f}",
                *[f"{proba7_s[i, j]:.6f}" for j in range(len(LABELS_7))],
            ])

    # Omnihand/MuJoCo 简化占位 CSV：24维关节轨迹，供后续接口替换
    omni_path = save_prefix + "_omnihand_demo.csv"
    write_omnihand_demo_csv(omni_path, times, smooth)
    return fig_path, csv_path, omni_path


def gesture_to_24dof(label_id: int) -> np.ndarray:
    """简化的 24 关节手势占位映射：0=伸展，1=弯曲。实际 Omnihand 接口可替换本函数。"""
    q = np.zeros(24, dtype=float)
    lab = LABELS_7[int(label_id)]
    # 假设每根手指 4 个自由度：thumb,index,middle,ring,pinky，共 20；其余 4 个置0
    fingers = {
        "thumb": [0, 1, 2, 3],
        "index": [4, 5, 6, 7],
        "middle": [8, 9, 10, 11],
        "ring": [12, 13, 14, 15],
        "pinky": [16, 17, 18, 19],
    }
    if lab == "rest":
        return q
    if lab == "grasp":
        q[:20] = 1.0
    elif lab == "scissor":
        # index + middle 伸展，其余弯曲，近似剪刀手
        q[:20] = 1.0
        q[fingers["index"]] = 0.0
        q[fingers["middle"]] = 0.0
    elif lab == "six":
        q[:20] = 1.0
        q[fingers["thumb"]] = 0.0
        q[fingers["pinky"]] = 0.0
    elif lab == "thumb":
        q[:20] = 1.0
        q[fingers["thumb"]] = 0.0
    elif lab == "index":
        q[:20] = 1.0
        q[fingers["index"]] = 0.0
    elif lab == "seven":
        q[:20] = 1.0
        q[fingers["thumb"]] = 0.0
        q[fingers["index"]] = 0.0
    return q


def write_omnihand_demo_csv(path: str, times: np.ndarray, labels: np.ndarray):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s"] + [f"q{i+1}" for i in range(24)])
        for t, lab in zip(times, labels):
            q = gesture_to_24dof(int(lab))
            writer.writerow([f"{t:.4f}"] + [f"{x:.3f}" for x in q])


def choose_plot_records(test_records: List[TrialRecord], n: int) -> List[TrialRecord]:
    chosen = []
    used_g = set()
    for r in test_records:
        if r.gesture not in used_g:
            chosen.append(r)
            used_g.add(r.gesture)
        if len(chosen) >= min(n, len(GESTURES)):
            break
    if len(chosen) < n:
        for r in test_records:
            if r not in chosen:
                chosen.append(r)
            if len(chosen) >= n:
                break
    return chosen[:n]


# =========================
# 8. 主流程
# =========================
def main():
    global DATA_ROOT, OUTPUT_DIR, N_PLOT_TRIALS
    parser = argparse.ArgumentParser(
        description="PBL Task4 all-in-one: v2 two-stage pseudo-online decoding + Fig.3a-style trajectory plotting."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=DATA_ROOT,
        help="hand-gesture 数据目录，例如 data/hand-gesture。",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=OUTPUT_DIR,
        help="任务4结果输出目录，默认 results_task4_v2。",
    )
    parser.add_argument(
        "--n_plot_trials",
        type=int,
        default=N_PLOT_TRIALS,
        help="训练完成后输出多少个测试 trial 的连续解码图和 Fig.3a 风格图。",
    )
    parser.add_argument(
        "--fig3a_only",
        action="store_true",
        help="不重新训练模型，只从已有 *_prediction.csv 生成 Fig.3a 风格图。",
    )
    parser.add_argument(
        "--fig3a_glob",
        type=str,
        default=None,
        help='fig3a_only 模式下的 CSV 搜索模式，例如 "results_task4_v2/*_prediction.csv"。',
    )
    parser.add_argument(
        "--fig3a_dir",
        type=str,
        default=None,
        help="Fig.3a 风格图输出目录。默认 output_dir/fig3a_v2。",
    )
    parser.add_argument(
        "--hide_hard_fig3a",
        action="store_true",
        help="生成 Fig.3a 风格图时隐藏绿色 hard decoded 轨迹。",
    )
    args = parser.parse_args()

    DATA_ROOT = args.data_root
    OUTPUT_DIR = args.output_dir
    N_PLOT_TRIALS = args.n_plot_trials
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig3a_dir = Path(args.fig3a_dir) if args.fig3a_dir else Path(OUTPUT_DIR) / FIG3A_DIR_NAME

    if args.fig3a_only:
        csv_files = collect_fig3a_inputs(None, args.fig3a_glob)
        if not csv_files:
            raise SystemExit(
                "No prediction CSV files found. 请提供 --fig3a_glob，或把脚本放在包含 results_task4_v2 的目录下运行。"
            )
        print("=" * 80)
        print("Fig.3a-only 模式：从已有 prediction.csv 生成五指连续轨迹图")
        print(f"CSV files: {len(csv_files)}")
        print(f"Output dir: {fig3a_dir}")
        print("=" * 80)
        for csv_path in csv_files:
            out_path = fig3a_dir / (csv_path.stem.replace("_prediction", "") + "_fig3a_style_v2.png")
            saved = plot_fig3a_style(csv_path, out_path=out_path, show_hard=not args.hide_hard_fig3a)
            print(f"  Saved: {saved}")
        print("完成。")
        return

    records, feature_names = load_all_records(DATA_ROOT)
    train_records, val_records, test_records = split_records_by_group(records)

    X_train, y7_train, ybin_train, gid_train = stack_records(train_records)
    X_val, y7_val, ybin_val, gid_val = stack_records(val_records)
    X_test, y7_test, ybin_test, gid_test = stack_records(test_records)

    print("\n窗口数量：")
    print(f"  X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
    print(f"  特征维度: {X_train.shape[1]}")

    # -------- stage 1: movement detector --------
    print("\n训练 Stage 1：Rest/Open vs Movement 二分类模型...")
    bin_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0, random_state=RANDOM_STATE),
    )
    bin_model.fit(X_train, ybin_train)
    p_move_val = bin_model.predict_proba(X_val)[:, 1]
    threshold = tune_movement_threshold(ybin_val, p_move_val)

    # -------- stage 2: gesture classifier, only movement windows --------
    print("\n训练 Stage 2：Movement-only 6类手势模型...")
    mov_train_mask = gid_train >= 0
    mov_val_mask = gid_val >= 0
    if GESTURE_MODEL == "rf":
        gesture_model = make_pipeline(
            StandardScaler(),
            RandomForestClassifier(
                n_estimators=250,
                max_depth=14,
                min_samples_leaf=3,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        )
    else:
        gesture_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1500, class_weight="balanced", C=1.0, random_state=RANDOM_STATE),
        )
    gesture_model.fit(X_train[mov_train_mask], gid_train[mov_train_mask])

    if np.any(mov_val_mask):
        val_gid_pred = gesture_model.predict(X_val[mov_val_mask])
        print(f"  Val movement-only gesture acc = {accuracy_score(gid_val[mov_val_mask], val_gid_pred):.4f}")
        print(f"  Val movement-only gesture BAcc = {balanced_accuracy_score(gid_val[mov_val_mask], val_gid_pred):.4f}")

    # -------- test prediction --------
    p_move_test = bin_model.predict_proba(X_test)[:, 1]
    p_gesture_test = gesture_model.predict_proba(X_test)
    y7_raw, y7_smooth = combine_two_stage_predictions(p_move_test, p_gesture_test, threshold)
    gid_pred_all = np.argmax(p_gesture_test, axis=1)

    # -------- save metrics --------
    metrics_path = os.path.join(OUTPUT_DIR, "task4_v2_metrics.txt")
    print_and_save_metrics(
        metrics_path,
        y7_test,
        y7_raw,
        y7_smooth,
        ybin_test,
        p_move_test,
        gid_test,
        gid_pred_all,
        feature_names,
        train_records,
        val_records,
        test_records,
        threshold,
    )

    # -------- confusion matrices --------
    cm7 = confusion_matrix(y7_test, y7_smooth, labels=np.arange(len(LABELS_7)))
    plot_confusion(
        cm7,
        LABELS_7,
        f"Task4 v2 7-class confusion matrix | Acc={accuracy_score(y7_test, y7_smooth):.3f}, BAcc={balanced_accuracy_score(y7_test, y7_smooth):.3f}",
        os.path.join(OUTPUT_DIR, "task4_v2_confusion_matrix_7class.png"),
    )

    # 二分类混淆矩阵：rest/open vs movement，便于报告和海报展示
    ybin_pred_test = (p_move_test >= threshold).astype(int)
    cm_bin = confusion_matrix(ybin_test, ybin_pred_test, labels=[0, 1])
    plot_confusion(
        cm_bin,
        ["rest/open", "movement"],
        f"Task4 v2 binary confusion | Acc={accuracy_score(ybin_test, ybin_pred_test):.3f}, BAcc={balanced_accuracy_score(ybin_test, ybin_pred_test):.3f}",
        os.path.join(OUTPUT_DIR, "task4_v2_confusion_matrix_binary_rest_movement.png"),
    )

    mov_mask_test = gid_test >= 0
    if np.any(mov_mask_test):
        cm6 = confusion_matrix(gid_test[mov_mask_test], gid_pred_all[mov_mask_test], labels=np.arange(len(GESTURES)))
        plot_confusion(
            cm6,
            GESTURES,
            f"Task4 v2 movement-only 6-gesture confusion | Acc={accuracy_score(gid_test[mov_mask_test], gid_pred_all[mov_mask_test]):.3f}",
            os.path.join(OUTPUT_DIR, "task4_v2_confusion_matrix_6gesture_movement_only.png"),
        )

    # -------- visualize selected test trials --------
    selected = choose_plot_records(test_records, N_PLOT_TRIALS)
    generated_prediction_csvs: List[str] = []
    print("\n输出连续解码可视化：")
    for i, rec in enumerate(selected, 1):
        safe_folder = re.sub(r"[^A-Za-z0-9_\-]+", "_", rec.folder)
        prefix = os.path.join(
            OUTPUT_DIR,
            f"task4_v2_continuous_{i:02d}_{safe_folder}_{rec.gesture}_epoch{rec.trial_index}",
        )
        fig_path, pred_csv, omni_csv = plot_one_record(rec, feature_names, bin_model, gesture_model, threshold, prefix)
        generated_prediction_csvs.append(pred_csv)
        print(f"  {i}. {fig_path}")
        print(f"     prediction CSV: {pred_csv}")
        print(f"     omnihand demo CSV: {omni_csv}")

    print("\n输出 Fig.3a 风格五指连续轨迹图：")
    fig3a_dir.mkdir(parents=True, exist_ok=True)
    for pred_csv in generated_prediction_csvs:
        csv_path = Path(pred_csv)
        out_path = fig3a_dir / (csv_path.stem.replace("_prediction", "") + "_fig3a_style_v2.png")
        saved = plot_fig3a_style(csv_path, out_path=out_path, show_hard=not args.hide_hard_fig3a)
        print(f"  {saved}")

    print("\n" + "=" * 80)
    print(f"完成！所有结果已保存到：{OUTPUT_DIR}")
    print(f"Fig.3a 风格图已保存到：{fig3a_dir}")
    print("=" * 80)



# =========================
# 9. Fig.3a 风格五指连续轨迹图
# =========================
FINGER_NAMES = ["thu.", "ind.", "mid.", "rin.", "lit."]
GESTURE_ORDER = ["grasp", "scissor", "six", "thumb", "index", "seven"]

# 0 = 完全伸展 / 张开，1 = 完全弯曲。
# 注意：这些模板只用于可视化，不是真实运动学测量。
GESTURE_TEMPLATES = {
    "rest":    np.array([0.00, 0.00, 0.00, 0.00, 0.00]),
    "open":    np.array([0.00, 0.00, 0.00, 0.00, 0.00]),
    "grasp":   np.array([0.95, 0.95, 0.95, 0.95, 0.95]),
    "scissor": np.array([0.55, 0.08, 0.08, 0.92, 0.92]),
    "six":     np.array([0.08, 0.92, 0.92, 0.92, 0.08]),
    "thumb":   np.array([0.05, 0.92, 0.92, 0.92, 0.92]),
    "index":   np.array([0.70, 0.05, 0.92, 0.92, 0.92]),
    "seven":   np.array([0.25, 0.22, 0.22, 0.88, 0.88]),
}

PHASES_FIG3A = [
    (0.0, 1.5, "Rest",   "#d9d9d9"),
    (1.5, 4.0, "Flex",   "#f3cccc"),
    (4.0, 5.5, "Hold",   "#efe3c6"),
    (5.5, 8.0, "Extend", "#d8ead6"),
]


def moving_average(arr: np.ndarray, win: int = 7) -> np.ndarray:
    """Centered moving average along axis 0, used only for Fig.3a-style visualization."""
    if win <= 1:
        return arr.copy()
    win = int(win)
    if win % 2 == 0:
        win += 1
    kernel = np.ones(win, dtype=float) / win

    if arr.ndim == 1:
        padded = np.pad(arr, (win // 2, win // 2), mode="edge")
        return np.convolve(padded, kernel, mode="valid")

    out = np.empty_like(arr, dtype=float)
    for i in range(arr.shape[1]):
        padded = np.pad(arr[:, i], (win // 2, win // 2), mode="edge")
        out[:, i] = np.convolve(padded, kernel, mode="valid")
    return out


def infer_true_gesture_for_fig3a(df: pd.DataFrame, path: Path) -> str:
    """Infer true gesture from non-rest labels or from file name."""
    labels = df["true_label"].astype(str).str.lower().tolist()
    non_rest = [x for x in labels if x != "rest"]
    if non_rest:
        return pd.Series(non_rest).mode().iloc[0]

    stem = path.stem.lower()
    for g in GESTURE_ORDER:
        if f"_{g}_" in stem or stem.endswith(g):
            return g
    return "grasp"


def phase_envelope_for_fig3a(t: np.ndarray) -> np.ndarray:
    """
    Build a proxy movement envelope according to the assignment paradigm:
    Rest 0-1.5s, Flex 1.5-4.0s, Hold 4.0-5.5s, Extend 5.5-8.0s.
    """
    env = np.zeros_like(t, dtype=float)

    mask = (t >= 1.5) & (t < 4.0)
    env[mask] = (t[mask] - 1.5) / (4.0 - 1.5)

    mask = (t >= 4.0) & (t < 5.5)
    env[mask] = 1.0

    mask = (t >= 5.5) & (t <= 8.0)
    env[mask] = 1.0 - (t[mask] - 5.5) / (8.0 - 5.5)

    return np.clip(env, 0.0, 1.0)


def build_ground_truth_trajectory(t: np.ndarray, true_gesture: str) -> np.ndarray:
    """Proxy ground-truth trajectory from known gesture + phase envelope."""
    template = GESTURE_TEMPLATES.get(true_gesture, GESTURE_TEMPLATES["grasp"])
    env = phase_envelope_for_fig3a(t)[:, None]
    return env * template[None, :]


def build_decoded_trajectories(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build decoded finger trajectories from v2 output CSV.

    Returns
    -------
    soft_traj : ndarray, shape (n_time, 5)
        Probability-weighted continuous finger trajectory.
    hard_traj : ndarray, shape (n_time, 5)
        Finger trajectory converted from pred_smooth hard labels.
    """
    prob_cols = [f"p_{g}" for g in GESTURE_ORDER]
    for c in prob_cols:
        if c not in df.columns:
            raise ValueError(f"Missing probability column: {c}")

    gesture_probs = df[prob_cols].to_numpy(dtype=float)

    if "p_movement" in df.columns:
        p_move = df["p_movement"].to_numpy(dtype=float)
    elif "p_rest" in df.columns:
        p_move = 1.0 - df["p_rest"].to_numpy(dtype=float)
    else:
        p_move = np.clip(gesture_probs.sum(axis=1), 0.0, 1.0)

    row_sum = gesture_probs.sum(axis=1, keepdims=True)
    norm_probs = gesture_probs / np.maximum(row_sum, 1e-8)
    template_mat = np.vstack([GESTURE_TEMPLATES[g] for g in GESTURE_ORDER])

    soft_shape = norm_probs @ template_mat
    soft_traj = soft_shape * p_move[:, None]
    soft_traj = moving_average(soft_traj, win=7)

    if "pred_smooth" in df.columns:
        hard_labels = df["pred_smooth"].astype(str).str.lower().tolist()
    elif "pred_raw" in df.columns:
        hard_labels = df["pred_raw"].astype(str).str.lower().tolist()
    else:
        hard_labels = ["rest"] * len(df)

    hard_traj = np.zeros((len(df), len(FINGER_NAMES)), dtype=float)
    for i, lab in enumerate(hard_labels):
        hard_traj[i] = GESTURE_TEMPLATES.get(lab, GESTURE_TEMPLATES["rest"])
    hard_traj = moving_average(hard_traj, win=5)

    return soft_traj, hard_traj


def plot_fig3a_style(
    csv_path: Path,
    out_path: Optional[Path] = None,
    show_hard: bool = True,
    line_scale: float = 0.82,
) -> Path:
    """
    Draw Fig.3a-style five-finger continuous trajectory from a v2 prediction CSV.

    Important:
    The black ground truth is a proxy template built from trial label and assignment timing.
    The blue trajectory is probability-to-template visualization, not true measured kinematics.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    required = {"time_s", "true_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    t = df["time_s"].to_numpy(dtype=float)
    true_gesture = infer_true_gesture_for_fig3a(df, csv_path)

    gt = build_ground_truth_trajectory(t, true_gesture)
    soft_traj, hard_traj = build_decoded_trajectories(df)

    fig, ax = plt.subplots(figsize=(13.5, 5.5), dpi=160)

    for start, end, name, color in PHASES_FIG3A:
        ax.axvspan(start, end, color=color, alpha=0.55, zorder=0)
    for x in [1.5, 4.0, 5.5]:
        ax.axvline(x, color="gray", linestyle="--", linewidth=1.0)

    offsets = np.arange(len(FINGER_NAMES) - 1, -1, -1, dtype=float)

    for off in offsets:
        ax.axhline(off, color="#c94d4d", linewidth=0.9, alpha=0.9, zorder=1)

    for i, (_, off) in enumerate(zip(FINGER_NAMES, offsets)):
        ax.plot(
            t,
            off + line_scale * gt[:, i],
            color="black",
            linewidth=2.0,
            label="Ground truth" if i == 0 else None,
            zorder=4,
        )
        if show_hard:
            ax.plot(
                t,
                off + line_scale * hard_traj[:, i],
                color="#2ca02c",
                linewidth=1.5,
                alpha=0.9,
                label="Hard decoded" if i == 0 else None,
                zorder=2,
            )
        ax.plot(
            t,
            off + line_scale * soft_traj[:, i],
            color="#4e79a7",
            linewidth=2.0,
            alpha=0.95,
            label="Soft decoded (v2)" if i == 0 else None,
            zorder=3,
        )

    ax.set_yticks(offsets)
    ax.set_yticklabels(FINGER_NAMES, fontsize=11)
    ax.set_ylim(-0.7, offsets.max() + 1.1)
    ax.set_xlim(float(np.min(t)), float(np.max(t)))
    ax.set_xlabel("time (s)", fontsize=12)
    ax.set_title(
        f"Task4 Fig.3a-style continuous decoding | {csv_path.stem}\n"
        f"True gesture = {true_gesture} | V2 probability-to-template trajectory",
        fontsize=13,
    )
    ax.grid(True, axis="x", alpha=0.25)
    ax.grid(False, axis="y")

    phase_handles = [
        Patch(facecolor=color, edgecolor="none", alpha=0.55, label=name)
        for _, _, name, color in PHASES_FIG3A
    ]
    line_handles, line_labels = ax.get_legend_handles_labels()
    handles = line_handles + phase_handles
    labels = line_labels + [h.get_label() for h in phase_handles]
    ax.legend(handles, labels, loc="upper center", ncol=4, frameon=True, fontsize=10)

    fig.tight_layout()

    if out_path is None:
        out_path = csv_path.with_name(csv_path.stem.replace("_prediction", "") + "_fig3a_style_v2.png")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def collect_fig3a_inputs(input_csv: Optional[str] = None, pattern: Optional[str] = None) -> List[Path]:
    """
    Collect prediction CSV files for Fig.3a-only mode.

    Priority:
    1. explicit input_csv if used by future extension
    2. --fig3a_glob
    3. default patterns under current working directory
    """
    files: List[Path] = []

    if input_csv:
        p = Path(input_csv)
        if p.exists():
            files.append(p)
        else:
            print(f"[Warning] input_csv does not exist: {p}")

    if pattern:
        matched = sorted(glob.glob(pattern))
        if not matched:
            print(f"[Warning] --fig3a_glob matched no files: {pattern}")
        files.extend(Path(x) for x in matched)

    if not files:
        default_patterns = [
            "results_task4_v2/*_prediction.csv",
            "*_prediction.csv",
            "../results_task4_v2/*_prediction.csv",
        ]
        print("[Info] No Fig.3a CSV pattern provided. Trying default search patterns:")
        for pat in default_patterns:
            matched = sorted(glob.glob(pat))
            print(f"  {pat}: {len(matched)} file(s)")
            files.extend(Path(x) for x in matched)

    seen = set()
    unique_files = []
    for f in files:
        f = Path(f)
        if f not in seen:
            unique_files.append(f)
            seen.add(f)
    return unique_files


if __name__ == "__main__":
    main()
