import mne
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import welch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 1. 配置区
# ============================================================
DATA_ROOT = "data"          # 你的数据根目录
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 手势映射（与之前一致）
HAND_MAP = {
    'A': {'gestures': ['scissor', 'six', 'grasp'], 'b_triggers': ['1', '4', '7']},
    'B': {'gestures': ['index', 'seven', 'thumb'], 'b_triggers': ['1', '4', '7']}
}
ALL_GESTURE_NAMES = ['grasp', 'scissor', 'six', 'thumb', 'index', 'seven']
LABEL_MAP = {name: idx for idx, name in enumerate(ALL_GESTURE_NAMES)}

# 频段定义（全频段）
BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 50), (50, 100)]
# 如果只想用 High Gamma，可以改为 BANDS = [(50, 100)]

print("=" * 70)
print("开始提取数据并进行手势分类...")
print("=" * 70)

# ============================================================
# 2. 数据提取（同之前的 task3_full_analysis.py）
# ============================================================
all_epochs_data = {name: [] for name in ALL_GESTURE_NAMES}

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

# 统计
print("\n数据提取完成，各手势试次数量：")
for name in ALL_GESTURE_NAMES:
    print(f"  {name:>10}: {len(all_epochs_data[name])} 个试次")

# ============================================================
# 3. 特征提取：切取 Flex 阶段，计算频带功率
# ============================================================
print("\n开始提取 Flex 阶段特征...")

X_all = []
y_all = []

for gesture_name in ALL_GESTURE_NAMES:
    data_list = all_epochs_data[gesture_name]
    for ep_data in data_list:  # (8, 8000)
        # 切取 Flex 阶段：1.5s ~ 4.0s (索引 1500~4000)
        flex_data = ep_data[:, 1500:4000]  # (8, 2500)
        
        # 提取特征：对每个通道、每个频段计算平均功率
        feature_vector = []
        for ch in range(flex_data.shape[0]):
            freqs, psd = welch(flex_data[ch], fs=1000, nperseg=256)
            for fmin, fmax in BANDS:
                mask = (freqs >= fmin) & (freqs <= fmax)
                if np.any(mask):
                    band_power = np.mean(psd[mask])
                else:
                    band_power = 0.0
                feature_vector.append(band_power)
        
        X_all.append(feature_vector)
        y_all.append(LABEL_MAP[gesture_name])

X_all = np.array(X_all)
y_all = np.array(y_all)

print(f"总样本数: {len(X_all)}, 特征维度: {X_all.shape[1]}")
print(f"各类别样本数: {np.bincount(y_all)}")

# ============================================================
# 4. 划分训练集和测试集
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)
print(f"训练集: {len(X_train)}, 测试集: {len(X_test)}")

# ============================================================
# 5. 标准化
# ============================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# 6. 分类器训练与评估 (SVM + RandomForest 对比)
# ============================================================
print("\n" + "=" * 70)
print("训练 SVM 分类器...")
clf_svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
clf_svm.fit(X_train_scaled, y_train)
y_pred_svm = clf_svm.predict(X_test_scaled)
acc_svm = accuracy_score(y_test, y_pred_svm)
print(f"SVM 准确率: {acc_svm * 100:.2f}%")
print("\nSVM 分类报告:")
print(classification_report(y_test, y_pred_svm, target_names=ALL_GESTURE_NAMES))

print("\n训练 RandomForest 分类器...")
clf_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
clf_rf.fit(X_train_scaled, y_train)
y_pred_rf = clf_rf.predict(X_test_scaled)
acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"RandomForest 准确率: {acc_rf * 100:.2f}%")
print("\nRandomForest 分类报告:")
print(classification_report(y_test, y_pred_rf, target_names=ALL_GESTURE_NAMES))

# ============================================================
# 7. 选择最佳分类器并绘制混淆矩阵
# ============================================================
if acc_svm >= acc_rf:
    best_clf = clf_svm
    y_pred_best = y_pred_svm
    best_name = f"SVM ({acc_svm*100:.1f}%)"
else:
    best_clf = clf_rf
    y_pred_best = y_pred_rf
    best_name = f"RandomForest ({acc_rf*100:.1f}%)"

cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=ALL_GESTURE_NAMES,
            yticklabels=ALL_GESTURE_NAMES)
plt.title(f'Confusion Matrix - Best: {best_name}')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'Confusion_Matrix.png'), dpi=150)
plt.close()
print(f"混淆矩阵已保存至 results/Confusion_Matrix.png")

# 保存结果摘要
with open(os.path.join(OUTPUT_DIR, 'classification_summary.txt'), 'w') as f:
    f.write("=" * 60 + "\n")
    f.write("分类结果总结\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"总样本数: {len(X_all)}\n")
    f.write(f"特征维度: {X_all.shape[1]}\n")
    f.write(f"训练集样本: {len(X_train)}, 测试集样本: {len(X_test)}\n")
    f.write(f"SVM 准确率: {acc_svm*100:.2f}%\n")
    f.write(f"RandomForest 准确率: {acc_rf*100:.2f}%\n")
    f.write(f"最佳模型: {best_name}\n")

print("\n" + "=" * 70)
print(f"🎉 分类完成！最佳准确率: {max(acc_svm, acc_rf)*100:.2f}%")
print(f"   结果保存在 results/ 目录下。")
print("=" * 70)