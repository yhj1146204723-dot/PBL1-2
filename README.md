# PBL1-2

## 📁 任务一：手部抓握运动神经活动分析（Task 1）

本部分主要完成了对 `single-movement` 数据中左手抓握和右手抓握任务的事件相关频谱扰动（ERSP）分析。按照作业要求，比较了**抓握运动尝试阶段**相对于**静息 Rest 阶段**的硬膜外皮质脑电（eECoG）频谱变化，并输出相应的时频图和频率响应曲线。

### 文件说明

* **`task1_ersp_pdf_rest_baseline.py`**：任务一核心代码脚本。用于读取 `single-movement` 数据，提取左手/右手抓握阶段与对应 Rest 阶段，计算 Movement / Rest 的 ERSP，并绘制时频图与频率曲线图。

* **`images/Fig_1c_Left_Grasp_vs_Rest_PDFRequirement.png`**：左手抓握阶段相对于 Rest 阶段的 8 通道 ERSP 时频图。该图展示了左手尝试抓握时，右半球 eECoG 中低频 beta 能量下降和 high gamma 能量增强的时间-频率变化。

* **`images/Fig_1d_Left_Right_vs_Rest_PDFRequirement.png`**：左手抓握与右手抓握在代表性通道 ch3 上的 ERSP 频率曲线对比图。该图展示了对侧左手抓握与同侧右手抓握在不同频段上的响应差异，尤其是 high gamma 频段的能量增强差异。

* **`Task1_Analysis.md`**：任务一分析说明文档。主要解释本任务的处理方法、两张结果图的含义，以及这些结果对后续连续抓握解码、多手势分类和海报展示的提示作用。

### 结果概述

任务一结果显示，左手尝试抓握阶段相对于 Rest 阶段，在右半球 eECoG 中表现出明显的运动相关频谱变化：

* beta 频段能量下降，表现为事件相关去同步化（ERD）；
* high gamma 频段能量增强，表现为事件相关同步化（ERS）；
* 左手抓握作为右半球电极的对侧运动，其 high gamma 响应强于右手抓握。

因此，后续任务中可以重点考虑使用 **50–100 Hz 或 50–150 Hz high gamma 能量** 作为连续解码和手势分类的重要特征。
