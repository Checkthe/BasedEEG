import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score
from torchmetrics import Accuracy
from config import cfg

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号


def plot_confusion_matrix(y_true, y_pred, class_names, title="混淆矩阵",
                          normalize=False, cmap='Blues', save_path=None):
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if normalize:
        cm_display = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    else:
        cm_display = cm

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(cm_display, annot=True, fmt='.2f' if normalize else 'd',
                     cmap=cmap, xticklabels=class_names, yticklabels=class_names,
                     cbar_kws={'label': '样本数量'}, square=True, linewidths=0.5,
                     annot_kws={'size':12})
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('预测标签', fontsize=12)
    plt.ylabel('真实标签', fontsize=12)

    accuracy = np.trace(cm) / np.sum(cm) if np.sum(cm) > 0 else 0
    plt.text(0.5, -0.15, f'总体准确率: {accuracy:.3f}',
             transform=ax.transAxes, ha='center', fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))

    thresh = cm_display.max() / 2.
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j+0.5, i+0.5, f"{cm_display[i,j]:.2f}" if normalize else str(cm[i,j]),
                    ha="center", va="center",
                    color="white" if cm_display[i,j] > thresh else "black",
                    fontsize=12)

    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}", dpi=300, bbox_inches='tight')
    plt.show()

    return cm, accuracy


def plot_metrics_bar(metrics, title="模型性能指标", save_path=None):
    """
    绘制性能指标柱状图
    """
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(metric_names, metric_values,
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'],
                   alpha=0.8, edgecolor='black', linewidth=1)

    # 添加数值标签
    for bar, value in zip(bars, metric_values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('评估指标', fontsize=12)
    plt.ylabel('分数', fontsize=12)
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.3)

    # 添加基准线
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='基准线 (0.5)')
    plt.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_roc_curve(y_true, y_scores, title="ROC曲线", save_path=None):
    """
    绘制ROC曲线
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC曲线 (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='随机分类器')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正率 (False Positive Rate)', fontsize=12)
    plt.ylabel('真正率 (True Positive Rate)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    return roc_auc


def plot_precision_recall_curve(y_true, y_scores, title="精确率-召回率曲线", save_path=None):
    """
    绘制精确率-召回率曲线
    """
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    avg_precision = average_precision_score(y_true, y_scores)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'PR曲线 (AP = {avg_precision:.3f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('召回率 (Recall)', fontsize=12)
    plt.ylabel('精确率 (Precision)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    return avg_precision


def plot_prediction_distribution(y_scores, y_true, title="预测概率分布", save_path=None):
    """
    绘制预测概率分布直方图
    """
    plt.figure(figsize=(10, 6))

    # 分别绘制正负样本的概率分布
    pos_scores = y_scores[y_true == 1]
    neg_scores = y_scores[y_true == 0]

    plt.hist(neg_scores, bins=30, alpha=0.7, label='负样本', color='lightcoral')
    plt.hist(pos_scores, bins=30, alpha=0.7, label='正样本', color='lightblue')

    plt.xlabel('预测概率', fontsize=12)
    plt.ylabel('频数', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

from sklearn.preprocessing import label_binarize

def evaluate_model(model, test_loader, device='auto', detailed_report=True,
                         plot_results=True, save_dir=None, class_names=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
    model = model.to(device)
    cfgs = cfg().get_args()

    all_preds = []
    all_labels = []
    all_scores = []

    model.eval()
    with torch.no_grad():
        for eeg, labels, bp in test_loader:
            eeg, labels, bp = (eeg.to(device, non_blocking=True),
                               labels.to(device, non_blocking=True),
                               bp.to(device, non_blocking=True))
            outputs = model(bp)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    num_classes = len(np.unique(all_labels))
    if class_names is None:
        class_names = [f"类别{i}" for i in range(num_classes)]

    # 使用torchmetrics计算准确率
    acc_metric = Accuracy(task='multiclass', num_classes=num_classes)
    metrics = {
        'accuracy': acc_metric(torch.tensor(all_preds), torch.tensor(all_labels)).item(),
        'balanced_accuracy': balanced_accuracy_score(all_labels, all_preds),
        'cohen_kappa': cohen_kappa_score(all_labels, all_preds),
        'weighted_f1': f1_score(all_labels, all_preds, average='weighted')
    }

    if detailed_report:
        print("\n=== 分类报告 ===")
        print(classification_report(all_labels, all_preds, target_names=class_names))
        print("\n=== 混淆矩阵 ===")
        print(confusion_matrix(all_labels, all_preds))

    if plot_results:
        # 混淆矩阵
        plot_confusion_matrix(all_labels, all_preds, class_names,
                              title="混淆矩阵",
                              save_path=f"{save_dir}/confusion_matrix_multi.png" if save_dir else None)

        # 性能柱状图
        plot_metrics_bar(metrics, title="模型性能指标",
                         save_path=f"{save_dir}/metrics_bar_multi.png" if save_dir else None)

        # 如果只考虑部分类的 ROC/PR 分析，可自行实现 one-vs-rest
        # 若全类绘制 ROC / PR，可使用 sklearn.metrics.roc_curve in one-vs-rest 形式
        # 示例：
        # y_bin = label_binarize(all_labels, classes=range(num_classes))
        # fpr, tpr, _ = roc_curve(y_bin.ravel(), all_scores.ravel())

    return metrics
