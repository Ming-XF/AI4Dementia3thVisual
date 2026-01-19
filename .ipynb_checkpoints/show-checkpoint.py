import os
import re
import matplotlib.pyplot as plt
import numpy as np

import argparse

import pdb

parser = argparse.ArgumentParser()
parser.add_argument("--type", type=int, required=True, help="")
parser.add_argument("--path", type=str, default="./log_dir", required=False, help="")
args = parser.parse_args()

LOG_DIR = args.path       # 日志文件所在目录
OUT_DIR = "./analysis"

if args.type == 200:
    OUTPUT_IMG = 'multi_model_metrics_dementia.png'
    OUTPUT_HTML = 'model_metrics_comparison_dementia.html'
    DATASET = "Dementia"
    metrix = "auc"
elif args.type == 400:
    OUTPUT_IMG = 'multi_model_metrics_dementia400.png'
    OUTPUT_HTML = 'model_metrics_comparison_dementia400.html'
    DATASET = "Dementia400"
    metrix = "f_score"
elif args.type == 2000:
    OUTPUT_IMG = 'multi_model_metrics_dementia2000.png'
    OUTPUT_HTML = 'model_metrics_comparison_dementia2000.html'
    DATASET = "Dementia2000"
    metrix = "auc"
else:
    OUTPUT_IMG = 'multi_model_metrics_dementia4000.png'
    OUTPUT_HTML = 'model_metrics_comparison_dementia4000.html'
    DATASET = "Dementia4000"
    metrix = "f_score"

def generate_html_table(model_data):
    """生成对比不同方法的HTML表格（三线表格式）"""
    html = """
    <html>
    <head>
        <title>Model Metrics Comparison</title>
        <style>
            table {
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
                font-family: 'Times New Roman', Times, serif;
                font-size: 12pt;
                text-align: center;
            }
            /* 三线表样式 */
            .three-line-table {
                border: none;
            }
            .three-line-table thead th {
                border-bottom: 2px solid #000;
                border-top: 2px solid #000;
                padding: 8px 12px;
                font-weight: bold;
                text-align: center;
            }
            .three-line-table tbody td {
                border-bottom: 1px solid #ddd;
                padding: 6px 12px;
                border-left: none;
                border-right: none;
                text-align: center;
            }
            .three-line-table tbody tr:last-child td {
                border-bottom: 2px solid #000;
            }
            .three-line-table th.metric-header {
                border-top: 2px solid #000;
                border-bottom: 1px solid #ddd;
                text-align: center;
            }
            /* 精度指标下框线样式 */
            .accuracy-header {
                border-bottom: 2px solid #000 !important;
            }
            /* 性能最优加粗样式 */
            .best-metric {
                font-weight: bold;
                background-color: #f8f9fa;
                text-align: center;
            }
            /* 次优指标下划线样式 */
            .second-best-metric {
                text-decoration: underline;
                text-align: center;
            }
            /* 表头样式 */
            .table-caption {
                caption-side: top;
                text-align: center;
                font-weight: bold;
                margin-bottom: 10px;
                font-size: 14pt;
            }
            /* 模型名称单元格居中 */
            .model-name {
                text-align: center;
                font-weight: normal;
            }
            .model-name-best {
                text-align: center;
                font-weight: bold;
            }
        </style>
    </head>
    <body>
        <h1 style="text-align: center; font-family: 'Times New Roman', Times, serif;">
            Model Metrics Comparison (Average of Best Epochs from 3 Runs)
        </h1>
        <table class="three-line-table">
            <caption class="table-caption">Table 1: Comparison of Model Performance Metrics</caption>
            <thead>
                <tr>
                    <th rowspan="2" style="border-top: 2px solid #000; text-align: center;">Model</th>
                    <th colspan="5" class="metric-header" style="text-align: center;">Metrics (Average of Best Epochs)</th>
                </tr>
                <tr>
                    <th class="accuracy-header" style="text-align: center;">AUC</th>
                    <th class="accuracy-header" style="text-align: center;">Accuracy</th>
                    <th class="accuracy-header" style="text-align: center;">Precision</th>
                    <th class="accuracy-header" style="text-align: center;">Recall</th>
                    <th class="accuracy-header" style="text-align: center;">F-score</th>
                </tr>
            </thead>
            <tbody>
    """
    
    sorted_models = []
    for model_name, best_epochs in model_data.items():
        # 计算平均值
        avg_metrics = {
            'auc': np.mean([e['auc'] for e in best_epochs]),
            'accuracy': np.mean([e['accuracy'] for e in best_epochs]),
            'precision': np.mean([e['precision'] for e in best_epochs]),
            'recall': np.mean([e['recall'] for e in best_epochs]),
            'f_score': np.mean([e['f_score'] for e in best_epochs])
        }
        sorted_models.append((model_name, avg_metrics))

    # 按选择的指标排序（需要定义metrix变量）
    sorted_models.sort(key=lambda x: x[1][metrix], reverse=False)
    
    # 找出每个指标的最优值和次优值
    best_values = {}
    second_best_values = {}
    metric_names = ['auc', 'accuracy', 'precision', 'recall', 'f_score']
    
    for metric in metric_names:
        # 获取该指标的所有值并排序
        all_values = sorted([model[1][metric] for model in sorted_models], reverse=True)
        best_values[metric] = all_values[0] if all_values else 0
        # 如果有多个值，取次优值（去重后的第二个值）
        unique_values = sorted(set(all_values), reverse=True)
        if len(unique_values) > 1:
            second_best_values[metric] = unique_values[1]
        else:
            second_best_values[metric] = best_values[metric]  # 如果只有一个值，次优就是最优
    
    for rank, (model_name, avg_metrics) in enumerate(sorted_models, 1):
        # 检查每个指标是否为最优或次优
        metric_classes = []
        for metric in metric_names:
            if abs(avg_metrics[metric] - best_values[metric]) < 1e-6:
                # 最优指标：加粗
                metric_classes.append('best-metric')
            elif len([m[1][metric] for m in sorted_models]) > 1 and abs(avg_metrics[metric] - second_best_values[metric]) < 1e-6:
                # 次优指标：下划线（确保有多个不同的值）
                metric_classes.append('second-best-metric')
            else:
                metric_classes.append('')
        
        # 模型名称单元格样式
        model_name_class = 'model-name'
        
        html += f"""
                <tr>
                    <td class="{model_name_class}">{model_name}</td>
                    <td class="{metric_classes[0]}">{avg_metrics['auc']:.4f}</td>
                    <td class="{metric_classes[1]}">{avg_metrics['accuracy']:.4f}</td>
                    <td class="{metric_classes[2]}">{avg_metrics['precision']:.4f}</td>
                    <td class="{metric_classes[3]}">{avg_metrics['recall']:.4f}</td>
                    <td class="{metric_classes[4]}">{avg_metrics['f_score']:.4f}</td>
                </tr>
        """
    
    html += """
            </tbody>
        </table>
        <div style="margin-top: 10px; font-family: 'Times New Roman', Times, serif; font-size: 10pt; text-align: center;">
            <p><strong>Note:</strong> 
            <span style="font-weight: bold;">Bold values</span> indicate the best performance for each metric. 
            <span style="text-decoration: underline;">Underlined values</span> indicate the second best performance. 
            Higher values indicate better performance for all metrics.</p>
        </div>
    </body>
    </html>
    """
    
    return html

def parse_log(path):
    with open(path, 'r') as f:
        lines = f.readlines()

    auc_pattern = re.compile(r'AUC: ([\d.]+)')
    acc_pattern = re.compile(r'Accuracy: ([\d.]+)')
    prec_pattern = re.compile(r'Precision: ([\d.]+)')
    recall_pattern = re.compile(r'Recall: ([\d.]+)')
    fscore_pattern = re.compile(r'F_score: ([\d.]+)')
    test_loss_pattern = re.compile(r'Loss: ([\d.]+)')
    train_loss_pattern = re.compile(r'Train loss: ([\d.]+)')

    # 按 Repeat 分割
    train_blocks = ''.join(lines).split('########## Repeat:')[1:]  # 去掉开头的空字符串

    best_epochs_per_run = []
    all_epochs = []
    
    for block in train_blocks:
        aucs = [float(m) for m in auc_pattern.findall(block)]
        accs = [float(m) for m in acc_pattern.findall(block)]
        precs = [float(m) for m in prec_pattern.findall(block)]
        recalls = [float(m) for m in recall_pattern.findall(block)]
        fscores = [float(m) for m in fscore_pattern.findall(block)]
        test_losses = [float(m) for m in test_loss_pattern.findall(block)]
        train_losses = [float(m) for m in train_loss_pattern.findall(block)]

        # 动态确定epoch数量，取所有指标列表的最小长度
        num_epochs = min(len(aucs), len(accs), len(precs), len(recalls), 
                         len(fscores), len(test_losses), len(train_losses))
        
        if num_epochs == 0:
            print(f"警告: {path} 中某个训练块没有找到任何指标")
            continue
            
        # 确保每个指标数量一致，截取到最小长度
        epochs = [{
            'auc': aucs[i],
            'accuracy': accs[i],
            'precision': precs[i],
            'recall': recalls[i],
            'f_score': fscores[i],
            'test_loss': test_losses[i],
            'train_loss': train_losses[i]
        } for i in range(num_epochs)]
        
        # 找到当前训练中具有最大相关指标的epoch
        best_epoch = max(epochs, key=lambda x: x[metrix])
        best_epochs_per_run.append(best_epoch)
        
        # 收集所有epoch数据用于绘图
        all_epochs.append(epochs)

    return best_epochs_per_run, all_epochs

# ------------------ 主流程 ------------------

model_data1 = {}
model_data2 = {}
metrics = ['auc', 'accuracy', 'precision', 'recall', 'f_score', 'test_loss', 'train_loss']
labels = ['AUC', 'Accuracy', 'Precision', 'Recall', 'F-score', 'Test-loss', 'Train-loss']

for fname in os.listdir(LOG_DIR):
    if not fname.endswith(DATASET+'.log'):
        continue
    model_name = fname.replace('train_', '').replace('.log', '').replace("_"+DATASET, "")
    path = os.path.join(LOG_DIR, fname)
    best_epochs_per_run, epochs = parse_log(path)
    if len(best_epochs_per_run) >  0:
        print(f"{model_name} 中找到 {len(best_epochs_per_run)} 次训练")
        model_data1[model_name] = best_epochs_per_run
        model_data2[model_name] = epochs[-1]
    else:
        print(f"无法解析 {fname}，已跳过")

if not model_data1:
    print("没有找到任何可解析的日志文件")
    exit()

# 生成HTML表格
html_content = generate_html_table(model_data1)
with open(os.path.join(OUT_DIR, OUTPUT_HTML), 'w') as f:
    f.write(html_content)
print(f"HTML表格已保存：{OUTPUT_HTML}")
    
# ------------------ 绘图 ------------------
fig, axes = plt.subplots(3, 3, figsize=(18, 12))
fig.suptitle('Multi-Model Metrics Comparison (Best Metrix Repeat)', fontsize=16)

# 绘制所有7个指标
for i, (metric, label) in enumerate(zip(metrics, labels)):
    row = i // 3
    col = i % 3
    ax = axes[row, col]
    
    for model_name, epochs in model_data2.items():
        x = list(range(1, len(epochs) + 1))
        y = [e[metric] for e in epochs]
        ax.plot(x, y, label=model_name)
    ax.set_title(label)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(label)
    ax.grid(True)
    ax.legend()

# 隐藏最后一个空子图（如果有8个指标但只有7个需要显示）
axes[2, 2].set_visible(False)
axes[2, 1].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(OUT_DIR, OUTPUT_IMG), dpi=300)
print(f"图像已保存：{OUTPUT_IMG}")
# plt.show()