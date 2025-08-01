import numpy as np
import matplotlib.pyplot as plt
from math import pi
import pandas as pd

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 学术论文标准设置
plt.rcParams.update({
    'font.size': 12,
    'axes.linewidth': 1,
    'lines.linewidth': 2,
    'patch.linewidth': 1,
    'grid.linewidth': 0.8,
    'xtick.major.width': 1,
    'ytick.major.width': 1,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

# 原始数据
data = {
    'LightTime': {
        'Yearly': {'SMAPE': 13.329, 'MASE': 2.995, 'OWA': 0.785},
        'Quarterly': {'SMAPE': 10.092, 'MASE': 1.182, 'OWA': 0.889},
        'Monthly': {'SMAPE': 12.651, 'MASE': 0.932, 'OWA': 0.877},
        'Others': {'SMAPE': 4.847, 'MASE': 3.304, 'OWA': 1.031},
        'Average': {'SMAPE': 11.803, 'MASE': 1.585, 'OWA': 0.850}
    },
    'TimesNet': {
        'Yearly': {'SMAPE': 13.387, 'MASE': 2.996, 'OWA': 0.786},
        'Quarterly': {'SMAPE': 10.100, 'MASE': 1.182, 'OWA': 0.890},
        'Monthly': {'SMAPE': 12.670, 'MASE': 0.933, 'OWA': 0.878},
        'Others': {'SMAPE': 4.891, 'MASE': 3.302, 'OWA': 1.035},
        'Average': {'SMAPE': 11.829, 'MASE': 1.585, 'OWA': 0.851}
    },
    'DLinear': {
        'Yearly': {'SMAPE': 16.965, 'MASE': 4.283, 'OWA': 1.058},
        'Quarterly': {'SMAPE': 12.148, 'MASE': 1.520, 'OWA': 1.106},
        'Monthly': {'SMAPE': 13.514, 'MASE': 1.037, 'OWA': 0.956},
        'Others': {'SMAPE': 6.709, 'MASE': 4.953, 'OWA': 1.487},
        'Average': {'SMAPE': 13.639, 'MASE': 2.095, 'OWA': 1.051}
    },
    'FEDFormer': {
        'Yearly': {'SMAPE': 13.728, 'MASE': 3.078, 'OWA': 0.807},
        'Quarterly': {'SMAPE': 10.792, 'MASE': 1.283, 'OWA': 0.958},
        'Monthly': {'SMAPE': 13.917, 'MASE': 1.097, 'OWA': 0.998},
        'Others': {'SMAPE': 6.302, 'MASE': 4.064, 'OWA': 1.304},
        'Average': {'SMAPE': 12.780, 'MASE': 1.701, 'OWA': 0.918}
    },
    'LightTS': {
        'Yearly': {'SMAPE': 14.247, 'MASE': 3.109, 'OWA': 0.827},
        'Quarterly': {'SMAPE': 11.364, 'MASE': 1.328, 'OWA': 1.000},
        'Monthly': {'SMAPE': 14.014, 'MASE': 1.053, 'OWA': 0.981},
        'Others': {'SMAPE': 15.880, 'MASE': 11.434, 'OWA': 3.474},
        'Average': {'SMAPE': 13.525, 'MASE': 2.111, 'OWA': 1.051}
    },
    'Autoformer': {
        'Yearly': {'SMAPE': 13.974, 'MASE': 3.134, 'OWA': 0.822},
        'Quarterly': {'SMAPE': 11.338, 'MASE': 1.365, 'OWA': 1.012},
        'Monthly': {'SMAPE': 13.958, 'MASE': 1.103, 'OWA': 1.002},
        'Others': {'SMAPE': 5.485, 'MASE': 3.865, 'OWA': 1.187},
        'Average': {'SMAPE': 12.909, 'MASE': 1.771, 'OWA': 0.939}
    }
}

def normalize_data_for_radar(data_dict, frequency='Average'):
    """
    标准化数据用于雷达图显示
    由于所有指标都是越小越好，我们使用相对排名进行转换
    """
    models = list(data_dict.keys())
    metrics = ['SMAPE', 'MASE', 'OWA']
    
    # 提取指定频率的数据
    values = {}
    for model in models:
        values[model] = [data_dict[model][frequency][metric] for metric in metrics]
    
    # 找到每个指标的最小值和最大值
    min_values = []
    max_values = []
    for i in range(len(metrics)):
        metric_values = [values[model][i] for model in models]
        min_val = min(metric_values)
        max_val = max(metric_values)
        min_values.append(min_val)
        max_values.append(max_val)
    
    # 使用Min-Max标准化，然后取倒数（因为越小越好）
    normalized_values = {}
    for model in models:
        normalized_values[model] = []
        for i, val in enumerate(values[model]):
            if max_values[i] == min_values[i]:
                # 如果所有值相同，设为0.5
                normalized_val = 0.5
            else:
                # Min-Max标准化：(max - val) / (max - min)，这样小的值变成大的值
                normalized_val = (max_values[i] - val) / (max_values[i] - min_values[i])
            # 将范围调整到0.2-1.0，避免中心空白过多
            normalized_val = 0.2 + 0.8 * normalized_val
            normalized_values[model].append(normalized_val)
    
    return normalized_values, min_values, max_values

def create_radar_chart(data_dict, frequency='Average', save_path=None):
    """
    创建学术标准的雷达图
    """
    # 标准化数据
    normalized_data, min_values, max_values = normalize_data_for_radar(data_dict, frequency)
    
    # 设置雷达图参数
    metrics = ['SMAPE', 'MASE', 'OWA']
    N = len(metrics)
    
    # 计算角度
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # 闭合多边形
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
    
    # 学术论文常用颜色（色盲友好）
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'D', 'v', 'p']
    linestyles = ['-', '--', '-.', ':', '-', '--']
    
    # 绘制每个模型
    for i, (model, values) in enumerate(normalized_data.items()):
        values += values[:1]  # 闭合多边形
        ax.plot(angles, values, 
               color=colors[i % len(colors)], 
               linewidth=2.5,
               linestyle=linestyles[i % len(linestyles)],
               marker=markers[i % len(markers)],
               markersize=8,
               label=model,
               alpha=0.8)
        
        # 填充区域（透明度较低）
        ax.fill(angles, values, 
               color=colors[i % len(colors)], 
               alpha=0.15)
    
    # 设置标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=14, fontweight='bold')
    
    # 设置网格 - 调整Y轴范围以更好显示数据
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['Best', '0.4', '0.6', '0.8', 'Worst'], fontsize=11)
    ax.grid(True, alpha=0.6)
    
    # 添加径向网格线
    ax.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0], alpha=0.6)
    
    # 设置标题
    plt.title(f'Time Series Forecasting Models Performance\n({frequency} Frequency)', 
             fontsize=16, fontweight='bold', pad=30)
    
    # 设置图例
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), 
              fontsize=12, frameon=True, fancybox=True, shadow=True)
    
    # 添加说明文本
    fig.text(0.5, 0.02, 
            'Note: Values are normalized (higher is better). Original metrics: lower is better.',
            ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"图表已保存至: {save_path}")
    
    plt.show()

def create_comparison_table(data_dict, frequency='Average'):
    """
    创建模型比较表格，包含排名信息
    """
    models = list(data_dict.keys())
    metrics = ['SMAPE', 'MASE', 'OWA']
    
    # 创建DataFrame
    df_data = []
    for model in models:
        row = [model]
        for metric in metrics:
            value = data_dict[model][frequency][metric]
            row.append(f"{value:.3f}")
        df_data.append(row)
    
    df = pd.DataFrame(df_data, columns=['Model'] + metrics)
    
    # 计算每个指标的排名（越小越好）
    for metric in metrics:
        values = [data_dict[model][frequency][metric] for model in models]
        ranks = pd.Series(values).rank(method='min').astype(int)
        df[f'{metric}_Rank'] = ranks
    
    print(f"\n=== {frequency} Frequency Performance Comparison ===")
    print(df.to_string(index=False))
    print("\nNote: For all metrics, lower values indicate better performance.")
    print("Rank: 1 = best, 6 = worst")

# 主函数
def main():
    # 创建不同频率的雷达图
    frequencies = ['Average', 'Yearly', 'Quarterly', 'Monthly', 'Others']
    
    for freq in frequencies:
        print(f"\n正在生成 {freq} 频率的雷达图...")
        create_radar_chart(data, frequency=freq, 
                         save_path=f'radar_chart_{freq.lower()}.png')
        create_comparison_table(data, frequency=freq)

# 如果需要单独运行某个频率的图表，可以使用以下代码：
if __name__ == "__main__":
    # 生成平均性能雷达图
    create_radar_chart(data, frequency='Average', save_path='radar_chart_average.png')
    create_comparison_table(data, frequency='Average')
    
    # 如果需要生成所有频率的图表，取消下面的注释
    main()