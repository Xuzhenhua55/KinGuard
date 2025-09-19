import matplotlib.pyplot as plt
import numpy as np

# 设置字体和样式
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

# 数据
rates = ['0.9:0.1', '0.8:0.2', '0.7:0.3', '0.6:0.4', '0.5:0.5', 
         '0.4:0.6', '0.3:0.7', '0.2:0.8', '0.1:0.9']

# Task 数据
IF_task = [1, 1, 0.25, 0, 0, 0, 0, 0, 0]
ChainHash_task = [0.9, 0.9, 0.9, 0.9, 0.8, 0.6, 0.1, 0, 0]
ProFlingo_task = [1, 1, 0.98, 0.96, 0.88, 0.68, 0.64, 0.62, 0.52]
Ours_task = [1, 1, 1, 1, 1, 1, 1, 0.95, 0.83]

# Dare-Task 数据
IF_daretask = [1, 1, 0.125, 0, 0, 0, 0, 0, 0]
ChainHash_daretask = [0.9, 0.9, 0.9, 0.9, 0.8, 0.5, 0.1, 0, 0]
ProFlingo_daretask = [1, 1, 0.96, 0.94, 0.86, 0.68, 0.66, 0.64, 0.52]
Ours_daretask = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.88, 0.83, 0.83]

# Tie 数据
IF_tie = [0.125, 0, 0, 0, 0, 0, 0, 0, 0]
ChainHash_tie = [0, 0, 0, 0, 0, 0, 0, 0, 0]
ProFlingo_tie = [0.64, 0.64, 0.64, 0.64, 0.64, 0.64, 0.64, 0.64, 0.64]
Ours_tie = [0.96, 0.96, 0.96, 0.96, 0.92, 0.92, 0.92, 0.92, 0.92]

# Dare-Tie 数据
IF_daretie = [0.125, 0, 0, 0, 0, 0, 0, 0, 0]
ChainHash_daretie = [0.1, 0, 0, 0, 0, 0, 0, 0, 0]
ProFlingo_daretie = [0.32, 0.46, 0.38, 0.44, 0.4, 0.44, 0.36, 0.4, 0.4]
Ours_daretie = [1, 1, 1, 1, 1, 1, 1, 1, 1]

# 配色和样式
colors = ['#073068', '#EE3B2A', '#6BADD7', '#A60E16']  
markers = ['o', 's', '^', 'D']
line_styles = ['-', '--', '-.', ':']
labels = ['IF', 'Chain&Hash', 'ProFlingo', 'Ours']

# 创建2x2子图
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 绘制每个子图的函数
def plot_subplot(ax, data_list, title, show_ylabel=True, hide_xticks=False):
    for i, data in enumerate(data_list):
        ax.plot(rates, data, marker=markers[i], label=labels[i], linewidth=4, markersize=14,
                color=colors[i], linestyle=line_styles[i],
                markerfacecolor='white' if i != 3 else colors[i],
                markeredgecolor='white' if i == 3 else colors[i], markeredgewidth=2)
    ax.set_title(title, fontsize=20, fontweight='bold')
    if show_ylabel:
        ax.set_ylabel('Success Rate', fontsize=20, fontweight='bold')
    else:
        ax.set_yticklabels([])
        ax.set_ylabel('')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([0, len(rates)//2, len(rates)-1])
    if hide_xticks:
        ax.set_xticklabels([])
    else:
        ax.set_xticklabels([rates[0], rates[len(rates)//2], rates[-1]], fontsize=18)
    ax.grid(True, alpha=0.3, linestyle='-', color='gray', linewidth=1.5)
    ax.tick_params(axis='y', labelsize=18)
    ax.tick_params(axis='x', labelsize=18, rotation=0)

# 绘制四张图（上面两张隐藏横轴刻度）
plot_subplot(axes[0,0], [IF_task, ChainHash_task, ProFlingo_task, Ours_task], 'Task', show_ylabel=True, hide_xticks=True)
plot_subplot(axes[0,1], [IF_daretask, ChainHash_daretask, ProFlingo_daretask, Ours_daretask], 'Dare-Task', show_ylabel=False, hide_xticks=True)
plot_subplot(axes[1,0], [IF_tie, ChainHash_tie, ProFlingo_tie, Ours_tie], 'Tie', show_ylabel=True, hide_xticks=False)
plot_subplot(axes[1,1], [IF_daretie, ChainHash_daretie, ProFlingo_daretie, Ours_daretie], 'Dare-Tie', show_ylabel=False, hide_xticks=False)

# 美化所有子图
for ax in axes.flat:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_alpha(0.5)
    ax.spines['bottom'].set_alpha(0.5)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

plt.tight_layout(rect=[0, 0, 1, 1])

# 添加整体图例（放在顶部原本大标题位置）
handles, _ = axes[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 1.02),
           ncol=4, fontsize=20, frameon=False)

# 显示并保存
plt.show()
fig.savefig('multi_task_performance_comparison_updated.png', dpi=300, bbox_inches='tight', facecolor='white')
