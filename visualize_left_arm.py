import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import CheckButtons
import argparse

# Configuration
parser = argparse.ArgumentParser(description='Visualize robot arm joint positions over time')
parser.add_argument('--arm', type=str, default='right', choices=['left', 'right'],
                    help='Which arm to visualize: left or right (default: left)')
args = parser.parse_args()

ARM_SIDE = args.arm  # 'left' or 'right'

# 解析数据文件
def load_data(filename, arm_side='left'):
    data = {f'{arm_side}_arm_joint_{i}': [] for i in range(1, 8)}
    chunk_boundaries = []
    chunk_states = []  # 存储每个chunk开头的state
    action_index = 0
    current_chunk = 0

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()

            # 检测chunk标记
            if line.startswith('=== Chunk'):
                if current_chunk > 0:
                    chunk_boundaries.append(action_index)
                current_chunk += 1
                continue

            # 解析Robot State行
            if line.startswith('Robot State'):
                try:
                    # 提取方括号内的数据
                    state_str = line[line.index('['):line.rindex(']')+1]
                    state_data = eval(state_str)

                    # 根据arm_side提取对应的7个关节state
                    if arm_side == 'left':
                        # 左臂是前7个关节
                        arm_state = state_data[0:7]
                    else:  # right
                        # 右臂是第8-14个关节（索引7-13）
                        arm_state = state_data[7:14]

                    chunk_states.append({
                        'action_index': action_index,
                        'state': arm_state
                    })
                except Exception as e:
                    print(f'Parse state error: {e}')

            # 解析action行
            if line.startswith('Action'):
                try:
                    json_str = line[line.index('{'):line.rindex('}')+1]
                    # 简单解析
                    action_data = eval(json_str)

                    # 提取指定手臂的关节数据
                    for joint in range(1, 8):
                        key = f'{arm_side}_arm_joint_{joint}.pos'
                        if key in action_data:
                            data[f'{arm_side}_arm_joint_{joint}'].append(action_data[key])

                    action_index += 1
                except Exception as e:
                    print(f'Parse error: {e}')

    return data, chunk_boundaries, chunk_states

# 加载数据
print("Loading data...")
data, chunk_boundaries, chunk_states = load_data('record_chunk.txt', ARM_SIDE)

# 创建图表
fig, ax = plt.subplots(figsize=(14, 8))
plt.subplots_adjust(left=0.08, right=0.85, top=0.95, bottom=0.08)

# 颜色方案
colors = ['#C4612F', '#2E7D32', '#1976D2', '#7B1FA2', '#D32F2F', '#F57C00', '#0097A7']

# 绘制每个关节的曲线
lines = []
labels = []
for joint in range(1, 8):
    joint_name = f'{ARM_SIDE}_arm_joint_{joint}'
    x = np.arange(len(data[joint_name]))
    line, = ax.plot(x, data[joint_name], label=f'Joint {joint}',
                    color=colors[joint-1], linewidth=2, alpha=0.9)
    lines.append(line)
    labels.append(f'Joint {joint}')

# 添加chunk边界的红色虚线，并标注差值
for boundary in chunk_boundaries:
    ax.axvline(x=boundary, color='red', linestyle='--', linewidth=2, alpha=0.7)

    # 计算每个关节在边界处的差值
    for joint in range(1, 8):
        joint_name = f'{ARM_SIDE}_arm_joint_{joint}'
        if boundary > 0 and boundary < len(data[joint_name]):
            prev_value = data[joint_name][boundary - 1]  # 上一个chunk的末尾
            curr_value = data[joint_name][boundary]       # 当前chunk的开头
            diff = curr_value - prev_value

            # 在边界处标注差值
            y_pos = curr_value
            ax.text(boundary, y_pos, f'{diff:.1f}',
                   fontsize=8, color=colors[joint-1],
                   ha='left', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor=colors[joint-1], alpha=0.7))

# 在每个chunk开始处标注state值
for chunk_state in chunk_states:
    action_idx = chunk_state['action_index']
    state_values = chunk_state['state']

    # 为每个关节在chunk开始处画一个标记点
    for joint in range(1, 8):
        joint_name = f'{ARM_SIDE}_arm_joint_{joint}'
        state_value = state_values[joint - 1]

        # 画一个圆圈标记state位置
        ax.plot(action_idx, state_value, 'o',
               color=colors[joint-1], markersize=6,
               markerfacecolor='white', markeredgewidth=2,
               zorder=10)

# 设置标签和标题
ax.set_xlabel('Action Index', fontsize=12)
ax.set_ylabel('Joint Position (degrees)', fontsize=12)
ax.set_title(f'{ARM_SIDE.capitalize()} Arm 7 Joints Over Time (Red dashed lines mark chunk boundaries)', fontsize=14, pad=15)
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(loc='upper left', fontsize=10)

# 添加交互式复选框
rax = plt.axes([0.87, 0.4, 0.12, 0.3])
visibility = [True] * 7
check = CheckButtons(rax, labels, visibility)

def toggle_visibility(label):
    index = labels.index(label)
    lines[index].set_visible(not lines[index].get_visible())
    plt.draw()

check.on_clicked(toggle_visibility)

print(f"Data loaded: {len(data[f'{ARM_SIDE}_arm_joint_1'])} actions in total")
print(f"Chunk boundaries at: {chunk_boundaries}")
print("\nTip: Click checkboxes on the right to show/hide curves")

plt.show()
