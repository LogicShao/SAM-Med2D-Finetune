import argparse
import os
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_log_file(log_path):
    """
    解析日志文件，提取 Epoch, Train Loss, 和 Validation Mean Dice。
    """
    # 更新后的正则表达式，以匹配您的日志格式
    # 匹配 "[...][INFO] Epoch 1, Train Loss: 0.9577"
    train_loss_regex = re.compile(r"\[INFO\] Epoch (\d+), Train Loss: (\d+\.\d+)")
    # 匹配 "[...][INFO] Epoch 1, Validation Mean Dice: 0.4857"
    val_dice_regex = re.compile(r"\[INFO\] Epoch (\d+), Validation Mean Dice: (\d+\.\d+)")

    data = []

    # 使用字典来临时存储每个epoch的数据，以确保Train Loss和Val Dice能正确配对
    epoch_data = {}

    with open(log_path, 'r') as f:
        for line in f:
            train_match = train_loss_regex.search(line)
            if train_match:
                epoch = int(train_match.group(1))
                train_loss = float(train_match.group(2))
                if epoch not in epoch_data:
                    epoch_data[epoch] = {}
                epoch_data[epoch]['Train Loss'] = train_loss
                continue

            val_match = val_dice_regex.search(line)
            if val_match:
                epoch = int(val_match.group(1))
                val_dice = float(val_match.group(2))
                if epoch not in epoch_data:
                    epoch_data[epoch] = {}
                epoch_data[epoch]['Validation Mean Dice'] = val_dice
                continue

    # 将字典转换为列表格式
    for epoch, values in sorted(epoch_data.items()):
        if 'Train Loss' in values and 'Validation Mean Dice' in values:
            data.append({
                'Epoch': epoch,
                'Train Loss': values['Train Loss'],
                'Validation Mean Dice': values['Validation Mean Dice']
            })

    if not data:
        raise ValueError(
            "No valid data pairs (Train Loss and Validation Dice) found in the log file. Please check the log format.")

    return pd.DataFrame(data)


def plot_curves(df, save_path, run_name):
    """
    使用解析出的 DataFrame 绘制训练曲线 (更新版)。
    """
    # 设置绘图风格
    sns.set_theme(style="whitegrid")

    # --- 1. 创建一个图包含两条曲线：训练损失 和 验证Dice ---
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # 设置标题
    fig.suptitle(f'Training & Validation Metrics for {run_name}', fontsize=16)

    # 绘制训练损失曲线 (左 Y 轴)
    color = 'tab:red'
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Train Loss', color=color, fontsize=12)
    line1 = sns.lineplot(data=df, x='Epoch', y='Train Loss', ax=ax1, color=color, marker='o', label='Train Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')

    # 创建第二个 Y 轴，共享 X 轴
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Validation Mean Dice', color=color, fontsize=12)
    line2 = sns.lineplot(data=df, x='Epoch', y='Validation Mean Dice', ax=ax2, color=color, marker='s',
                         label='Validation Mean Dice')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 1)  # Dice 分数在 0 到 1 之间
    ax2.legend(loc='upper right')

    # 找到并标记最佳 Mean Dice
    best_epoch = df.loc[df['Validation Mean Dice'].idxmax()]
    best_mean_dice = best_epoch['Validation Mean Dice']
    best_epoch_num = int(best_epoch['Epoch'])

    plt.axvline(x=best_epoch_num, color='g', linestyle='--',
                label=f'Best Mean Dice ({best_mean_dice:.4f}) at Epoch {best_epoch_num}')
    # 把最佳标记的图例也放在右上角
    handles, labels = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    # Manually add the vline to the legend
    vline_handle = plt.Line2D([0], [0], color='g', linestyle='--',
                              label=f'Best Mean Dice ({best_mean_dice:.4f}) at Epoch {best_epoch_num}')
    ax2.legend(handles=handles2 + [vline_handle], loc='upper right')

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # 调整布局以适应总标题

    # 保存图表
    fig_path = os.path.join(save_path, f"{run_name}_metrics_curves.png")
    plt.savefig(fig_path, dpi=300)
    print(f"Combined metrics curve saved to: {fig_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Parse a training log file and plot learning curves.")
    parser.add_argument("--log_path", type=str, required=True, help="Path to the training log file.")
    parser.add_argument("--save_dir", type=str, default="plots", help="Directory to save the plots.")
    args = parser.parse_args()

    run_name = os.path.splitext(os.path.basename(args.log_path))[0]
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"Parsing log file: {args.log_path}")
    df = parse_log_file(args.log_path)

    print("Log data parsed successfully:")
    print(df.head().to_string())

    print("\nPlotting curves...")
    plot_curves(df, args.save_dir, run_name)
    print("\nDone.")


if __name__ == "__main__":
    main()
