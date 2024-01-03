import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tensorboard.backend.event_processing import event_accumulator

def read_tensorboard_data(tensorboard_path, val_name):
    """读取tensorboard数据，
    tensorboard_path是tensorboard数据地址val_name是需要读取的变量名称"""
    ea = event_accumulator.EventAccumulator(tensorboard_path)
    ea.Reload()
    print(ea.scalars.Keys())
    val = ea.scalars.Items(val_name)
    return val
 
# def draw_plt(val_pairs, val_name):
    # """将数据绘制成曲线图，val是数据，val_name是变量名称"""
    # plt.figure()
    # for tag, val in val_pairs.items():
    #     plt.plot([i.step for i in val], [j.value for j in val], label=tag)
    # """横坐标是step，迭代次数
    # 纵坐标是变量值"""
    # plt.xlabel('step')
    # plt.ylabel(val_name)
    # plt.savefig('./results_plot', bbox_inches='tight')

log_path = './heart_disease/runs/fed/'
methods = ['privacy_free']
num_epochs = 50

kv_pairs = dict()
val_names = ['client3/test/AUC', 'client3/train/Loss']
all_results = dict()
for val_name in val_names:
    results = []
    for method in methods:
        tensorboard_path = log_path + method + '/'
        filename = sorted(os.listdir(tensorboard_path))[-1]
        print(filename)

        val = read_tensorboard_data(tensorboard_path + filename, val_name)
        val = [i.value for i in val]
        print(val)
        results.append(val)

    all_results[val_name] = results

plt.rcParams['figure.facecolor'] = 'white'
legend_font = {
    'family': 'sans-serif',  # 字体
    'style': 'normal',
    'size': 10,  # 字号
    'weight': "normal",  # 是否加粗，不加粗
}
label_font = {
    'family':'sans-serif',
    'size': 10,  # 字号
}
title_font = {
    'family':'sans-serif',
    'size': 10.5,  # 字号
    'weight': "bold",  # 是否加粗，不加粗
}

fig, axs = plt.subplots(1, 2, figsize=(6, 3), constrained_layout=True, dpi=500)

# subplot 1: test loss
df1 = pd.DataFrame(all_results[val_names[0]], columns=range(num_epochs), index=methods)
df1_T = pd.DataFrame(df1.T, columns=methods, index=range(num_epochs))
sns.lineplot(data=df1_T, ax=axs[0])
# axs[0].set(xscale="log", yscale="log")
# axs[0].get_legend().remove()
# axs[0].set_ylim(1e-6, 1e6)
# axs[0].set_ylabel(r'RDP $\rho(\alpha)$', **label_font)
# axs[0].set_xlabel(r'Orders $\alpha$', **label_font)
axs[0].tick_params(labelsize=9) #刻度
axs[0].set_title('Test AUC', **title_font)

# subplot 2: test acc
df2 = pd.DataFrame(all_results[val_names[1]], columns=range(num_epochs), index=methods)
df2_T = pd.DataFrame(df2.T, columns=methods, index=range(num_epochs))
sns.lineplot(data=df2_T, ax=axs[1])
# axs[1].set(xscale="log", yscale="log")
# axs[1].get_legend().remove()
# axs[1].set_ylabel(r'DP $\varepsilon(\alpha, \delta)$', **label_font)
# axs[1].set_ylim(0, 100000)
# axs[1].set_xlabel(r'Orders $\alpha$', **label_font)
axs[1].tick_params(labelsize=9)
axs[1].set_title('Test Loss', **title_font)

# handles, labels = axs[0].get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper center', bbox_to_anchor = (0, 0.12, 1, 1), ncol=len(labels), prop=legend_font)
plt.savefig(log_path+'heart_disease_fed.pdf', dpi=500, bbox_inches='tight')