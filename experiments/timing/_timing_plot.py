import matplotlib.pyplot as plt 
import pickle 
import os
import numpy as np 

'''
Experiments for timing the forward process and backward process.
'''

def plot_stats(stats, xlabel='', ylabel='runtime [sec / sample]', save_path='', show=False, log_scale=True, with_std=False):
    plt.rcParams["font.size"] = 18   
    # plt.rcParams["font.family"] = "Times New Roman" 
    
    xs = list(stats.keys())
    fwd_runtimes = [runtime_d['fwd_runtime'] for runtime_d in stats.values()]
    bwd_runtimes = [runtime_d['bwd_runtime'] for runtime_d in stats.values()]
    plt.plot(xs, fwd_runtimes, linestyle='--', color='red', alpha=0.5, label='forward', marker='o', linewidth=4)
    plt.plot(xs, bwd_runtimes, linestyle='-', color='blue', alpha=0.5, label='backward', marker='s', linewidth=4)
    if with_std:
        fwd_runtimes_std = np.array([runtime_d['fwd_runtime_std'] for runtime_d in stats.values()])
        bwd_runtimes_std = np.array([runtime_d['bwd_runtime_std'] for runtime_d in stats.values()])
        plt.fill_between(xs, fwd_runtimes-fwd_runtimes_std, fwd_runtimes+fwd_runtimes_std, color=['red'], alpha=0.2)
        plt.fill_between(xs, bwd_runtimes-bwd_runtimes_std, bwd_runtimes+bwd_runtimes_std, color=['blue'], alpha=0.2)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    if log_scale: 
        # import matplotlib.ticker as ptick
        # plt.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))   # こっちを先に書くこと。
        # plt.ticklabel_format(style="sci", axis="y", scilimits=(3,3))   # 10^3（10の3乗）単位にする。
        plt.yscale('log')
        # plt.gca().yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
        # plt.gca().yaxis.set_minor_formatter(plt.ScalarFormatter(useMathText=True))
        # plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))
        
    plt.tight_layout()
    plt.plot()
    if save_path: plt.savefig(save_path)
    if show: plt.show()
    
if __name__ == '__main__':
    max_num_var = 5
    num_samples = 10
    exp_tag = f'timing_num_var_n={max_num_var}_N={num_samples}'
    with_std = False
    show = False
    
    save_dir = 'results/timing/'
    with open(os.path.join(save_dir, f'{exp_tag}.pkl'), 'rb') as f:
        stats = pickle.load(f)

    filename = f'{exp_tag}{"_with_std" if with_std else ""}.pdf'
    save_path = os.path.join(save_dir, filename)
    # save_path = None 
    plot_stats(stats, xlabel='# variables', ylabel='runtime [sec / sample]', save_path=save_path, show=show, log_scale=True, with_std=with_std)
    # # plt.xlabel('number of variables')