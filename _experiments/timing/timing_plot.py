import matplotlib.pyplot as plt 
import pickle 
import os
import numpy as np 

'''
Experiments for timing the forward process and backward process.
'''

def plot_stats(ns, stats, xlabel='', ylabel='runtime', save_path='', show=False, log_scale=True, with_std=False):
    plt.rcParams["font.size"] = 18   
    # plt.rcParams["font.family"] = "Times New Roman" 

    for key in stats:
        plt.plot(ns, stats[key], linestyle='--', alpha=0.5, label=key, marker='o', linewidth=4)        

    # fwd_runtimes = [runtime_d['fwd_runtime'] for runtime_d in stats.values()]
    # bwd_runtimes = [runtime_d['bwd_runtime'] for runtime_d in stats.values()]
    # plt.plot(xs, fwd_runtimes, linestyle='--', color='red', alpha=0.5, label='forward', marker='o', linewidth=4)
    # plt.plot(xs, bwd_runtimes, linestyle='-', color='blue', alpha=0.5, label='backward', marker='s', linewidth=4)
    if with_std:
        fwd_runtimes_std = np.array([runtime_d['fwd_runtime_std'] for runtime_d in stats.values()])
        bwd_runtimes_std = np.array([runtime_d['bwd_runtime_std'] for runtime_d in stats.values()])
        plt.fill_between(xs, fwd_runtimes-fwd_runtimes_std, fwd_runtimes+fwd_runtimes_std, color=['red'], alpha=0.2)
        plt.fill_between(xs, bwd_runtimes-bwd_runtimes_std, bwd_runtimes+bwd_runtimes_std, color=['blue'], alpha=0.2)

    plt.tight_layout()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.legend()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    if log_scale: 
        # import matplotlib.ticker as ptick
        # plt.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))   # こっちを先に書くこと。
        # plt.ticklabel_format(style="sci", axis="y", scilimits=(3,3))   # 10^3（10の3乗）単位にする。
        plt.yscale('log')
        # plt.gca().yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
        # plt.gca().yaxis.set_minor_formatter(plt.ScalarFormatter(useMathText=True))
        # plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))
        
    plt.plot()
    if save_path: plt.savefig(save_path, bbox_inches='tight')
    if show: plt.show()
    
    
if __name__ == '__main__':

    import yaml 
    
    ns = range(2, 6)
    # for postfix in ['', '_density=1']:
    for postfix in ['']:
        use_log_scale = bool(postfix)
        
        for char in [7, 31]:
            result_file = f'timing_results_char={char}{postfix}.yaml'
            # config_files = [os.path.basename(f'config/gb_dataset_n={n}_char={char}.yaml') for n in ns]
            # gb_algorithms = ['libsingular:std', 'libsingular:slimgb', 'libsingular:stdfglm']
            gb_algorithms = ['libsingular:std', 'libsingular:stdfglm']
            
            save_dir = 'results/timing/'
            with open(os.path.join(save_dir, result_file), 'r') as f:
                all_results = yaml.safe_load(f)
            
            config_files = list(all_results.keys())
            stats = {}
            keys = ['bwd_runtime'] + [f'fwd_runtime_{gb_algorithm}' for gb_algorithm in gb_algorithms]
            labels = ['backward'] + [f'forward ({gb_algorithm.split(":")[-1]})' for gb_algorithm in gb_algorithms]
            for k, label in zip(keys, labels):
                stats[label] = [all_results[config_file][k] for config_file in config_files]            

            num_samples = list(all_results.values())[0]['num_samples']
            exp_tag = f'timing_num_var_char={char}{postfix}_N={num_samples}_part'
            with_std = False
            show = False
            
            filename = f'{exp_tag}{"_with_std" if with_std else ""}.pdf'
            save_path = os.path.join(save_dir, filename)

            plot_stats(ns, stats, xlabel='# variables', ylabel='runtime [sec]', save_path=save_path, show=show, log_scale=use_log_scale, with_std=with_std)
            # # plt.xlabel('number of variables')
            plt.close()