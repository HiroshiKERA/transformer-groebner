import os
import yaml
from glob import glob

def read_yaml(file_path):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def process_directory(base_path):
    dirs = glob(os.path.join(base_path, 'shape_n=*'))
    print(dirs)
    for dir_path in sorted(dirs):
        dir_name = os.path.basename(dir_path)
        # config = dict(item.split('=') for item in dir_name.split('_')[1:])
        
        f_stats_file = glob(os.path.join(dir_path, 'dataset_stats_F_summary_data_*_test.yaml'))
        g_stats_file = glob(os.path.join(dir_path, 'dataset_stats_G_summary_data_*_test.yaml'))
        
        if f_stats_file and g_stats_file:
            stats_f = read_yaml(f_stats_file[0])
            stats_g = read_yaml(g_stats_file[0])
            
            print(f"\nConfiguration: {dir_name}")
            print("F statistics:")
            metrics = {
                'Size of F': ('size_mean', 'size_std'),
                'Max degree in F': ('max_degree_mean', 'max_degree_std'),
                'Min degree in F': ('min_degree_mean', 'min_degree_std'),
                '# of terms in F': ('total_num_monoms_mean', 'total_num_monoms_std'),
                'GB ratio': ('is_GB_mean', 'is_GB_std')
            }
            
            for metric, (mean_key, std_key) in metrics.items():
                if mean_key in stats_f and std_key in stats_f:
                    print(f"{metric}: {stats_f[mean_key]:.3f} (±{stats_f[std_key]:.3f})")

            print("G statistics:")
            metrics = {
                'Size of G': ('size_mean', 'size_std'),
                'Max degree in G': ('max_degree_mean', 'max_degree_std'),
                'Min degree in G': ('min_degree_mean', 'min_degree_std'),
                '# of terms in G': ('total_num_monoms_mean', 'total_num_monoms_std'),
                'GB ratio': ('is_GB_mean', 'is_GB_std')
            }
            
            for metric, (mean_key, std_key) in metrics.items():
                if mean_key in stats_g and std_key in stats_g:
                    print(f"{metric}: {stats_g[mean_key]:.3f} (±{stats_g[std_key]:.3f})")

if __name__ == "__main__":
    process_directory("/app/data/shape")