import os
import yaml
from glob import glob
from collections import defaultdict

def read_timing_results(base_path, print_fg=False):
    patterns = [
        "shape_n=[45]_field=*density=*",
        "shape_n=[45]_field=GF*",
        "shape_n=[45]_field=QQ*"
    ]
    
    results = defaultdict(list)
    
    for pattern in patterns:
        dirs = glob(os.path.join(base_path, pattern))
        for dir_path in sorted(dirs):
            dir_name = os.path.basename(dir_path)
            yaml_file = os.path.join(dir_path, "transformer_supermacy_timeout=100.0.yaml")
            
            if not os.path.exists(yaml_file):
                continue
                
            with open(yaml_file, 'r') as f:
                data = yaml.safe_load(f)
                
            config = dir_name.replace("shape_", "")
            for item in data:
                if 'runtime' not in item or 'success' not in item:
                    continue
                
                if item['success']['libsingular:std'] and item['success']['libsingular:slimgb'] and item['success']['libsingular:stdfglm']:
                    continue
                
                if not item['success']['transformer']:
                    continue 
                
                runtime_data = {
                    "std": item['runtime']['libsingular:std'],
                    "slimgb": item['runtime']['libsingular:slimgb'],
                    "stdfglm": item['runtime']['libsingular:stdfglm'],
                    "transformer": item['runtime']['transformer']
                }
                
                success_data = {
                    "std": item['success']['libsingular:std'],
                    "slimgb": item['success']['libsingular:slimgb'],
                    "stdfglm": item['success']['libsingular:stdfglm'],
                    "transformer": item['success']['transformer']
                }
                
                result = {
                    'id': item['id'],
                    "config": config,
                    "runtime": runtime_data,
                    "success": success_data
                }
                
                if print_fg:
                    result["F"] = item.get('F', '')
                    result["G"] = item.get('G', '')
                
                results[config].append(result)
                
    # Print results
    for config, items in sorted(results.items()):
        print(f"\nConfiguration: {config}")
        for i, item in enumerate(items, 1):
            print(f"Sample {item['id']}:")
            print("Runtime:")
            print(f"  std: {item['runtime']['std']:.3f} ({'Success' if item['success']['std'] else 'Failure'})")
            print(f"  slimgb: {item['runtime']['slimgb']:.3f} ({'Success' if item['success']['slimgb'] else 'Failure'})")
            print(f"  stdfglm: {item['runtime']['stdfglm']:.3f} ({'Success' if item['success']['stdfglm'] else 'Failure'})")
            print(f"  transformer: {item['runtime']['transformer']:.3f} ({'Success' if item['success']['transformer'] else 'Failure'})")
            
            if print_fg:
                print("F:", item['F'])
                print("G:", item['G'])

if __name__ == "__main__":
    read_timing_results("/app/results/timing/shape")
