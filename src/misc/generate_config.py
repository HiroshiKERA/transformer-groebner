import argparse
import itertools
import os 
import yaml

def generate_yaml(num_var, field, gb_type, density):
    data = {
        'num_var': num_var,
        'field': field,
        'num_samples_train': 1000000,
        'num_samples_test': 1000,
        'max_degree_F': 3,
        'max_degree_G': 5,
        'max_num_terms_F': 2,
        'max_num_terms_G': 5,
        'max_size_F': num_var + 2,
        'num_duplicants': 1,
        'density': density,  
        'degree_sampling': ('', 'uniform')[0],  # no control; just showing the setting
        'term_sampling': ('', 'uniform')[1],    # no control; just showing the setting
        'gb_type': gb_type
    }
    
    if field.startswith('GF'):
        # Remove unnecessary fields for GF
        for key in ['max_coeff_F', 'max_coeff_G', 'num_bound_F', 'num_bound_G']:
            data.pop(key, None)
    else:
        # Add fields for QQ and RR
        data.update({
            'coeff_bound': 100,
            'max_coeff_F': 5,
            'max_coeff_G': 5,
            'num_bound_F': 5,
            'num_bound_G': 5
        })
    return data

def save_yaml(data, filename):
    with open(filename, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

def main():
    parser = argparse.ArgumentParser(description='Generate YAML configuration files.')
    parser.add_argument('--save_path', default='config', help='Path to save YAML files')
    parser.add_argument('--densities', nargs='+', type=float, default=[1.0, 1.0, 1.0, 1.0],
                        help='Densities for num_vars 2, 3, 4, 5 respectively')
    args = parser.parse_args()

    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    
    # Generate all combinations
    num_vars = range(2, 6)
    fields = ['GF7', 'GF31', 'QQ', 'RR']
    gb_types = ['shape', 'cauchy']
    
    # densitiesのリストが不足している場合、最後の値で埋める
    densities = args.densities + [args.densities[-1]] * (len(num_vars) - len(args.densities))
    
    for (num_var, density), field, gb_type in itertools.product(zip(num_vars, densities), fields, gb_types):
        yaml_data = generate_yaml(num_var, field, gb_type, density)
        
        if density == 1.0:
            filename = f"{gb_type}_n={num_var}_field={field}.yaml"
        else:
            filename = f"{gb_type}_n={num_var}_field={field}_density={density}.yaml"
        save_yaml(yaml_data, os.path.join(save_path, filename))
        print(f"Generated: {filename} with density {density}")
    
    print("All YAML files have been generated.")

if __name__ == '__main__':
    main()