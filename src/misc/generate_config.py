import yaml
import itertools
import os 

def generate_yaml(num_var, field, gb_type):
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
        'density': 1.0,
        'degree_sampling': 'uniform',
        'term_sampling': 'uniform',
        'gb_type': gb_type
    }

    if field.startswith('GF'):
        # Remove unnecessary fields for GF
        for key in ['max_coeff_F', 'max_coeff_G', 'num_bound_F', 'num_bound_G']:
            data.pop(key, None)
    else:
        # Add fields for QQ and RR
        data.update({
            'coeff_bound': 100,  # bound of coefficients of polynomials (F and G)
            'max_coeff_F': 5,    # max_coeff for random polyomials used for generating F 
            'max_coeff_G': 5,    # max_coeff for random polyomials used for generating G
            'num_bound_F': 5,
            'num_bound_G': 5
        })

    return data

def save_yaml(data, filename):
    with open(filename, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

def main():
    save_path = 'config'
    os.makedirs(save_path, exist_ok=True)
    
    # Generate all combinations
    num_vars = range(2, 6)
    fields = ['GF7', 'GF31', 'QQ', 'RR']
    gb_types = ['shape', 'cauchy']

    for num_var, field, gb_type in itertools.product(num_vars, fields, gb_types):
        yaml_data = generate_yaml(num_var, field, gb_type)
        filename = f"{gb_type}_n={num_var}_field={field}_init.yaml"
        save_yaml(yaml_data, os.path.join(save_path, filename))
        print(f"Generated: {filename}")

    print("All YAML files have been generated.")

if __name__ == '__main__':
    main()