from typing import Dict, Any
from utils.reader import ResultsReader
from utils.generators import (
    generate_generation_latex,
    create_generation_table, 
    generate_timing_latex,
    create_timing_table, 
    generate_supremacy_latex,
    create_supremacy_table, 
    display_table, 
    get_config,
)

def process_generation_results(results: Dict[str, Dict[str, Any]]):
    """Process and display generation experiment results"""
    print("\nProcessing Generation Results...")
    
    # Display normal tables
    tasks = ['shape', 'cauchy']
    encoding_methods = ['standard_bart', 'hybrid_bart+']
    
    for task in tasks:
        for encoding_method in encoding_methods:
            config = get_config(task, encoding_method)
            table = create_generation_table(results, task, encoding_method, config)
            display_table(table, f"Generation: {task.upper()} - {encoding_method}")
    
    # Generate and display LaTeX table
    print("\nGeneration LaTeX Table:")
    print("-" * 80)
    print(generate_generation_latex(results))
    print("-" * 80)
    
def process_timing_results(results: Dict[str, Dict[str, Any]]):
    """Process and display timing experiment results"""
    print("\nProcessing Timing Results...")
    
    fields = ['GF7', 'GF31', 'QQ']  # RR is skipped
    conditions = [
        ('without density (d=1.0)', False),
        ('with density', True)
    ]
    
    for field in fields:
        for desc, with_density in conditions:
            table = create_timing_table(results, field, with_density)
            if table is not None:  # Only display if table was created
                display_table(table, f"Timing: {field} {desc}")
                latex_code = generate_timing_latex(results, field, with_density)
                if latex_code is not None:
                    print(f"\nTiming LaTeX Table for {field} {desc}:")
                    print(latex_code)

def process_supremacy_results(results: Dict[str, Dict[str, Any]]):
    """Process and display supremacy experiment results"""
    print("\nProcessing Timing Results...")
    
    fields = ['GF7', 'GF31', 'QQ']  # RR is skipped
    conditions = [
        ('without density (d=1.0)', False),
        ('with density', True)
    ]
    
    for field in fields:
        for desc, with_density in conditions:
            table = create_supremacy_table(results, field, with_density)
            if table is not None:  # Only display if table was created
                display_table(table, f"Supremacy: {field} {desc}")
                latex_code = generate_supremacy_latex(results, field, with_density)
                if latex_code is not None:
                    print(f"\nTiming LaTeX Table for {field} {desc}:")
                    print(latex_code)
                    
def main():
    """Main function to run the results collection and analysis"""
    # Generation experiments
    # print("Processing Generation Experiments...")
    # gen_reader = ResultsReader(
    #     experiment_name='generation',
    #     yaml_filename='generation_results.yaml',
    #     use_encoding_method=True,
    #     use_density=True
    # )
    # gen_results = gen_reader.read_all_results(verbose=True)
    # process_generation_results(gen_results)
    
    # # Timing experiments
    # print("\nProcessing Timing Experiments...")
    # timing_reader = ResultsReader(
    #     experiment_name='timing',
    #     yaml_filename='timing.yaml',
    #     use_encoding_method=False,
    #     use_density=False
    # )
    # timing_reader_ = ResultsReader(
    #     experiment_name='timing',
    #     yaml_filename='timing.yaml',
    #     use_encoding_method=False,
    #     use_density=True
    # )
    # timing_results = timing_reader.read_all_results(verbose=True)
    # timing_results_ = timing_reader_.read_all_results(verbose=True)
    # timing_results.update(timing_results_)

    # process_timing_results(timing_results)
    
    
    # Timing experiments
    print("\nProcessing Transformer Supremacy Experiments...")
    supremacy_reader = ResultsReader(
        experiment_name='timing',
        yaml_filename='transformer_supermacy_timeout=100.0.yaml',
        use_encoding_method=False,
        use_density=True
    )
    
    supremacy_results = supremacy_reader.read_all_results(verbose=True)
    
    process_supremacy_results(supremacy_results)
    

if __name__ == '__main__':
    main()