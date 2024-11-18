import pandas as pd
from typing import Dict, Any, Optional
from utils.formatters import format_metrics, format_timing_value, get_field_name
from utils.config import TimingConfig, SupremacyConfig, SuccessRateConfig
from utils.log_reader import read_runtime_from_log, format_transformer_runtime

def print_available_files(results: Dict[str, Dict[str, Any]]):
    """Print all available files for debugging"""
    print("\nAll available files in timing_results:")
    for path in sorted(results.keys()):
        print(f"  {path}")

def get_closest_match(target_path: str, results: Dict[str, Dict[str, Any]]) -> Optional[str]:
    """Find the closest matching path in results
    
    Args:
        target_path: Target path we're looking for
        results: Dictionary of results
        
    Returns:
        Matching path if found, None otherwise
    """
    # Remove 'density' part from path for matching
    base_pattern = target_path.split('density=')[0].rstrip('_')
    matching_paths = [p for p in results.keys() if base_pattern in p]
    return matching_paths[0] if matching_paths else None

def should_skip_combination(task: str, field: str) -> bool:
    """Determine if this combination should be skipped
    
    Args:
        task: Task name
        field: Field name
    
    Returns:
        bool: True if combination should be skipped
    """
    return field == 'RR'  # Skip RR for timing experiments

def get_config(task: str, encoding_method: str) -> Dict[str, Any]:
    """Get configuration based on task and encoding method
    
    Args:
        task: Task name ('shape' or 'cauchy')
        encoding_method: Encoding method ('standard_bart' or 'hybrid_bart+')
    
    Returns:
        Dict containing configuration parameters
    """
    # Base fields
    fields = ['GF7', 'GF31', 'QQ']
    
    # Add RR only for hybrid_bart+
    if encoding_method == 'hybrid_bart+':
        fields.append('RR')
    
    # n values based on task
    if task == 'cauchy':
        n_values = [2, 3]
    else:  # shape
        n_values = [2, 3, 4, 5]
    
    return {
        'fields': fields,
        'n_values': n_values
    }
    
def display_table(df: pd.DataFrame, title: str):
    """Display a table with proper formatting and title
    
    Args:
        df: DataFrame to display
        title: Title for the table
    """
    max_width = max(60, len(title) + 10)
    print(f"\n{'-'*max_width}")
    print(f"Results for {title}")
    print(f"{'-'*max_width}")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_rows', None)
    print(df)
    print(f"{'-'*max_width}")

def create_generation_table(
    results: Dict[str, Dict[str, Any]], 
    task: str, 
    encoding_method: str,
    config: Dict[str, Any]
) -> pd.DataFrame:
    """Create summary table for generation experiments"""
    fields = config['fields']
    n_values = config['n_values']
    densities = {3: 0.6, 4: 0.3, 5: 0.2}
    
    # Create column names
    columns = []
    for n in n_values:
        if n == 2:
            columns.append(f'n={n}')
        else:
            density = densities[n]
            columns.append(f'n={n}\n(d={density})')
    
    # Create DataFrame with larger row heights for multi-line entries
    df = pd.DataFrame(index=fields, columns=columns)
    
    # Fill the table with results
    for field in fields:
        for n in n_values:
            # Construct the expected path
            if n == 2:
                path = f"{task}/{encoding_method}/{task}_n={n}_field={field}/generation_results.yaml"
            else:
                density = densities[n]
                path = f"{task}/{encoding_method}/{task}_n={n}_field={field}_density={density}/generation_results.yaml"
            
            # Get both accuracy values if path exists
            if path in results:
                result = results[path]
                acc = result.get('acc')
                support_acc = result.get('support_acc')
                
                # Format both metrics
                df.loc[field, columns[n-2]] = format_metrics(acc, support_acc)
            else:
                df.loc[field, columns[n-2]] = "N/A"
    
    return df

def generate_generation_latex(results: Dict[str, Dict[str, Any]]) -> str:
    """Generate LaTeX table for generation experiments
    
    Args:
        results: Dictionary of results
        
    Returns:
        LaTeX table code as string
    """
    latex_template = r"""\begin{tabularx}{\linewidth}{ll*{4}{YY}YY}
    \toprule
    \multicolumn{2}{c}{\multirow{2}{*}{Coeff.}} & \multicolumn{4}{c|}{Shape position} & \multicolumn{2}{c}{Cauchy module} \\
    \cmidrule{3-6}\cmidrule{7-8}
    & & $n=2$ & $n=3$ & $n=4$ & $n=5$ & $n=2$ & $n=3$ \\
    \Xhline{2\arrayrulewidth}"""
    
    # Field order and their LaTeX representations
    fields = [
        ('QQ', r'$\bQ$'),
        ('GF7', r'$\bF_{7}$'),
        ('GF31', r'$\bF_{31}$'),
        ('RR', r'$\bR$')
    ]
    
    # Methods and their LaTeX representations
    methods = [
        ('standard_bart', r'\textit{disc.}'),
        ('hybrid_bart+', r'\textit{hyb.}')
    ]

    latex_rows = []
    for field, field_tex in fields:
        if field == 'RR':
            # Special case for RR: only hybrid_bart+
            row_values = []
            # Shape values (n=2-5)
            for n in [2, 3, 4, 5]:
                if n == 2:
                    path = f"shape/hybrid_bart+/shape_n={n}_field={field}/generation_results.yaml"
                else:
                    density = {3: 0.6, 4: 0.3, 5: 0.2}[n]
                    path = f"shape/hybrid_bart+/shape_n={n}_field={field}_density={density}/generation_results.yaml"
                if path in results:
                    result = results[path]
                    acc = result.get('acc', 0) * 100
                    support_acc = result.get('support_acc', 0) * 100
                    row_values.append(f"{acc:4.1f} / {support_acc:4.1f}")
                else:
                    row_values.append("--")
            
            # Cauchy values (n=2-3)
            for n in [2, 3]:
                if n == 2:
                    path = f"cauchy/hybrid_bart+/cauchy_n={n}_field={field}/generation_results.yaml"
                else:
                    path = f"cauchy/hybrid_bart+/cauchy_n={n}_field={field}_density=0.6/generation_results.yaml"
                if path in results:
                    result = results[path]
                    acc = result.get('acc', 0) * 100
                    support_acc = result.get('support_acc', 0) * 100
                    row_values.append(f"{acc:4.1f} / {support_acc:4.1f}")
                else:
                    row_values.append("--")
                    
            latex_rows.append(
                f"    {field_tex} & \\textit{{hyb.}} & {' & '.join(row_values)} \\\\"
            )
            
        else:
            # Other fields: both methods
            for i, (method, method_tex) in enumerate(methods):
                row_values = []
                # Shape values (n=2-5)
                for n in [2, 3, 4, 5]:
                    if n == 2:
                        path = f"shape/{method}/shape_n={n}_field={field}/generation_results.yaml"
                    else:
                        density = {3: 0.6, 4: 0.3, 5: 0.2}[n]
                        path = f"shape/{method}/shape_n={n}_field={field}_density={density}/generation_results.yaml"
                    if path in results:
                        result = results[path]
                        acc = result.get('acc', 0) * 100
                        support_acc = result.get('support_acc', 0) * 100
                        row_values.append(f"{acc:4.1f} / {support_acc:4.1f}")
                    else:
                        row_values.append("--")
                
                # Cauchy values (n=2-3)
                for n in [2, 3]:
                    if n == 2:
                        path = f"cauchy/{method}/cauchy_n={n}_field={field}/generation_results.yaml"
                    else:
                        path = f"cauchy/{method}/cauchy_n={n}_field={field}_density=0.6/generation_results.yaml"
                    if path in results:
                        result = results[path]
                        acc = result.get('acc', 0) * 100
                        support_acc = result.get('support_acc', 0) * 100
                        row_values.append(f"{acc:4.1f} / {support_acc:4.1f}")
                    else:
                        row_values.append("--")
                
                # First row of field gets \multirow
                if i == 0:
                    latex_rows.append(
                        f"    \\multirow{{2}}{{*}}{{{field_tex}}} & {method_tex} & {' & '.join(row_values)} \\\\"
                    )
                else:
                    latex_rows.append(
                        f"    & {method_tex} & {' & '.join(row_values)} \\\\"
                    )
            latex_rows.append("    \\cline{1-8}")
    
    latex_end = r"    \bottomrule \end{tabularx}"
    
    return "\n".join([latex_template] + latex_rows + [latex_end])

def create_timing_table(
    results: Dict[str, Dict[str, Any]], 
    field: str, 
    with_density: bool = False,
    metric='runtime'
) -> Optional[pd.DataFrame]:
    """Create timing summary table"""
    if should_skip_combination('shape', field):
        return None
        
    if metric == 'runtime':
        config = TimingConfig()
    elif metric == 'success_rate':
        config = SuccessRateConfig()
    else:
        raise ValueError(f"Invalid metric: {metric}")
    
    # Create column names with density information
    columns = []
    for n in config.ns:
        if not with_density:
            columns.append(f'n={n}')
        else:
            if n == 2:
                columns.append(f'n={n}\n(d=1.0)')
            else:
                density = config.density_map[n]
                columns.append(f'n={n}\n(d={density})')
    
    # Create DataFrame
    df = pd.DataFrame(
        index=config.method_names,
        columns=columns
    )
    
    # Fill table with results
    for n in config.ns:
        # Base path pattern
        base_path = f"shape/shape_n={n}_field={field}"
        
        # Find the matching path in results
        matching_path = None
        if with_density and n > 2:
            density = config.density_map[n]
            full_path = f"{base_path}_density={density}/timing.yaml"
            
            if full_path in results:
                matching_path = full_path
        else:
            full_path = f"{base_path}/timing.yaml"
            
            if full_path in results:
                matching_path = full_path 
                
                   
        if matching_path:
            for method_name, method_key in zip(config.method_names, config.methods):
                if method_name == 'B. (ours)':
                    # Read runtime from log file
                    runtime = read_runtime_from_log(field, n, with_density)
                    df.loc[method_name, columns[n-2]] = format_transformer_runtime(runtime)
                    continue
                    
                value = results[matching_path].get(method_key)
                if value is not None:
                    df.loc[method_name, columns[n-2]] = format_timing_value(value, method_name)
                else:
                    df.loc[method_name, columns[n-2]] = 'N/A'
        else:
            print(f"Warning: No matching path found for {base_path}")
            print(f"Available paths: {[p for p in results.keys() if str(n) in p and field in p]}")
            for method_name in config.method_names:
                df.loc[method_name, columns[n-2]] = 'N/A'
    
    return df

def generate_timing_latex(results: Dict[str, Dict[str, Any]], field: str, with_density: bool = False, metric: str = 'runtime') -> Optional[str]:
    """Generate LaTeX table for timing experiments"""
    if should_skip_combination('shape', field):
        return None
        
    if metric == 'runtime':
        config = TimingConfig()
    elif metric == 'success_rate':
        config = SuccessRateConfig()
    else:
        raise ValueError(f"Invalid metric: {metric}")
    
    latex_template = r"""\begin{tabularx}{\linewidth}{l*{3}{Y}*{1}{Y}}
    \toprule
    Method & $n=2$ & $n=3$ & $n=4$ & $n=5$\\"""
    
    # Add density information if needed
    if with_density:
        latex_template += r"""\\
    & (d=1.0) & (d=0.6) & (d=0.3) & (d=0.2)"""
    
    latex_template += r"""
    \Xhline{2\arrayrulewidth}"""
    
    latex_rows = []
    for method_name, method_key in zip(
        ['F. (\\textsc{std})', 'F. (\\textsc{slimgb})', 'F. (\\textsc{stdfglm})', 'B.~(ours)'],
        config.methods
    ):
        values = []
        for n in config.ns:
            if method_name == 'B.~(ours)' and metric == 'runtime':
                # Read runtime from log file
                runtime = read_runtime_from_log(field, n, with_density)
                values.append(format_transformer_runtime(runtime))
                continue
                
            # Construct path based on experiment type
            if with_density and n > 2:
                path = f"shape/shape_n={n}_field={field}_density={config.density_map[n]}/timing.yaml"
            else:
                path = f"shape/shape_n={n}_field={field}/timing.yaml"
            
            if path in results:
                value = results[path].get(method_key)
                if value is not None:
                    if metric == 'success_rate':
                        values.append(f"{value*100:.1f}")
                    else:
                        values.append(format_timing_value(value, method_name))
                else:
                    values.append("--")
            else:
                values.append("--")
        
        if method_name == 'B.~(ours)' and metric == 'runtime':
            latex_rows.append("    \\hline")
            row = f"    {method_name} & " + " & ".join(values) + " \\\\"
        else:
            row = f"    {method_name} & " + " & ".join(values) + " \\\\"
        latex_rows.append(row)
    
    latex_end = r"    \bottomrule \end{tabularx}"
    
    return "\n".join([latex_template] + latex_rows + [latex_end])


def create_supremacy_table(
    results: Dict[str, Dict[str, Any]], 
    field: str, 
    with_density: bool = False
) -> Optional[pd.DataFrame]:
    """Create supremacy summary table"""
    if should_skip_combination('shape', field):
        return None
        
    config = SupremacyConfig()
    
    # Create column names with density information
    columns = []
    for n in config.ns:
        if not with_density:
            columns.append(f'n={n}')
        else:
            if n == 2:
                columns.append(f'n={n}\n(d=1.0)')
            else:
                density = config.density_map[n]
                columns.append(f'n={n}\n(d={density})')
    
    # Create DataFrame
    df = pd.DataFrame(
        index=config.method_names,
        columns=columns
    )
    
    # Fill table with results
    for n in config.ns:
        # Base path pattern
        base_path = f"shape/shape_n={n}_field={field}"
        
        # Find the matching path in results
        matching_path = None
        if with_density and n > 2:
            density = config.density_map[n]
            full_path = f"{base_path}_density={density}/transformer_supermacy_timeout=100.0.yaml"
            
            if full_path in results:
                matching_path = full_path
        else:
            full_path = f"{base_path}/transformer_supermacy_timeout=100.0.yaml"
            
            if full_path in results:
                matching_path = full_path 
                
                   
        if matching_path:
            for method_name, method_key in zip(config.method_names, config.methods):
                if method_name == 'B. (ours)':
                    # Read runtime from log file
                    runtime = read_runtime_from_log(field, n, with_density)
                    df.loc[method_name, columns[n-2]] = format_transformer_runtime(runtime)
                    continue
                    
                value = results[matching_path].get(method_key)
                if value is not None:
                    df.loc[method_name, columns[n-2]] = format_timing_value(value, method_name)
                else:
                    df.loc[method_name, columns[n-2]] = 'N/A'
        else:
            print(f"Warning: No matching path found for {base_path}")
            print(f"Available paths: {[p for p in results.keys() if str(n) in p and field in p]}")
            for method_name in config.method_names:
                df.loc[method_name, columns[n-2]] = 'N/A'
    
    return df
