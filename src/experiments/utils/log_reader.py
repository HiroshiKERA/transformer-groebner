import re
from pathlib import Path
from typing import Optional

def read_runtime_from_log(field: str, n: int, with_density: bool = False, base_dir: str = 'data') -> Optional[float]:
    """Read runtime from log file
    
    Args:
        field: Field name (e.g., 'GF7')
        n: Dimension parameter
        with_density: Whether to use density in path
        base_dir: Base directory for log files
        
    Returns:
        Runtime value if found, None otherwise
    """
    # Construct log file path
    base_path = f"shape_n={n}_field={field}"
    if with_density and n > 2:
        density_map = {3: 0.6, 4: 0.3, 5: 0.2}
        base_path += f"_density={density_map[n]}"
    
    log_file = Path(base_dir) / "shape" / Path(base_path) / f"run_{base_path}.log"

    try:
        with open(log_file, 'r') as f:
            content = f.read()
            # Find the line containing "backward generation"
            match = re.search(r'backward generation \| (\d+\.\d+) \[sec\]', content)
            if match:
                return float(match.group(1))
    except (FileNotFoundError, IOError) as e:
        print(f"Warning: Could not read log file {log_file}: {e}")
    except Exception as e:
        print(f"Error processing log file {log_file}: {e}")
    
    return None

def format_transformer_runtime(runtime: Optional[float]) -> str:
    """Format transformer runtime with appropriate precision
    
    Args:
        runtime: Runtime value to format
        
    Returns:
        Formatted string
    """
    if runtime is None:
        return ""
    elif runtime < 0.01:
        return f".{int(runtime*1000):03d}"
    elif runtime < 1:
        return f"{runtime:.3f}"
    elif runtime < 10:
        return f"{runtime:.2f}"
    else:
        return f"{runtime:.1f}"