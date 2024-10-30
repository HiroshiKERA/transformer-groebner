from typing import Optional

def format_metrics(acc: Optional[float], support_acc: Optional[float]) -> str:
    """Format accuracy metrics with slash separator
    
    Args:
        acc: Accuracy value to format
        support_acc: Support accuracy value to format
        
    Returns:
        Formatted string in the format "acc% / support_acc%"
    """
    if acc is None or support_acc is None:
        return "N/A"
    return f"{acc*100:.1f}% / {support_acc*100:.1f}%"

def format_timing_value(runtime: float, method_name: str) -> str:
    """Format runtime with appropriate units and precision
    
    Args:
        runtime: Runtime value to format
        method_name: Name of the method (to handle special cases)
        
    Returns:
        Formatted string with appropriate precision and units
    """
    if method_name == 'B. (ours)':
        return ""  # Return empty string for Transformer results
    
    if runtime < 0.01:
        return f".{int(runtime*1000):03d}"
    elif runtime < 1:
        return f"{runtime:.3f}"
    elif runtime < 10:
        return f"{runtime:.2f}"
    else:
        return f"{runtime:.1f}"

def get_field_name(field: str) -> str:
    """Get LaTeX formatted field name
    
    Args:
        field: Field name ('GF7', 'GF31', 'QQ', or 'RR')
        
    Returns:
        LaTeX formatted field name
    """
    field_map = {
        'GF7': r'$\bF_{7}$',
        'GF31': r'$\bF_{31}$',
        'QQ': r'$\bQ$',
        'RR': r'$\bR$'
    }
    return field_map.get(field, field)