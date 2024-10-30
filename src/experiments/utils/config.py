from dataclasses import dataclass
from typing import Dict, Tuple, Optional

@dataclass
class BaseConfig:
    """Base configuration class for experiment parameters"""
    ns: Tuple[int, ...] = (2, 3, 4, 5)
    fields: Tuple[str, ...] = ('GF7', 'GF31', 'QQ', 'RR')
    density_map: Dict[int, Optional[float]] = None
    
    def __post_init__(self):
        self.density_map = {
            2: None,
            3: 0.6,
            4: 0.3,
            5: 0.2
        }

@dataclass
class GenerationConfig(BaseConfig):
    """Configuration for generation experiments"""
    tasks: Tuple[str, ...] = ('shape', 'cauchy')
    encoding_methods: Tuple[str, ...] = ('standard_bart', 'hybrid_bart+')
    yaml_filename: str = 'generation_results.yaml'

@dataclass
class TimingConfig(BaseConfig):
    """Configuration for timing experiments"""
    task: str = 'shape'
    yaml_filename: str = 'timing.yaml'
    methods: Tuple[str, ...] = (
        'libsingular:std_runtime',
        'libsingular:slimgb_runtime',
        'libsingular:stdfglm_runtime',
        'transformer_runtime'
    )
    method_names: Tuple[str, ...] = (
        'F. (std)',
        'F. (slimgb)',
        'F. (stdfglm)',
        'B. (ours)'
    )
    
    
@dataclass
class SupremacyConfig(BaseConfig):
    """Configuration for supremacy experiments"""
    task: str = 'shape'
    yaml_filename: str = 'supremacy.yaml'
    methods: Tuple[str, ...] = (
        'libsingular:std_runtime',
        'libsingular:slimgb_runtime',
        'libsingular:stdfglm_runtime',
        'transformer_runtime'
    )
    method_names: Tuple[str, ...] = (
        'F. (std)',
        'F. (slimgb)',
        'F. (stdfglm)',
        'B. (ours)'
    )