from pathlib import Path
import yaml
from typing import List, Dict, Any, Optional
from itertools import product
from utils.config import GenerationConfig, TimingConfig

class ResultsReader:
    """Class for reading experimental results from YAML files"""
    def __init__(
        self, 
        experiment_name: str, 
        yaml_filename: str = 'results.yaml',
        base_dir: str = 'results',
        use_encoding_method: bool = False,
        use_density: bool = False
    ):
        """Initialize ResultsReader
        
        Args:
            experiment_name: Name of the experiment (e.g., 'generation', 'timing')
            yaml_filename: Name of the YAML files to read
            base_dir: Base directory for results
            use_encoding_method: Whether to include encoding method in path
            use_density: Whether to include density in path
        """
        self.experiment_name = experiment_name
        self.base_dir = Path(base_dir)
        self.yaml_filename = yaml_filename
        self.use_encoding_method = use_encoding_method
        self.use_density = use_density
        
        # Set up configuration based on experiment type
        if experiment_name == 'generation':
            self.config = GenerationConfig(yaml_filename=yaml_filename)
        else:  # timing
            self.config = TimingConfig(yaml_filename=yaml_filename)
    
    def _construct_path(self, task: str, n: int, field: str, encoding_method: Optional[str] = None) -> Path:
        """Construct file path based on parameters"""
        components = [self.base_dir, self.experiment_name, task]
        
        if self.use_encoding_method and encoding_method:
            components.append(encoding_method)
        
        dir_parts = [f"{task}_n={n}_field={field}"]
        
        if self.use_density and self.config.density_map[n] is not None:
            dir_parts.append(f"density={self.config.density_map[n]}")
        
        dir_name = "_".join(dir_parts)
        components.append(dir_name)
        components.append(self.yaml_filename)
        
        return Path(*components)
    
    def _get_all_possible_paths(self) -> List[Path]:
        """Get all possible file paths based on configuration"""
        if isinstance(self.config, GenerationConfig):
            tasks = self.config.tasks
        else:  # TimingConfig
            tasks = [self.config.task]
            
        params = [tasks, self.config.ns, self.config.fields]
        
        if self.use_encoding_method and isinstance(self.config, GenerationConfig):
            params.append(self.config.encoding_methods)
            base_product = product(*params)
            return [self._construct_path(task, n, field, enc) 
                    for task, n, field, enc in base_product]
        
        base_product = product(*params)
        return [self._construct_path(task, n, field) 
                for task, n, field in base_product]

    def read_yaml_file(self, file_path: Path) -> Dict[str, Any]:
        """Read a single YAML file"""
        try:
            with open(file_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return {}

    def read_all_results(self, verbose: bool = False) -> Dict[str, Dict[str, Any]]:
        """Read all experiment results from YAML files"""
        all_results = {}
        all_paths = self._get_all_possible_paths()
        
        if verbose:
            print("\nChecking for YAML files...")
        
        for path in all_paths:
            relative_path = path.relative_to(self.base_dir / self.experiment_name)
            
            if path.exists():
                if verbose:
                    print(f"Found: {relative_path}")
                results = self.read_yaml_file(path)
                if results:
                    all_results[str(relative_path)] = results
            else:
                print(f"Missing file: {relative_path}")
        
        if verbose:
            print(f"\nSuccessfully loaded {len(all_results)} files")
        
        return all_results