"""
Dataset Manager for curating training datasets from analyzer matches.

This module:
1. Manages dataset folders with configurable naming (train1, train2, etc.)
2. Saves matched spectrograms (1024 frequency bins) with labels and metadata
3. Provides utilities to load and merge datasets for training
4. Handles incremental dataset curation across multiple runs
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import shutil


class DatasetManager:
    """Manages dataset curation and organization"""
    
    def __init__(self, base_output_dir, dataset_base_name="train"):
        """
        Initialize dataset manager.
        
        Args:
            base_output_dir: Base output directory (e.g., simulator/outputs)
            dataset_base_name: Base name for dataset folders (e.g., "train" -> train1, train2, etc.)
        """
        self.base_output_dir = Path(base_output_dir)
        self.dataset_root = self.base_output_dir / 'dataset'
        self.dataset_root.mkdir(parents=True, exist_ok=True)
        self.dataset_base_name = dataset_base_name
        
        # Create config file if not exists
        self.config_path = self.dataset_root / 'dataset_config.json'
        if not self.config_path.exists():
            self._create_default_config()
    
    def _create_default_config(self):
        """Create default dataset configuration"""
        config = {
            'created_at': datetime.now().isoformat(),
            'dataset_base_name': self.dataset_base_name,
            'active_dataset': f'{self.dataset_base_name}1',
            'confidence_threshold': 0.05,  # Only save matches with confidence >= this
            'datasets': {}
        }
        self._save_config(config)
    
    def _load_config(self):
        """Load dataset configuration, merging with defaults for any missing keys"""
        defaults = {
            'created_at': datetime.now().isoformat(),
            'dataset_base_name': self.dataset_base_name,
            'active_dataset': f'{self.dataset_base_name}1',
            'unknown_dataset': None,
            'confidence_threshold': 0.05,  # Spatial angular match confidence (cos of angle error)
            'datasets': {}
        }
        with open(self.config_path, 'r') as f:
            on_disk = json.load(f)
        # Merge: defaults first, then on-disk values override
        merged = {**defaults, **on_disk}
        # If merged differs from on-disk (new keys added), persist the fix
        if merged != on_disk:
            self._save_config(merged)
        return merged
    
    def _save_config(self, config):
        """Save dataset configuration"""
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def get_active_dataset_name(self):
        """Get the currently active dataset name, auto-creating one if none is set"""
        config = self._load_config()
        name = config['active_dataset']
        if not name:
            # No active dataset — create the first one automatically
            name, _ = self.create_new_dataset()
        return name
    
    def set_active_dataset(self, dataset_name):
        """Set the active dataset for curation"""
        config = self._load_config()
        config['active_dataset'] = dataset_name
        
        # Create dataset folder if not exists
        dataset_path = self.dataset_root / dataset_name
        dataset_path.mkdir(exist_ok=True)
        
        # Initialize dataset metadata if new
        if dataset_name not in config['datasets']:
            config['datasets'][dataset_name] = {
                'created_at': datetime.now().isoformat(),
                'sample_count': 0,
                'runs_processed': []
            }
        
        self._save_config(config)
        return dataset_path
    
    def get_dataset_path(self, dataset_name=None):
        """Get path to a dataset folder"""
        if dataset_name is None:
            dataset_name = self.get_active_dataset_name()
        return self.dataset_root / dataset_name
    
    def list_datasets(self):
        """List all available datasets"""
        config = self._load_config()
        return list(config['datasets'].keys())
    
    def create_new_dataset(self, dataset_name=None):
        """Create a new dataset folder with incremental naming"""
        config = self._load_config()
        
        if dataset_name is None:
            # Find next available number
            existing = [name for name in config['datasets'].keys() 
                       if name.startswith(self.dataset_base_name)]
            
            if existing:
                # Extract numbers and find max
                numbers = []
                for name in existing:
                    try:
                        num = int(name.replace(self.dataset_base_name, ''))
                        numbers.append(num)
                    except ValueError:
                        continue
                next_num = max(numbers) + 1 if numbers else 1
            else:
                next_num = 1
            
            dataset_name = f'{self.dataset_base_name}{next_num}'
        
        # Create and set as active
        dataset_path = self.set_active_dataset(dataset_name)
        return dataset_name, dataset_path
    
    def save_matches_to_dataset(self, matches, run_id, confidence_threshold=None):
        """
        Save matched detections to the active dataset.
        
        Args:
            matches: List of match dictionaries from analyzer
            run_id: Unique identifier for this run
            confidence_threshold: Minimum confidence to save (uses config default if None)
        
        Returns:
            dict: Statistics about saved samples
        """
        config = self._load_config()
        active_dataset = config['active_dataset']
        dataset_path = self.get_dataset_path(active_dataset)
        
        # Ensure dataset directory exists
        dataset_path.mkdir(parents=True, exist_ok=True)
        
        if confidence_threshold is None:
            confidence_threshold = config['confidence_threshold']
        
        # Filter matches - only save high confidence matches to known sources
        high_confidence_matches = [
            m for m in matches 
            if m.get('label', m.get('source_label', 'unknown')) != 'unknown' 
            and m.get('confidence', 0) >= confidence_threshold
        ]
        
        if not high_confidence_matches:
            return {
                'saved': 0,
                'skipped_low_confidence': len([m for m in matches if m.get('label', m.get('source_label', 'unknown')) != 'unknown']),
                'skipped_unknown': len([m for m in matches if m.get('label', m.get('source_label', 'unknown')) == 'unknown']),
                'dataset': active_dataset
            }
        
        # Prepare data
        rows = []
        bin_count = None  # Auto-detect bin count from first match
        
        for match in high_confidence_matches:
            # Handle both full match objects and simplified saved matches
            if 'detection' in match:
                bins = match['detection'].get('bins', [])
            else:
                # This is a loaded match from JSON without bins
                continue
            
            if len(bins) == 0:
                continue
            
            # Auto-detect bin count from first valid match
            if bin_count is None:
                bin_count = len(bins)
                print(f"INFO: Detected {bin_count} frequency bins from ODAS output")
            
            if len(bins) == bin_count:  # Ensure consistent bin count
                row = {
                    'run_id': run_id,
                    'timestamp': match.get('timestamp', match['detection'].get('timestamp', 0)),
                    'label': match.get('label', match.get('source_label', 'unknown')),
                    'confidence': match.get('confidence', 0),
                    'angular_error': match.get('angular_error', -1),
                    'activity': match['detection'].get('activity', 0),
                    'position_x': match['detection']['x'],
                    'position_y': match['detection']['y'],
                    'position_z': match['detection']['z']
                }
                
                # Add frequency bins
                for i, bin_val in enumerate(bins):
                    row[f'bin_{i}'] = bin_val
                
                rows.append(row)
            else:
                print(f"WARNING: Skipping match with {len(bins)} bins (expected {bin_count})")
        
        # Create DataFrame
        df = pd.DataFrame(rows)
        
        # Save to CSV with timestamp
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_filename = f'{run_id}_{timestamp_str}.csv'
        csv_path = dataset_path / csv_filename
        
        df.to_csv(csv_path, index=False)
        
        # Update config - ensure dataset exists in config
        if active_dataset not in config['datasets']:
            config['datasets'][active_dataset] = {
                'created_at': datetime.now().isoformat(),
                'sample_count': 0,
                'runs_processed': []
            }
        
        config['datasets'][active_dataset]['sample_count'] += len(df)
        config['datasets'][active_dataset]['runs_processed'].append({
            'run_id': run_id,
            'timestamp': datetime.now().isoformat(),
            'samples_added': len(df),
            'file': csv_filename
        })
        config['datasets'][active_dataset]['last_updated'] = datetime.now().isoformat()
        self._save_config(config)
        
        return {
            'saved': len(df),
            'skipped_low_confidence': len([m for m in matches 
                                          if m['label'] != 'unknown' 
                                          and m['confidence'] < confidence_threshold]),
            'skipped_unknown': len([m for m in matches if m['label'] == 'unknown']),
            'file': str(csv_path),
            'dataset': active_dataset
        }
    
    def load_dataset(self, dataset_name=None, include_metadata=False):
        """
        Load all samples from a dataset.
        
        Args:
            dataset_name: Name of dataset to load (uses active if None)
            include_metadata: If True, return metadata columns; if False, only bins and label
        
        Returns:
            pd.DataFrame: Loaded dataset
        """
        if dataset_name is None:
            dataset_name = self.get_active_dataset_name()
        
        dataset_path = self.get_dataset_path(dataset_name)
        csv_files = list(dataset_path.glob('*.csv'))
        
        if not csv_files:
            return pd.DataFrame()
        
        # Load and concatenate all CSV files
        dfs = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            dfs.append(df)
        
        combined_df = pd.concat(dfs, ignore_index=True)
        
        if not include_metadata:
            # Return only bins and label for training
            bin_cols = [col for col in combined_df.columns if col.startswith('bin_')]
            bin_cols = sorted(bin_cols, key=lambda x: int(x.split('_')[1]))
            return combined_df[bin_cols + ['label']]
        
        return combined_df
    
    def get_dataset_stats(self, dataset_name=None):
        """Get statistics about a dataset"""
        config = self._load_config()
        
        if dataset_name is None:
            dataset_name = self.get_active_dataset_name()
        
        if dataset_name not in config['datasets']:
            return None
        
        dataset_info = config['datasets'][dataset_name]
        dataset_path = self.get_dataset_path(dataset_name)
        
        # Load dataset to get label distribution
        df = self.load_dataset(dataset_name, include_metadata=True)
        
        if df.empty:
            label_dist = {}
        else:
            label_dist = df['label'].value_counts().to_dict()
        
        return {
            'name': dataset_name,
            'path': str(dataset_path),
            'total_samples': len(df),
            'label_distribution': label_dist,
            'runs_processed': len(dataset_info.get('runs_processed', [])),
            'created_at': dataset_info.get('created_at'),
            'last_updated': dataset_info.get('last_updated')
        }
    
    def merge_datasets(self, dataset_names, output_name=None):
        """
        Merge multiple datasets into a new one.
        
        Args:
            dataset_names: List of dataset names to merge
            output_name: Name for merged dataset (auto-generated if None)
        
        Returns:
            str: Name of merged dataset
        """
        if output_name is None:
            output_name = f'{self.dataset_base_name}_merged_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        
        # Load all datasets
        dfs = []
        for name in dataset_names:
            df = self.load_dataset(name, include_metadata=True)
            if not df.empty:
                dfs.append(df)
        
        if not dfs:
            raise ValueError("No data to merge")
        
        # Combine
        merged_df = pd.concat(dfs, ignore_index=True)
        
        # Create new dataset
        dataset_path = self.set_active_dataset(output_name)
        
        # Save merged data
        csv_path = dataset_path / f'{output_name}.csv'
        merged_df.to_csv(csv_path, index=False)
        
        # Update config
        config = self._load_config()
        config['datasets'][output_name]['sample_count'] = len(merged_df)
        config['datasets'][output_name]['merged_from'] = dataset_names
        self._save_config(config)
        
        return output_name
    
    def delete_dataset(self, dataset_name):
        """Delete a dataset folder and its configuration"""
        config = self._load_config()
        
        if dataset_name not in config['datasets']:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        # Remove folder
        dataset_path = self.get_dataset_path(dataset_name)
        if dataset_path.exists():
            shutil.rmtree(dataset_path)
        
        # Remove from config
        del config['datasets'][dataset_name]
        
        # If this was active, set a new active dataset
        if config['active_dataset'] == dataset_name:
            remaining = list(config['datasets'].keys())
            if remaining:
                config['active_dataset'] = remaining[0]
            else:
                # Create a new default dataset
                config['active_dataset'] = f'{self.dataset_base_name}1'
        
        self._save_config(config)
