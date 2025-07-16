#!/usr/bin/env python3
"""
MERGE Dataset Loader
===================

Provides easy-to-use functions for loading the MERGE dataset in various configurations.

Author: Generated for Big Data Processing Project
Date: 2025-07-15
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import yaml
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MERGEDatasetLoader:
    """Loader for the MERGE dataset with flexible filtering options."""
    
    def __init__(self, dataset_path: str):
        """
        Initialize the loader.
        
        Args:
            dataset_path: Path to the merge-dataset directory
        """
        self.dataset_path = Path(dataset_path)
        self.metadata_path = self.dataset_path / "metadata" / "base_metadata.csv"
        self.data_path = self.dataset_path / "data"
        
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        
        self._metadata = None
        self._schema = None
    
    @property
    def metadata(self) -> pd.DataFrame:
        """Lazy load the metadata."""
        if self._metadata is None:
            self._metadata = pd.read_csv(self.metadata_path)
            logger.info(f"Loaded metadata with {len(self._metadata)} records")
        return self._metadata
    
    @property
    def schema(self) -> Dict:
        """Lazy load the schema."""
        if self._schema is None:
            schema_path = self.dataset_path / "schema" / "metadata_schema.yml"
            if schema_path.exists():
                with open(schema_path, 'r') as f:
                    self._schema = yaml.safe_load(f)
            else:
                self._schema = {}
        return self._schema
    
    def get_available_versions(self) -> List[str]:
        """Get list of available dataset versions."""
        versions_dir = self.dataset_path / "metadata" / "versions"
        if versions_dir.exists():
            return [d.name for d in versions_dir.iterdir() if d.is_dir()]
        return []
    
    def get_summary_stats(self, version: str = "v1.1") -> Dict:
        """Get summary statistics for a specific version."""
        summary_path = self.dataset_path / "metadata" / "versions" / version / "summary.yml"
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def load_merge_dataset(
        self,
        version: str = 'v1.1',
        split: Optional[str] = None,
        strategy: str = '70_15_15',
        balanced: Optional[bool] = None,
        mode: str = 'bimodal',
        return_format: str = 'pandas',
        include_paths: bool = True,
        validate_files: bool = False,
        keep_availability_columns: bool = False
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict]]:
        """
        Load the MERGE dataset with specified filters.
        
        Args:
            version: Dataset version (default: 'v1.1')
            split: Train/validation/test split ('train', 'validate', 'test', 'all', or None)
            strategy: Split strategy ('70_15_15' or '40_30_30')
            balanced: Use balanced subset (True/False/None for both)
            mode: Modality ('audio', 'lyrics', 'bimodal', or 'all')
            return_format: Return format ('pandas' or 'dict')
            include_paths: Include file paths in the result
            validate_files: Validate that files exist on disk
            keep_availability_columns: Keep availability and split columns for analysis
            
        Returns:
            DataFrame or tuple of (DataFrame, metadata_dict)
        """
        df = self.metadata.copy()
        
        # Apply version filter
        if 'version' in df.columns:
            df = df[df['version'] == version]
        
        # Apply mode filter
        if mode != 'all':
            availability_cols = self._get_availability_columns(mode, balanced)
            if availability_cols:
                # Keep rows where at least one of the required availability columns is True
                mask = df[availability_cols].any(axis=1)
                df = df[mask]
            else:
                logger.warning(f"No availability columns found for mode={mode}, balanced={balanced}")
        
        # Apply split filter
        if split and split != 'all':
            split_cols = self._get_split_columns(strategy, mode, balanced)
            if split_cols:
                # Keep rows where at least one split column matches the requested split
                mask = df[split_cols].eq(split).any(axis=1)
                df = df[mask]
                
                # Add a unified split column
                df['split'] = split
            else:
                logger.warning(f"No split columns found for strategy={strategy}, mode={mode}, balanced={balanced}")
        elif split == 'all':
            # Include split information but don't filter
            split_cols = self._get_split_columns(strategy, mode, balanced)
            if split_cols:
                df['split'] = df[split_cols].bfill(axis=1).iloc[:, 0]
        
        # Add file paths if requested
        if include_paths:
            df = self._add_file_paths(df, mode, balanced)
        
        # Validate files if requested
        if validate_files and include_paths:
            df = self._validate_file_paths(df)
        
        # Clean up columns
        df = self._cleanup_dataframe(df, mode, balanced, include_paths, keep_availability_columns)
        
        # Prepare metadata
        metadata = {
            'version': version,
            'split': split,
            'strategy': strategy,
            'balanced': balanced,
            'mode': mode,
            'total_samples': len(df),
            'quadrant_distribution': df['quadrant'].value_counts().to_dict() if 'quadrant' in df.columns else {},
            'split_distribution': df['split'].value_counts().to_dict() if 'split' in df.columns else {}
        }
        
        if return_format == 'pandas':
            return df
        else:
            return df, metadata
    
    def _get_availability_columns(self, mode: str, balanced: Optional[bool]) -> List[str]:
        """Get the availability columns for the specified mode and balanced setting."""
        cols = []
        
        if balanced is None:
            # Include both balanced and complete
            if mode in ['audio', 'all']:
                cols.extend(['available_audio_balanced', 'available_audio_complete'])
            if mode in ['lyrics', 'all']:
                cols.extend(['available_lyrics_balanced', 'available_lyrics_complete'])
            if mode in ['bimodal', 'all']:
                cols.extend(['available_bimodal_balanced', 'available_bimodal_complete'])
        else:
            suffix = 'balanced' if balanced else 'complete'
            if mode in ['audio', 'all']:
                cols.append(f'available_audio_{suffix}')
            if mode in ['lyrics', 'all']:
                cols.append(f'available_lyrics_{suffix}')
            if mode in ['bimodal', 'all']:
                cols.append(f'available_bimodal_{suffix}')
        
        # Filter to columns that actually exist
        existing_cols = [col for col in cols if col in self.metadata.columns]
        return existing_cols
    
    def _get_split_columns(self, strategy: str, mode: str, balanced: Optional[bool]) -> List[str]:
        """Get the split columns for the specified parameters."""
        cols = []
        
        modes = [mode] if mode != 'all' else ['audio', 'lyrics', 'bimodal']
        
        for m in modes:
            if balanced is None:
                # Include both balanced and complete
                cols.extend([
                    f'split_{strategy}_balanced_{m}',
                    f'split_{strategy}_complete_{m}'
                ])
            else:
                suffix = 'balanced' if balanced else 'complete'
                cols.append(f'split_{strategy}_{suffix}_{m}')
        
        # Filter to columns that actually exist
        existing_cols = [col for col in cols if col in self.metadata.columns]
        return existing_cols
    
    def _add_file_paths(self, df: pd.DataFrame, mode: str, balanced: Optional[bool]) -> pd.DataFrame:
        """Add unified file path columns."""
        df = df.copy()
        
        # Initialize path columns
        if mode in ['audio', 'bimodal', 'all']:
            df['audio_path'] = None
        if mode in ['lyrics', 'bimodal', 'all']:
            df['lyrics_path'] = None
        
        # Get the appropriate path columns
        if balanced is None:
            # Prefer balanced, fallback to complete
            audio_cols = ['audio_path_balanced', 'audio_path_complete']
            lyrics_cols = ['lyrics_path_balanced', 'lyrics_path_complete']
        else:
            suffix = 'balanced' if balanced else 'complete'
            audio_cols = [f'audio_path_{suffix}']
            lyrics_cols = [f'lyrics_path_{suffix}']
        
        # Assign audio paths
        if mode in ['audio', 'bimodal', 'all']:
            for col in audio_cols:
                if col in df.columns:
                    mask = df['audio_path'].isna() & df[col].notna()
                    df.loc[mask, 'audio_path'] = df.loc[mask, col]
        
        # Assign lyrics paths
        if mode in ['lyrics', 'bimodal', 'all']:
            for col in lyrics_cols:
                if col in df.columns:
                    mask = df['lyrics_path'].isna() & df[col].notna()
                    df.loc[mask, 'lyrics_path'] = df.loc[mask, col]
        
        return df
    
    def _validate_file_paths(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate that file paths exist on disk."""
        df = df.copy()
        
        if 'audio_path' in df.columns:
            df['audio_file_exists'] = df['audio_path'].apply(
                lambda x: (self.data_path / x).exists() if pd.notna(x) else False
            )
        
        if 'lyrics_path' in df.columns:
            df['lyrics_file_exists'] = df['lyrics_path'].apply(
                lambda x: (self.data_path / x).exists() if pd.notna(x) else False
            )
        
        return df
    
    def _cleanup_dataframe(self, df: pd.DataFrame, mode: str, balanced: Optional[bool], include_paths: bool, keep_availability_columns: bool = False) -> pd.DataFrame:
        """Clean up the dataframe by removing unnecessary columns."""
        # Remove internal availability and path columns
        cols_to_remove = []
        
        for col in df.columns:
            # Only remove availability and split columns if keep_availability_columns is False
            if not keep_availability_columns:
                if col.startswith('available_') or col.startswith('split_') and col != 'split':
                    cols_to_remove.append(col)
            if col.endswith('_balanced') or col.endswith('_complete'):
                if col.startswith('audio_path_') or col.startswith('lyrics_path_'):
                    if include_paths:  # Only remove if we've created unified columns
                        cols_to_remove.append(col)
        
        df = df.drop(columns=cols_to_remove, errors='ignore')
        
        # Reorder columns for better usability
        priority_cols = [
            'song_id', 'artist', 'title', 'quadrant', 'arousal', 'valence',
            'split', 'audio_path', 'lyrics_path', 'duration', 'actual_year'
        ]
        
        existing_priority_cols = [col for col in priority_cols if col in df.columns]
        other_cols = [col for col in df.columns if col not in existing_priority_cols]
        
        df = df[existing_priority_cols + other_cols]
        
        return df
    
    def get_dataset_info(self) -> Dict:
        """Get comprehensive dataset information."""
        info = {
            'total_songs': len(self.metadata),
            'versions': self.get_available_versions(),
            'quadrants': self.metadata['quadrant'].value_counts().to_dict(),
            'modalities': {}
        }
        
        # Count available samples by modality
        for modality in ['audio', 'lyrics', 'bimodal']:
            balanced_col = f'available_{modality}_balanced'
            complete_col = f'available_{modality}_complete'
            
            info['modalities'][modality] = {
                'balanced': self.metadata[balanced_col].sum() if balanced_col in self.metadata.columns else 0,
                'complete': self.metadata[complete_col].sum() if complete_col in self.metadata.columns else 0
            }
        
        return info
    
    def export_subset(
        self,
        output_path: str,
        version: str = 'v1.1',
        split: Optional[str] = None,
        strategy: str = '70_15_15',
        balanced: Optional[bool] = None,
        mode: str = 'bimodal',
        format: str = 'csv'
    ):
        """
        Export a filtered subset to a file.
        
        Args:
            output_path: Path to save the exported file
            version, split, strategy, balanced, mode: Same as load_merge_dataset
            format: Export format ('csv', 'json', 'parquet')
        """
        df = self.load_merge_dataset(
            version=version, split=split, strategy=strategy,
            balanced=balanced, mode=mode, return_format='pandas'
        )
        
        output_path = Path(output_path)
        
        if format == 'csv':
            df.to_csv(output_path, index=False)
        elif format == 'json':
            df.to_json(output_path, orient='records', indent=2)
        elif format == 'parquet':
            df.to_parquet(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Exported {len(df)} records to {output_path}")
    
    def load_for_analysis(
        self,
        version: str = 'v1.1',
        include_paths: bool = True
    ) -> pd.DataFrame:
        """
        Load the complete dataset with all availability columns for analysis.
        
        Args:
            version: Dataset version (default: 'v1.1')
            include_paths: Include file paths in the result
            
        Returns:
            DataFrame with all availability columns preserved
        """
        return self.load_merge_dataset(
            version=version,
            split='all',
            mode='all',
            balanced=None,
            include_paths=include_paths,
            keep_availability_columns=True
        )

    def export_dataset(self, df: pd.DataFrame, output_path: str, format: str = 'csv'):
        """
        Export the dataset to a file.
        
        Args:
            df: DataFrame to export
            output_path: Path for the output file
            format: Output format ('csv', 'json', 'parquet')
        """
        output_path = Path(output_path)
        
        if format == 'csv':
            df.to_csv(output_path, index=False)
        elif format == 'json':
            df.to_json(output_path, orient='records', indent=2)
        elif format == 'parquet':
            df.to_parquet(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Exported {len(df)} records to {output_path}")


# Convenience function for easy imports
def load_merge_dataset(
    dataset_path: str = "/home/tom/p1_bigdata/merge-dataset",
    version: str = 'v1.1',
    split: Optional[str] = None,
    strategy: str = '70_15_15',
    balanced: Optional[bool] = None,
    mode: str = 'bimodal',
    return_format: str = 'pandas',
    include_paths: bool = True,
    validate_files: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict]]:
    """
    Convenience function to load the MERGE dataset.
    
    Example usage:
        # Load balanced audio training set
        df = load_merge_dataset(mode='audio', balanced=True, split='train')
        
        # Load all bimodal data with metadata
        df, metadata = load_merge_dataset(mode='bimodal', return_format='dict')
        
        # Load complete test set for all modalities
        df = load_merge_dataset(balanced=False, split='test', mode='all')
    """
    loader = MERGEDatasetLoader(dataset_path)
    return loader.load_merge_dataset(
        version=version, split=split, strategy=strategy,
        balanced=balanced, mode=mode, return_format=return_format,
        include_paths=include_paths, validate_files=validate_files
    )


def get_dataset_info(dataset_path: str = "/home/tom/p1_bigdata/merge-dataset") -> Dict:
    """Get comprehensive information about the dataset."""
    loader = MERGEDatasetLoader(dataset_path)
    return loader.get_dataset_info()


# Convenience function for analysis
def load_for_analysis(
    dataset_path: str = "/home/tom/p1_bigdata/merge-dataset",
    version: str = 'v1.1',
    include_paths: bool = True
) -> pd.DataFrame:
    """
    Convenience function to load the complete dataset for analysis.
    
    This function loads the dataset with all availability columns preserved,
    making it suitable for modality analysis and data exploration.
    
    Example usage:
        # Load complete dataset for analysis
        df = load_for_analysis()
        
        # Check availability
        print("Audio available:", df['available_audio_balanced'].sum())
        print("Lyrics available:", df['available_lyrics_balanced'].sum())
    """
    loader = MERGEDatasetLoader(dataset_path)
    return loader.load_for_analysis(version=version, include_paths=include_paths)


if __name__ == "__main__":
    # Example usage
    loader = MERGEDatasetLoader("/home/tom/p1_bigdata/merge-dataset")
    
    print("Dataset Info:")
    info = loader.get_dataset_info()
    for key, value in info.items():
        print(f"{key}: {value}")
    
    print("\nLoading balanced audio training set:")
    df = loader.load_merge_dataset(mode='audio', balanced=True, split='train')
    print(f"Loaded {len(df)} samples")
    print(df[['song_id', 'artist', 'title', 'quadrant']].head())
