#!/usr/bin/env python3
"""
MERGE Dataset ETL Script
========================

This script processes the MERGE dataset files, consolidates metadata,
and creates a unified structure following best practices.

Author: Generated for Big Data Processing Project
Date: 2025-07-15
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import hashlib
from typing import Dict, List, Optional, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MERGEDatasetETL:
    """ETL processor for the MERGE dataset."""
    
    def __init__(self, source_dir: str, target_dir: str, version: str = "v1.1"):
        """
        Initialize the ETL processor.
        
        Args:
            source_dir: Path to extracted MERGE dataset files
            target_dir: Path to target merge-dataset directory
            version: Dataset version
        """
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.version = version
        
        # Define subset mappings
        self.subsets = {
            'MERGE_Audio_Balanced': {'subset': 'audio', 'balanced': True},
            'MERGE_Audio_Complete': {'subset': 'audio', 'balanced': False},
            'MERGE_Lyrics_Balanced': {'subset': 'lyrics', 'balanced': True},
            'MERGE_Lyrics_Complete': {'subset': 'lyrics', 'balanced': False},
            'MERGE_Bimodal_Balanced': {'subset': 'bimodal', 'balanced': True},
            'MERGE_Bimodal_Complete': {'subset': 'bimodal', 'balanced': False}
        }
        
    def normalize_column_names(self, df: pd.DataFrame, is_bimodal: bool = False) -> pd.DataFrame:
        """Normalize column names to snake_case."""
        column_mapping = {
            'Song': 'song_id',
            'Audio_Song': 'audio_song_id',
            'Lyric_Song': 'lyrics_song_id',
            'Quadrant': 'quadrant',
            'AllMusic Id': 'allmusic_id',
            'AllMusic Extraction Date': 'allmusic_extraction_date',
            'Artist': 'artist',
            'Title': 'title',
            'Relevance': 'relevance',
            'Year': 'year',
            'LowestYear': 'lowest_year',
            'Duration': 'duration',
            'Moods': 'moods',
            'MoodsAll': 'moods_all',
            'MoodsAllWeights': 'moods_all_weights',
            'Genres': 'genres',
            'GenreWeights': 'genre_weights',
            'Themes': 'themes',
            'ThemeWeights': 'theme_weights',
            'Styles': 'styles',
            'StyleWeights': 'style_weights',
            'AppearancesTrackIDs': 'appearances_track_ids',
            'AppearancesAlbumIDs': 'appearances_album_ids',
            'Sample': 'sample',
            'SampleURL': 'sample_url',
            'ActualYear': 'actual_year',
            'num_Genres': 'num_genres',
            'num_MoodsAll': 'num_moods_all',
            'Arousal': 'arousal',
            'Valence': 'valence'
        }
        
        df = df.rename(columns=column_mapping)
        
        # For bimodal datasets, create a unified song_id
        if is_bimodal and 'audio_song_id' in df.columns and 'lyrics_song_id' in df.columns:
            df['song_id'] = df['audio_song_id'] + '_' + df['lyrics_song_id']
            # Keep the original columns for split matching
            # df = df.drop(['audio_song_id', 'lyrics_song_id'], axis=1)  # Don't drop these yet
        
        return df
    
    def load_metadata_file(self, subset_dir: Path) -> pd.DataFrame:
        """Load and normalize metadata for a single subset."""
        metadata_files = list(subset_dir.glob("*_metadata.csv"))
        if not metadata_files:
            raise FileNotFoundError(f"No metadata file found in {subset_dir}")
        
        metadata_file = metadata_files[0]
        logger.info(f"Loading metadata from {metadata_file}")
        
        df = pd.read_csv(metadata_file)
        
        # Check if this is a bimodal dataset
        is_bimodal = 'Bimodal' in str(subset_dir)
        df = self.normalize_column_names(df, is_bimodal=is_bimodal)
        
        return df
    
    def load_av_values(self, subset_dir: Path) -> pd.DataFrame:
        """Load arousal/valence values for a subset."""
        av_files = list(subset_dir.glob("*_av_values.csv"))
        if not av_files:
            logger.warning(f"No av_values file found in {subset_dir}")
            return pd.DataFrame()
        
        av_file = av_files[0]
        logger.info(f"Loading AV values from {av_file}")
        
        df = pd.read_csv(av_file)
        
        # Check if this is a bimodal dataset
        is_bimodal = 'Bimodal' in str(subset_dir)
        df = self.normalize_column_names(df, is_bimodal=is_bimodal)
        
        return df
    
    def load_split_assignments(self, subset_dir: Path) -> Dict[str, pd.DataFrame]:
        """Load train/validation/test split assignments."""
        splits = {}
        tvt_dir = subset_dir / "tvt_dataframes"
        
        if not tvt_dir.exists():
            logger.warning(f"No tvt_dataframes directory found in {subset_dir}")
            return splits
        
        for strategy_dir in tvt_dir.iterdir():
            if strategy_dir.is_dir():
                strategy = strategy_dir.name
                splits[strategy] = {}
                
                for split_file in strategy_dir.glob("*.csv"):
                    # Extract split name from filename
                    split_name = split_file.stem.split('_')[4]  # e.g., 'train', 'validate', 'test'
                    
                    df = pd.read_csv(split_file)
                    is_bimodal = 'Bimodal' in str(subset_dir)
                    df = self.normalize_column_names(df, is_bimodal=is_bimodal)
                    splits[strategy][split_name] = df
                    
                    logger.info(f"Loaded {len(df)} samples for {strategy}/{split_name}")
        
        return splits
    
    def add_file_paths(self, df: pd.DataFrame, subset_dir: Path, subset_info: Dict) -> pd.DataFrame:
        """Add file paths to the dataframe based on available files."""
        df = df.copy()
        df['audio_path'] = None
        df['lyrics_path'] = None
        
        subset_type = subset_info['subset']
        is_bimodal = subset_type == 'bimodal'
        
        if subset_type in ['audio', 'bimodal']:
            # Look for audio files
            if subset_type == 'audio':
                # Audio files are in Q1, Q2, Q3, Q4 directories
                for quadrant in ['Q1', 'Q2', 'Q3', 'Q4']:
                    quad_dir = subset_dir / quadrant
                    if quad_dir.exists():
                        for audio_file in quad_dir.glob("*.mp3"):
                            song_id = audio_file.stem
                            if song_id in df['song_id'].values:
                                df.loc[df['song_id'] == song_id, 'audio_path'] = f"audio/{quadrant}/{audio_file.name}"
            else:
                # Bimodal - audio files are in audio/ subdirectory
                audio_dir = subset_dir / "audio"
                if audio_dir.exists():
                    for quadrant in ['Q1', 'Q2', 'Q3', 'Q4']:
                        quad_dir = audio_dir / quadrant
                        if quad_dir.exists():
                            for audio_file in quad_dir.glob("*.mp3"):
                                song_id = audio_file.stem
                                # For bimodal, match against audio_song_id
                                if is_bimodal and 'audio_song_id' in df.columns:
                                    mask = df['audio_song_id'] == song_id
                                    if mask.any():
                                        df.loc[mask, 'audio_path'] = f"audio/{quadrant}/{audio_file.name}"
                                elif song_id in df['song_id'].values:
                                    df.loc[df['song_id'] == song_id, 'audio_path'] = f"audio/{quadrant}/{audio_file.name}"
        
        if subset_type in ['lyrics', 'bimodal']:
            # Look for lyrics files
            lyrics_base = subset_dir if subset_type == 'lyrics' else subset_dir / "lyrics"
            if lyrics_base.exists():
                for quadrant in ['Q1', 'Q2', 'Q3', 'Q4']:
                    quad_dir = lyrics_base / quadrant
                    if quad_dir.exists():
                        for lyrics_file in quad_dir.glob("*.txt"):
                            song_id = lyrics_file.stem
                            # For bimodal, match against lyrics_song_id
                            if is_bimodal and 'lyrics_song_id' in df.columns:
                                mask = df['lyrics_song_id'] == song_id
                                if mask.any():
                                    df.loc[mask, 'lyrics_path'] = f"lyrics/{quadrant}/{lyrics_file.name}"
                            elif song_id in df['song_id'].values:
                                df.loc[df['song_id'] == song_id, 'lyrics_path'] = f"lyrics/{quadrant}/{lyrics_file.name}"
        
        return df
    
    def process_subset(self, subset_name: str) -> pd.DataFrame:
        """Process a single subset and return consolidated DataFrame."""
        subset_dir = self.source_dir / subset_name
        subset_info = self.subsets[subset_name]
        
        logger.info(f"Processing subset: {subset_name}")
        
        # Load metadata
        df = self.load_metadata_file(subset_dir)
        
        # Add subset information
        df['subset'] = subset_info['subset']
        df['balanced'] = subset_info['balanced']
        df['version'] = self.version
        
        # Load and merge arousal/valence values
        av_df = self.load_av_values(subset_dir)
        if not av_df.empty:
            df = df.merge(av_df, on='song_id', how='left')
        
        # Add file paths
        df = self.add_file_paths(df, subset_dir, subset_info)
        
        # Load split assignments
        splits = self.load_split_assignments(subset_dir)
        
        # Add split information
        for strategy, strategy_splits in splits.items():
            split_col = f"split_{strategy.replace('tvt_', '')}"
            df[split_col] = None
            
            for split_name, split_df in strategy_splits.items():
                song_ids = split_df['song_id'].values
                
                # For bimodal datasets, match against audio_song_id
                # Handle both direct column names and merge suffixes (_x, _y)
                audio_col = None
                for col in ['audio_song_id', 'audio_song_id_x', 'audio_song_id_y']:
                    if col in df.columns:
                        audio_col = col
                        break
                
                if audio_col and any('lyrics_song_id' in col for col in df.columns):
                    mask = df[audio_col].isin(song_ids)
                else:
                    mask = df['song_id'].isin(song_ids)
                
                df.loc[mask, split_col] = split_name
        
        logger.info(f"Processed {len(df)} samples from {subset_name}")
        return df
    
    def consolidate_metadata(self) -> pd.DataFrame:
        """Consolidate metadata from all subsets into a single DataFrame."""
        all_dfs = []
        
        for subset_name in self.subsets.keys():
            subset_dir = self.source_dir / subset_name
            if subset_dir.exists():
                df = self.process_subset(subset_name)
                all_dfs.append(df)
            else:
                logger.warning(f"Subset directory not found: {subset_dir}")
        
        if not all_dfs:
            raise ValueError("No subset data found")
        
        # Concatenate all DataFrames
        consolidated_df = pd.concat(all_dfs, ignore_index=True)
        
        # Remove duplicates (songs might appear in multiple subsets)
        # Keep the most complete record (bimodal > audio/lyrics > balanced > complete)
        subset_priority = {'bimodal': 3, 'audio': 2, 'lyrics': 2}
        consolidated_df['subset_priority'] = consolidated_df['subset'].map(subset_priority).fillna(1)
        consolidated_df['balanced_priority'] = consolidated_df['balanced'].astype(int)
        
        # Sort by priority and drop duplicates
        consolidated_df = consolidated_df.sort_values([
            'song_id', 'subset_priority', 'balanced_priority'
        ], ascending=[True, False, False])
        
        # For true consolidation, we need to merge information rather than drop duplicates
        # Let's create a proper consolidated view
        return self.create_master_metadata(all_dfs)
    
    def create_master_metadata(self, all_dfs: List[pd.DataFrame]) -> pd.DataFrame:
        """Create a master metadata DataFrame by intelligently merging all subsets."""
        # Start with a base DataFrame containing all unique songs
        all_songs = set()
        for df in all_dfs:
            all_songs.update(df['song_id'].unique())
        
        # Find the common columns across all dataframes
        common_cols = set(all_dfs[0].columns)
        for df in all_dfs[1:]:
            common_cols = common_cols.intersection(set(df.columns))
        
        # Create base DataFrame with song metadata (use the most complete version)
        base_df = None
        for df in all_dfs:
            # Use columns that exist in this dataframe and are in common columns
            available_cols = [col for col in [
                'song_id', 'quadrant', 'allmusic_id', 'allmusic_extraction_date',
                'artist', 'title', 'relevance', 'year', 'lowest_year', 'duration',
                'moods', 'moods_all', 'moods_all_weights', 'genres', 'genre_weights',
                'themes', 'theme_weights', 'styles', 'style_weights',
                'appearances_track_ids', 'appearances_album_ids', 'sample',
                'sample_url', 'actual_year', 'num_genres', 'num_moods_all',
                'arousal', 'valence', 'audio_song_id', 'lyrics_song_id'
            ] if col in df.columns]
            
            if base_df is None:
                base_df = df[available_cols].copy()
            else:
                # Merge additional songs
                new_songs = df[~df['song_id'].isin(base_df['song_id'])]
                if not new_songs.empty:
                    # Only use columns that exist in both dataframes
                    common_cols_for_merge = [col for col in available_cols if col in base_df.columns]
                    base_df = pd.concat([base_df, new_songs[common_cols_for_merge]], ignore_index=True)
        
        # Add availability flags for each subset/balanced combination
        for subset_name, subset_info in self.subsets.items():
            subset_type = subset_info['subset']
            balanced = subset_info['balanced']
            
            col_name = f"available_{subset_type}_{'balanced' if balanced else 'complete'}"
            base_df[col_name] = False
            
            # Find matching DataFrame
            matching_df = None
            for df in all_dfs:
                if (df['subset'].iloc[0] == subset_type and 
                    df['balanced'].iloc[0] == balanced):
                    matching_df = df
                    break
            
            if matching_df is not None:
                available_songs = matching_df['song_id'].unique()
                base_df.loc[base_df['song_id'].isin(available_songs), col_name] = True
                
                # Add file paths
                audio_col = f"audio_path_{'balanced' if balanced else 'complete'}"
                lyrics_col = f"lyrics_path_{'balanced' if balanced else 'complete'}"
                
                if subset_type in ['audio', 'bimodal'] and 'audio_path' in matching_df.columns:
                    base_df[audio_col] = None
                    for _, row in matching_df.iterrows():
                        if pd.notna(row['audio_path']):
                            base_df.loc[base_df['song_id'] == row['song_id'], audio_col] = row['audio_path']
                
                if subset_type in ['lyrics', 'bimodal'] and 'lyrics_path' in matching_df.columns:
                    base_df[lyrics_col] = None
                    for _, row in matching_df.iterrows():
                        if pd.notna(row['lyrics_path']):
                            base_df.loc[base_df['song_id'] == row['song_id'], lyrics_col] = row['lyrics_path']
                
                # Add split information
                split_cols = [col for col in matching_df.columns if col.startswith('split_')]
                for split_col in split_cols:
                    new_col = f"{split_col}_{'balanced' if balanced else 'complete'}_{subset_type}"
                    base_df[new_col] = None
                    for _, row in matching_df.iterrows():
                        if pd.notna(row[split_col]):
                            base_df.loc[base_df['song_id'] == row['song_id'], new_col] = row[split_col]
        
        base_df['version'] = self.version
        
        return base_df
    
    def save_metadata(self, df: pd.DataFrame):
        """Save the consolidated metadata to the target directory."""
        metadata_dir = self.target_dir / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Save master metadata
        master_file = metadata_dir / "base_metadata.csv"
        df.to_csv(master_file, index=False)
        logger.info(f"Saved master metadata with {len(df)} records to {master_file}")
        
        # Save version-specific files
        version_dir = metadata_dir / "versions" / self.version
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a summary file
        summary = {
            'total_songs': len(df),
            'audio_balanced': df['available_audio_balanced'].sum(),
            'audio_complete': df['available_audio_complete'].sum(),
            'lyrics_balanced': df['available_lyrics_balanced'].sum(),
            'lyrics_complete': df['available_lyrics_complete'].sum(),
            'bimodal_balanced': df['available_bimodal_balanced'].sum(),
            'bimodal_complete': df['available_bimodal_complete'].sum(),
            'by_quadrant': df['quadrant'].value_counts().to_dict()
        }
        
        summary_file = version_dir / "summary.yml"
        with open(summary_file, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)
        
        logger.info(f"Saved version summary to {summary_file}")
    
    def copy_data_files(self):
        """Copy and organize data files to the target structure."""
        data_dir = self.target_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (data_dir / "audio").mkdir(exist_ok=True)
        (data_dir / "lyrics").mkdir(exist_ok=True)
        
        for subset_name in self.subsets.keys():
            subset_dir = self.source_dir / subset_name
            if not subset_dir.exists():
                continue
                
            logger.info(f"Copying data files from {subset_name}")
            
            # Copy audio files
            if 'Audio' in subset_name or 'Bimodal' in subset_name:
                audio_source = subset_dir if 'Audio' in subset_name else subset_dir / "audio"
                if audio_source.exists():
                    for quadrant in ['Q1', 'Q2', 'Q3', 'Q4']:
                        quad_dir = audio_source / quadrant
                        if quad_dir.exists():
                            target_quad_dir = data_dir / "audio" / quadrant
                            target_quad_dir.mkdir(parents=True, exist_ok=True)
                            
                            import shutil
                            for audio_file in quad_dir.glob("*.mp3"):
                                target_file = target_quad_dir / audio_file.name
                                if not target_file.exists():
                                    shutil.copy2(audio_file, target_file)
            
            # Copy lyrics files
            if 'Lyrics' in subset_name or 'Bimodal' in subset_name:
                lyrics_source = subset_dir if 'Lyrics' in subset_name else subset_dir / "lyrics"
                if lyrics_source.exists():
                    for quadrant in ['Q1', 'Q2', 'Q3', 'Q4']:
                        quad_dir = lyrics_source / quadrant
                        if quad_dir.exists():
                            target_quad_dir = data_dir / "lyrics" / quadrant
                            target_quad_dir.mkdir(parents=True, exist_ok=True)
                            
                            import shutil
                            for lyrics_file in quad_dir.glob("*.txt"):
                                target_file = target_quad_dir / lyrics_file.name
                                if not target_file.exists():
                                    shutil.copy2(lyrics_file, target_file)
    
    def generate_file_hashes(self):
        """Generate file hashes for integrity checking."""
        data_dir = self.target_dir / "data"
        schema_dir = self.target_dir / "schema"
        schema_dir.mkdir(parents=True, exist_ok=True)
        
        hashes = []
        
        for file_path in data_dir.rglob("*"):
            if file_path.is_file():
                with open(file_path, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                
                rel_path = file_path.relative_to(data_dir)
                hashes.append({
                    'file_path': str(rel_path),
                    'md5_hash': file_hash,
                    'size_bytes': file_path.stat().st_size
                })
        
        hash_df = pd.DataFrame(hashes)
        hash_file = schema_dir / "data_hashes.csv"
        hash_df.to_csv(hash_file, index=False)
        
        logger.info(f"Generated hashes for {len(hashes)} files: {hash_file}")
    
    def run_etl(self):
        """Run the complete ETL process."""
        logger.info("Starting MERGE Dataset ETL process")
        
        # Step 1: Consolidate metadata
        logger.info("Step 1: Consolidating metadata")
        consolidated_df = self.consolidate_metadata()
        
        # Step 2: Save metadata
        logger.info("Step 2: Saving consolidated metadata")
        self.save_metadata(consolidated_df)
        
        # Step 3: Copy and organize data files
        logger.info("Step 3: Copying and organizing data files")
        self.copy_data_files()
        
        # Step 4: Generate file hashes
        logger.info("Step 4: Generating file integrity hashes")
        self.generate_file_hashes()
        
        logger.info("ETL process completed successfully")
        return consolidated_df


def main():
    """Main function to run the ETL process."""
    source_dir = "/home/tom/p1_bigdata/extracted"
    target_dir = "/home/tom/p1_bigdata/merge-dataset"
    
    etl = MERGEDatasetETL(source_dir, target_dir, version="v1.1")
    consolidated_df = etl.run_etl()
    
    print(f"\nETL Summary:")
    print(f"Total songs processed: {len(consolidated_df)}")
    print(f"Quadrant distribution:")
    print(consolidated_df['quadrant'].value_counts())
    
    print(f"\nAvailability by subset:")
    for col in consolidated_df.columns:
        if col.startswith('available_'):
            print(f"{col}: {consolidated_df[col].sum()}")


if __name__ == "__main__":
    main()
