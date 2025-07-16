#!/usr/bin/env python3
"""
MERGE Dataset Validation Suite
==============================

Comprehensive validation system for the MERGE dataset using JSON Schema
and custom validation rules.

Author: Generated for Big Data Processing Project
Date: 2025-07-16
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import jsonschema
from jsonschema import validate, ValidationError
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MERGEDatasetValidator:
    """Comprehensive validation for MERGE dataset."""
    
    def __init__(self, dataset_path: str):
        """
        Initialize validator.
        
        Args:
            dataset_path: Path to merge-dataset directory
        """
        self.dataset_path = Path(dataset_path)
        self.metadata_path = self.dataset_path / "metadata" / "base_metadata.csv"
        self.schema_path = self.dataset_path / "schema" / "metadata_schema.json"
        
        # Load JSON schema
        with open(self.schema_path, 'r') as f:
            self.json_schema = json.load(f)
            
    def validate_metadata_schema(self) -> Dict[str, any]:
        """
        Validate metadata against JSON schema.
        
        Returns:
            Dictionary with validation results
        """
        logger.info("Starting metadata schema validation...")
        
        # Load metadata
        df = pd.read_csv(self.metadata_path)
        
        results = {
            'total_records': len(df),
            'valid_records': 0,
            'invalid_records': 0,
            'validation_errors': [],
            'schema_compliance': True
        }
        
        for idx, row in df.iterrows():
            try:
                # Convert row to dict and validate
                record = row.to_dict()
                
                # Handle NaN values (convert to None for JSON schema)
                for key, value in record.items():
                    if pd.isna(value):
                        record[key] = None
                
                validate(instance=record, schema=self.json_schema)
                results['valid_records'] += 1
                
            except ValidationError as e:
                results['invalid_records'] += 1
                results['validation_errors'].append({
                    'record_index': idx,
                    'song_id': row.get('song_id', 'unknown'),
                    'error_message': str(e.message),
                    'error_path': list(e.path)
                })
                
                if len(results['validation_errors']) > 10:  # Limit error reporting
                    break
                    
        if results['invalid_records'] > 0:
            results['schema_compliance'] = False
            
        logger.info(f"Schema validation complete: {results['valid_records']}/{results['total_records']} valid")
        return results
    
    def validate_data_integrity(self) -> Dict[str, any]:
        """
        Validate data integrity and consistency.
        
        Returns:
            Dictionary with integrity validation results
        """
        logger.info("Starting data integrity validation...")
        
        df = pd.read_csv(self.metadata_path)
        
        results = {
            'total_songs': len(df),
            'unique_song_ids': df['song_id'].nunique(),
            'duplicate_song_ids': [],
            'missing_required_fields': [],
            'invalid_quadrants': [],
            'invalid_splits': [],
            'availability_consistency': True,
            'split_consistency': True
        }
        
        # Check for duplicate song IDs
        duplicates = df[df.duplicated(subset=['song_id'], keep=False)]
        if not duplicates.empty:
            results['duplicate_song_ids'] = duplicates['song_id'].unique().tolist()
            
        # Check required fields
        required_fields = ['song_id', 'quadrant', 'version']
        for field in required_fields:
            missing = df[df[field].isna()]
            if not missing.empty:
                results['missing_required_fields'].append({
                    'field': field,
                    'missing_count': len(missing),
                    'song_ids': missing['song_id'].tolist()[:10]  # First 10
                })
        
        # Check quadrant values
        valid_quadrants = ['Q1', 'Q2', 'Q3', 'Q4']
        invalid_quad = df[~df['quadrant'].isin(valid_quadrants)]
        if not invalid_quad.empty:
            results['invalid_quadrants'] = invalid_quad[['song_id', 'quadrant']].to_dict('records')
            
        # Check split consistency
        split_columns = [col for col in df.columns if col.startswith('split_')]
        valid_splits = ['train', 'validate', 'test']
        
        for split_col in split_columns:
            invalid_splits = df[
                df[split_col].notna() & 
                (~df[split_col].isin(valid_splits))
            ]
            if not invalid_splits.empty:
                results['invalid_splits'].append({
                    'column': split_col,
                    'invalid_values': invalid_splits[split_col].unique().tolist()
                })
                results['split_consistency'] = False
        
        # Check availability flag consistency
        availability_cols = [col for col in df.columns if col.startswith('available_')]
        for avail_col in availability_cols:
            if df[avail_col].dtype != 'bool':
                # Try to convert to boolean
                try:
                    df[avail_col] = df[avail_col].astype(bool)
                except:
                    results['availability_consistency'] = False
                    
        logger.info(f"Integrity validation complete: {results['unique_song_ids']} unique songs")
        return results
    
    def validate_file_existence(self) -> Dict[str, any]:
        """
        Validate that referenced files actually exist.
        
        Returns:
            Dictionary with file validation results
        """
        logger.info("Starting file existence validation...")
        
        df = pd.read_csv(self.metadata_path)
        
        results = {
            'total_audio_references': 0,
            'missing_audio_files': [],
            'total_lyrics_references': 0,
            'missing_lyrics_files': [],
            'file_integrity': True
        }
        
        # Check audio files
        audio_path_cols = [col for col in df.columns if 'audio_path' in col]
        for col in audio_path_cols:
            valid_paths = df[df[col].notna()][col]
            results['total_audio_references'] += len(valid_paths)
            
            for path in valid_paths:
                full_path = self.dataset_path / "data" / path
                if not full_path.exists():
                    results['missing_audio_files'].append(str(path))
                    results['file_integrity'] = False
        
        # Check lyrics files  
        lyrics_path_cols = [col for col in df.columns if 'lyrics_path' in col]
        for col in lyrics_path_cols:
            valid_paths = df[df[col].notna()][col]
            results['total_lyrics_references'] += len(valid_paths)
            
            for path in valid_paths:
                full_path = self.dataset_path / "data" / path
                if not full_path.exists():
                    results['missing_lyrics_files'].append(str(path))
                    results['file_integrity'] = False
                    
        logger.info(f"File validation complete: {len(results['missing_audio_files']) + len(results['missing_lyrics_files'])} missing files")
        return results
    
    def run_comprehensive_validation(self) -> Dict[str, any]:
        """
        Run all validation checks.
        
        Returns:
            Complete validation report
        """
        logger.info("Starting comprehensive validation suite...")
        
        validation_report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'dataset_path': str(self.dataset_path),
            'validation_summary': {
                'overall_status': 'UNKNOWN',
                'total_tests': 3,
                'passed_tests': 0,
                'failed_tests': 0
            },
            'schema_validation': {},
            'integrity_validation': {},
            'file_validation': {}
        }
        
        # Run schema validation
        try:
            schema_results = self.validate_metadata_schema()
            validation_report['schema_validation'] = schema_results
            if schema_results['schema_compliance']:
                validation_report['validation_summary']['passed_tests'] += 1
            else:
                validation_report['validation_summary']['failed_tests'] += 1
        except Exception as e:
            logger.error(f"Schema validation failed: {e}")
            validation_report['schema_validation'] = {'error': str(e)}
            validation_report['validation_summary']['failed_tests'] += 1
            
        # Run integrity validation
        try:
            integrity_results = self.validate_data_integrity()
            validation_report['integrity_validation'] = integrity_results
            if (len(integrity_results['duplicate_song_ids']) == 0 and 
                len(integrity_results['missing_required_fields']) == 0 and
                integrity_results['split_consistency'] and
                integrity_results['availability_consistency']):
                validation_report['validation_summary']['passed_tests'] += 1
            else:
                validation_report['validation_summary']['failed_tests'] += 1
        except Exception as e:
            logger.error(f"Integrity validation failed: {e}")
            validation_report['integrity_validation'] = {'error': str(e)}
            validation_report['validation_summary']['failed_tests'] += 1
            
        # Run file validation
        try:
            file_results = self.validate_file_existence()
            validation_report['file_validation'] = file_results
            if file_results['file_integrity']:
                validation_report['validation_summary']['passed_tests'] += 1
            else:
                validation_report['validation_summary']['failed_tests'] += 1
        except Exception as e:
            logger.error(f"File validation failed: {e}")
            validation_report['file_validation'] = {'error': str(e)}
            validation_report['validation_summary']['failed_tests'] += 1
        
        # Determine overall status
        if validation_report['validation_summary']['failed_tests'] == 0:
            validation_report['validation_summary']['overall_status'] = 'PASSED'
        elif validation_report['validation_summary']['passed_tests'] > 0:
            validation_report['validation_summary']['overall_status'] = 'PARTIAL'
        else:
            validation_report['validation_summary']['overall_status'] = 'FAILED'
            
        logger.info(f"Comprehensive validation complete: {validation_report['validation_summary']['overall_status']}")
        return validation_report
    
    def generate_validation_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate human-readable validation report.
        
        Args:
            output_path: Optional path to save report
            
        Returns:
            Report as string
        """
        validation_results = self.run_comprehensive_validation()
        
        report = f"""
# MERGE Dataset Validation Report
Generated: {validation_results['timestamp']}
Dataset: {validation_results['dataset_path']}

## Overall Status: {validation_results['validation_summary']['overall_status']}
- Tests Passed: {validation_results['validation_summary']['passed_tests']}/3
- Tests Failed: {validation_results['validation_summary']['failed_tests']}/3

## Schema Validation Results
"""
        
        schema_results = validation_results['schema_validation']
        if 'error' not in schema_results:
            report += f"""
- Total Records: {schema_results['total_records']}
- Valid Records: {schema_results['valid_records']}
- Invalid Records: {schema_results['invalid_records']}
- Schema Compliance: {'✅ PASS' if schema_results['schema_compliance'] else '❌ FAIL'}
"""
            if schema_results['validation_errors']:
                report += "\n### Schema Errors (First 10):\n"
                for error in schema_results['validation_errors'][:10]:
                    report += f"- Song {error['song_id']}: {error['error_message']}\n"
        else:
            report += f"❌ ERROR: {schema_results['error']}\n"
            
        report += "\n## Data Integrity Results\n"
        integrity_results = validation_results['integrity_validation']
        if 'error' not in integrity_results:
            report += f"""
- Total Songs: {integrity_results['total_songs']}
- Unique Song IDs: {integrity_results['unique_song_ids']}
- Duplicate IDs: {len(integrity_results['duplicate_song_ids'])} {'✅' if len(integrity_results['duplicate_song_ids']) == 0 else '❌'}
- Missing Required Fields: {len(integrity_results['missing_required_fields'])} {'✅' if len(integrity_results['missing_required_fields']) == 0 else '❌'}
- Split Consistency: {'✅ PASS' if integrity_results['split_consistency'] else '❌ FAIL'}
- Availability Consistency: {'✅ PASS' if integrity_results['availability_consistency'] else '❌ FAIL'}
"""
        else:
            report += f"❌ ERROR: {integrity_results['error']}\n"
            
        report += "\n## File Existence Results\n"
        file_results = validation_results['file_validation']
        if 'error' not in file_results:
            report += f"""
- Audio References: {file_results['total_audio_references']}
- Missing Audio Files: {len(file_results['missing_audio_files'])} {'✅' if len(file_results['missing_audio_files']) == 0 else '❌'}
- Lyrics References: {file_results['total_lyrics_references']}
- Missing Lyrics Files: {len(file_results['missing_lyrics_files'])} {'✅' if len(file_results['missing_lyrics_files']) == 0 else '❌'}
- File Integrity: {'✅ PASS' if file_results['file_integrity'] else '❌ FAIL'}
"""
        else:
            report += f"❌ ERROR: {file_results['error']}\n"
            
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Validation report saved to: {output_path}")
            
        return report


if __name__ == "__main__":
    # Example usage
    validator = MERGEDatasetValidator("/home/tom/p1_bigdata/merge-dataset")
    
    print("🔍 Running comprehensive dataset validation...")
    report = validator.generate_validation_report()
    print(report)
    
    # Save detailed results
    results = validator.run_comprehensive_validation()
    with open("/home/tom/p1_bigdata/merge-dataset/validation_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📋 Detailed results saved to: validation_results.json")
