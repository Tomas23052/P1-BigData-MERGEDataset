#!/usr/bin/env python3
"""
MERGE Dataset Download Script
============================

This script downloads the original MERGE dataset from Zenodo and prepares it for ETL processing.

Author: Generated for Big Data Processing Project
Date: 2025-07-15
"""

import requests
import zipfile
import os
import hashlib
from pathlib import Path
import argparse
import logging
from typing import Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MERGEDatasetDownloader:
    """Downloader for the MERGE dataset from Zenodo."""
    
    def __init__(self, target_dir: str = "../extracted"):
        """
        Initialize the downloader.
        
        Args:
            target_dir: Directory to extract files to
        """
        self.target_dir = Path(target_dir)
        self.zenodo_base_url = "https://zenodo.org/api/records/13939205"
        
    def get_download_urls(self) -> dict:
        """Get download URLs from Zenodo API."""
        logger.info("Fetching download URLs from Zenodo API...")
        
        try:
            response = requests.get(self.zenodo_base_url)
            response.raise_for_status()
            
            data = response.json()
            files = data.get('files', [])
            
            download_urls = {}
            for file_info in files:
                filename = file_info['key']
                download_url = file_info['links']['self']
                size = file_info['size']
                checksum = file_info['checksum'].split(':')[1]  # Remove algorithm prefix
                
                download_urls[filename] = {
                    'url': download_url,
                    'size': size,
                    'checksum': checksum
                }
            
            logger.info(f"Found {len(download_urls)} files available for download")
            return download_urls
            
        except requests.RequestException as e:
            logger.error(f"Failed to fetch download URLs: {e}")
            raise
    
    def download_file(self, url: str, filename: str, expected_size: int, 
                     expected_checksum: str) -> bool:
        """
        Download a single file with progress tracking and validation.
        
        Args:
            url: Download URL
            filename: Target filename
            expected_size: Expected file size in bytes
            expected_checksum: Expected MD5 checksum
            
        Returns:
            True if download successful, False otherwise
        """
        filepath = Path(filename)
        
        # Check if file already exists and is valid
        if filepath.exists():
            if filepath.stat().st_size == expected_size:
                if self.verify_checksum(filepath, expected_checksum):
                    logger.info(f"✅ {filename} already exists and is valid")
                    return True
                else:
                    logger.warning(f"❌ {filename} exists but checksum mismatch, re-downloading...")
            else:
                logger.warning(f"❌ {filename} exists but size mismatch, re-downloading...")
        
        logger.info(f"📥 Downloading {filename} ({expected_size / 1024 / 1024:.1f} MB)...")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            downloaded_size = 0
            chunk_size = 8192
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        
                        # Progress indicator
                        progress = (downloaded_size / expected_size) * 100
                        if downloaded_size % (1024 * 1024) == 0:  # Every MB
                            logger.info(f"  Progress: {progress:.1f}%")
            
            # Verify download
            if downloaded_size != expected_size:
                logger.error(f"❌ Size mismatch for {filename}: {downloaded_size} != {expected_size}")
                return False
            
            if not self.verify_checksum(filepath, expected_checksum):
                logger.error(f"❌ Checksum mismatch for {filename}")
                return False
            
            logger.info(f"✅ Successfully downloaded {filename}")
            return True
            
        except requests.RequestException as e:
            logger.error(f"❌ Failed to download {filename}: {e}")
            return False
    
    def verify_checksum(self, filepath: Path, expected_checksum: str) -> bool:
        """Verify file checksum."""
        md5_hash = hashlib.md5()
        
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
        
        actual_checksum = md5_hash.hexdigest()
        return actual_checksum == expected_checksum
    
    def extract_files(self, zip_filename: str) -> bool:
        """Extract ZIP files to target directory."""
        zip_path = Path(zip_filename)
        
        if not zip_path.exists():
            logger.error(f"❌ ZIP file not found: {zip_filename}")
            return False
        
        logger.info(f"📦 Extracting {zip_filename} to {self.target_dir}...")
        
        try:
            self.target_dir.mkdir(parents=True, exist_ok=True)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.target_dir)
            
            logger.info(f"✅ Successfully extracted {zip_filename}")
            return True
            
        except zipfile.BadZipFile as e:
            logger.error(f"❌ Invalid ZIP file {zip_filename}: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to extract {zip_filename}: {e}")
            return False
    
    def download_dataset(self, extract: bool = True, keep_zips: bool = False) -> bool:
        """
        Download the complete MERGE dataset.
        
        Args:
            extract: Whether to extract ZIP files after download
            keep_zips: Whether to keep ZIP files after extraction
            
        Returns:
            True if successful, False otherwise
        """
        logger.info("🎵 Starting MERGE dataset download...")
        
        try:
            # Get download URLs
            download_urls = self.get_download_urls()
            
            # Download each file
            success_count = 0
            for filename, file_info in download_urls.items():
                if self.download_file(
                    file_info['url'], 
                    filename, 
                    file_info['size'], 
                    file_info['checksum']
                ):
                    success_count += 1
                    
                    # Extract if requested and is a ZIP file
                    if extract and filename.endswith('.zip'):
                        if self.extract_files(filename):
                            if not keep_zips:
                                os.remove(filename)
                                logger.info(f"🗑️  Removed {filename} after extraction")
            
            total_files = len(download_urls)
            logger.info(f"📊 Download complete: {success_count}/{total_files} files successful")
            
            if success_count == total_files:
                logger.info("✅ All files downloaded successfully!")
                return True
            else:
                logger.warning(f"⚠️  {total_files - success_count} files failed to download")
                return False
                
        except Exception as e:
            logger.error(f"❌ Download failed: {e}")
            return False


def main():
    """Main function to run the download script."""
    parser = argparse.ArgumentParser(description="Download MERGE dataset from Zenodo")
    parser.add_argument(
        '--target-dir', 
        default='../extracted',
        help='Directory to extract files to (default: ../extracted)'
    )
    parser.add_argument(
        '--no-extract', 
        action='store_true',
        help='Skip extraction of ZIP files'
    )
    parser.add_argument(
        '--keep-zips', 
        action='store_true',
        help='Keep ZIP files after extraction'
    )
    parser.add_argument(
        '--version',
        default='v1.1',
        help='Dataset version to download (default: v1.1)'
    )
    
    args = parser.parse_args()
    
    downloader = MERGEDatasetDownloader(args.target_dir)
    
    success = downloader.download_dataset(
        extract=not args.no_extract,
        keep_zips=args.keep_zips
    )
    
    if success:
        print(f"\n🎉 MERGE dataset download completed successfully!")
        print(f"Files extracted to: {Path(args.target_dir).absolute()}")
        print(f"\nNext steps:")
        print(f"1. Run ETL: python scripts/etl.py")
        print(f"2. Explore data: jupyter notebook notebooks/merge_dataset_exploration.ipynb")
    else:
        print(f"\n❌ Download failed. Please check the logs above for details.")
        exit(1)


if __name__ == "__main__":
    main()
