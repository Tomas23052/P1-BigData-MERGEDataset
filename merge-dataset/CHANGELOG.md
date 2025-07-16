# CHANGELOG

All notable changes to the MERGE dataset reorganization project will be documented in this file.

## [v1.1] - 2025-07-15

### Added
- Complete ETL pipeline for processing raw MERGE dataset
- Unified metadata schema with consistent column naming
- Flexible dataset loader with multiple filtering options
- Comprehensive data exploration Jupyter notebook
- File integrity checking with MD5 hashes
- Train/validation/test split management
- Version tracking and reproducibility features

### Changed
- Reorganized file structure following best practices
- Consolidated fragmented metadata into single base file
- Normalized column names to snake_case convention
- Unified bimodal dataset handling (Audio_Song + Lyric_Song)

### Fixed
- Inconsistent naming across different subsets
- Manual preprocessing requirements
- Poor documentation and usability issues
- Fragmented data access patterns

### Technical Details
- Total songs processed: 6,255 unique tracks
- Modalities: Audio (3,554), Lyrics (2,568), Bimodal (2,216)
- Split strategies: 70-15-15 and 40-30-30
- File integrity: MD5 hashes for 6,122 data files
- Metadata schema: 25+ standardized fields

## [Original] - 2024-10-09

### Original Dataset
- Source: MIRlab at CISUC
- Publication: "MERGE - A Bimodal Dataset For Static Music Emotion Recognition"
- Authors: Louro, P. L. et al.
- Platform: Zenodo (https://zenodo.org/records/13939205)
- License: Original dataset licensing terms apply
