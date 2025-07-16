# MERGE Dataset - Reorganized and Enhanced

This repository contains a reorganized and enhanced version of the MERGE dataset for Music Emotion Recognition (MER). The original dataset was developed by the MIRlab at CISUC and is available on [Zenodo](https://zenodo.org/records/13939205).

## 📁 Repository Structure

```
merge-dataset/
├── data/                     # All raw and processed data files
│   ├── audio/               # Audio files organized by quadrant (Q1-Q4)
│   └── lyrics/              # Lyrics files organized by quadrant (Q1-Q4)
├── metadata/                # All metadata and split information
│   ├── base_metadata.csv    # Master consolidated metadata file
│   └── versions/            # Version-specific information
│       └── v1.1/
│           └── summary.yml  # Dataset summary statistics
├── schema/                  # Data dictionary and integrity information
│   ├── metadata_schema.yml  # Field definitions and constraints
│   └── data_hashes.csv     # File integrity hashes
├── scripts/                 # ETL, loader, and utility scripts
│   ├── etl.py              # Main ETL processing script
│   └── loader.py           # Dataset loader with flexible filtering
├── notebooks/               # Jupyter notebooks for exploration and demos
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone/download this repository
cd merge-dataset

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Load the Dataset

```python
from scripts.loader import load_merge_dataset

# Load balanced audio training set
df = load_merge_dataset(
    mode='audio', 
    balanced=True, 
    split='train'
)

# Load all bimodal data with metadata
df, metadata = load_merge_dataset(
    mode='bimodal', 
    return_format='dict'
)

# Load complete test set for all modalities
df = load_merge_dataset(
    balanced=False, 
    split='test', 
    mode='all'
)
```

### 3. Explore the Data

```python
import pandas as pd
from scripts.loader import get_dataset_info

# Get dataset overview
info = get_dataset_info()
print(info)

# Load and examine the data
df = load_merge_dataset(mode='bimodal', balanced=True)
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Quadrant distribution:\n{df['quadrant'].value_counts()}")
```

## 📊 Dataset Overview

The MERGE dataset contains multimodal data for Music Emotion Recognition:

- **Total unique songs**: 6,255
- **Audio files**: 6,122 MP3 files
- **Lyrics files**: 4,968 text files
- **Emotion quadrants**: Q1 (High Arousal/High Valence), Q2 (High Arousal/Low Valence), Q3 (Low Arousal/Low Valence), Q4 (Low Arousal/High Valence)

### Subset Availability

| Subset | Balanced | Complete |
|--------|----------|----------|
| Audio | 3,232 | 3,554 |
| Lyrics | 2,400 | 2,568 |
| Bimodal | 2,000 | 2,216 |

### Train/Validation/Test Splits

Two splitting strategies are provided:
- **70-15-15**: 70% train, 15% validation, 15% test
- **40-30-30**: 40% train, 30% validation, 30% test

## 🔧 Loader API

The `load_merge_dataset()` function provides flexible data loading:

```python
def load_merge_dataset(
    version='v1.1',           # Dataset version
    split=None,               # 'train', 'validate', 'test', 'all', or None
    strategy='70_15_15',      # '70_15_15' or '40_30_30'
    balanced=None,            # True (balanced), False (complete), None (both)
    mode='bimodal',           # 'audio', 'lyrics', 'bimodal', 'all'
    return_format='pandas',   # 'pandas' or 'dict'
    include_paths=True,       # Include file paths
    validate_files=False      # Validate file existence
)
```

## 📈 Data Exploration Examples

See the `notebooks/` directory for detailed examples:

- **Data Exploration**: Basic statistics and visualizations
- **Emotion Distribution**: Analysis of arousal/valence distributions
- **Modality Analysis**: Comparison between audio and lyrics subsets

## 🔍 Key Improvements

This reorganized version addresses several issues in the original dataset:

### ✅ Issues Resolved

1. **Poor Documentation**: Clear README, schema documentation, and examples
2. **Inconsistent Naming**: Unified column names and consistent file organization
3. **Fragmented Subsets**: Single consolidated metadata file with availability flags
4. **Manual Preprocessing**: Automated ETL pipeline and easy-to-use loader
5. **Too Many Variants**: Simplified access through flexible loader API
6. **Versioning Issues**: Clear version tracking and reproducible data access

### 🎯 Best Practices Implemented

- **Centralized Metadata**: Single `base_metadata.csv` with all information
- **Clear Schema**: Documented field definitions in `metadata_schema.yml`
- **File Integrity**: MD5 hashes for all data files
- **Flexible Loading**: Version-aware loader supporting multiple output formats
- **Reproducibility**: Clear versioning and split tracking

## 📝 Citation

If you use this reorganized dataset, please cite the original work:

```bibtex
@article{louro2024merge,
  title={MERGE - A Bimodal Dataset For Static Music Emotion Recognition},
  author={Louro, P. L. and others},
  journal={arXiv preprint arXiv:2407.06060},
  year={2024}
}
```

## 🔗 Links

- [Original Dataset on Zenodo](https://zenodo.org/records/13939205)
- [MIRlab at CISUC](https://www.cisuc.uc.pt/projects/mir)
- [Paper on arXiv](https://arxiv.org/abs/2407.06060)

## 📧 Support

For questions about this reorganized version, please create an issue in this repository.
For questions about the original dataset, please contact the MIRlab team.
