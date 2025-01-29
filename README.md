# Face of Populism Image Annotation: Model Comparison Study

This repository contains code and datasets for comparing different machine learning models' performance on annotating images from the Face of Populism project. We evaluate several state-of-the-art vision models against human annotations.

## Models Evaluated

- CLIP-based models:
  - OpenAI CLIP (with and without emotion detection)
  - Meta CLIP (with and without emotion detection)
  - Apple Vision CLIP (with and without emotion detection)
- CNN-based models:
  - CNN-strong
  - CNN-weak
  - CNN-fer (Facial Expression Recognition based on `fer` library)

## Datasets

### Face of Populism Dataset
- Primary dataset used for model evaluation
- Contains annotated images from political contexts
- Human annotations available for benchmarking

### MAFW Dataset Reference
- The project includes scripts for MAFW dataset analysis, but the dataset itself is not included
- MAFW dataset access must be requested separately from [MAFW Dataset website](https://mafw-database.github.io/MAFW/)
- If you have MAFW access and wish to run the FER analysis:
  1. Place video files in `data/mafw/videos/`
  2. Place annotation files in `data/mafw/annotations/`
  3. Use the provided analysis scripts in `code/mafw_fer_analysis.py`

## Repository Structure

- `code/` - Model implementations and analysis scripts:
  - CLIP variants (`CLIP-2.py`, `META-CLIP.py`, `apple-CLIP.py` and their -NE variants)
  - CNN implementations (`CNN-strong.py`, `CNN-weak.py`)
  - FER analysis scripts (`mafw_fer_analysis.py`)
  - Utility modules (`utils.py`, `detector.py`)
- `analysis/` - R scripts for analyzing model performance
- `figures/` - Generated visualizations
- `fop/` - Face of Populism dataset images
- `results/` - Model predictions and analysis results
- `data/` - Data directory (MAFW data not included)

## Running the Models

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Use the provided shell scripts to run specific models:
   - `run-model.sh` - Main model execution script
   - `run-CLIP.sh` - For CLIP-based models
   - `run-meta.sh` - For Meta CLIP models
4. Results will be saved in the `results/` directory

## Analysis

The analysis pipeline consists of several R scripts:
- `run.R` - Main analysis entry point
- `setup.R` - Setup and configuration
- Additional analysis scripts in the `analysis/` directory

## FER Analysis

For Facial Expression Recognition (FER) analysis:
1. Install required dependencies from `requirements.txt`
2. If using MAFW dataset (requires separate access):
   - Place MAFW files in appropriate directories
   - Run `mafw_fer_analysis.py` for analysis
3. Results will be saved in `results/fer_analysis/`

## License

See the LICENSE file for details.

Note: The MAFW dataset is not included in this repository and must be obtained separately through the [MAFW website](https://mafw-database.github.io/MAFW/).
