# Soccer Video Analysis Pipeline

This is me trying to regroup all the code I used in a cleaner, more organized way for my soccer video analysis project. The goal is to analyze soccer video clips and predict whether they lead to goals using a combination of visual features and crowd behavior analysis.

## Setup

### Virtual Environments

You need to set up **two separate virtual environments** because of conflicting dependencies:

1. **Main Environment** (for tracking, crowd analysis, and training)
   - Contains PyTorch, ByteTrack dependencies, pandas, scipy, h5py
   - Used for: tracking, crowd analysis, model training, prediction pipeline

2. **Visual Features Environment** (for SlowFast video processing)
   - Contains SlowFast, PyTorchVideo, and related video processing libraries
   - Used for: visual feature extraction only

### Installation

```bash
# Create main environment
conda create -n envsoccer python=3.8
conda activate envsoccer
pip install -r tracking_requirements.txt

# Create visual features environment  
conda create -n envslowfast python=3.8
conda activate envslowfast
pip install -r slowfast_requirements.txt
```

### Configuration

1. Modify all paths in `.env` according to your setup
2. Set your conda environment names in `TRACKING_ENV_NAME` and `VISUAL_ENV_NAME`

## Project Structure

- **`src/data/`** - Data downloading, clip extraction, and dataset creation
- **`src/tracking/`** - ByteTrack tracking and crowd behavior analysis
- **`src/visual_head/`** - SlowFast visual feature extraction
- **`src/multitask_model/`** - Model training and evaluation
- **`src/full_pipeline/`** - End-to-end prediction pipeline with precomputed embeddings
- **`scripts/`** - SLURM job scripts for the Montefiore Alan cluster

This project is designed to run on the **Montefiore Alan cluster** using SLURM for GPU access. All scripts are optimized for SLURM job submission but can also be run directly.

## Usage Pipeline

### Preprocessing (Steps 1-8)

**1. Download SoccerNet Data**
```bash
sbatch scripts/download_job.sh
# Or run directly: python src/data/download_data.py /path/to/data/dir
```

**2. Extract Video Clips**
```bash
sbatch scripts/extract_clips.sh
# Or run directly:
# python src/data/extract_clips.py $DATA_DIR $CLIPS_OUTPUT_DIR --type goals
# python src/data/extract_clips.py $DATA_DIR $CLIPS_OUTPUT_DIR --type background --clips-per-game 5
```

**3. Create Balanced Subset**
```bash
sbatch scripts/create_subset.sh
# Or run directly:
# python src/data/subset_creator.py $ORGANIZED_CLIPS_DIR $SUBSET_OUTPUT_DIR $SUBSET_TOTAL_VIDEOS \
#     --background-prop $SUBSET_BACKGROUND_PROP \
#     --before-goal-prop $SUBSET_BEFORE_GOAL_PROP \
#     --free-kicks-prop $SUBSET_FREE_KICKS_PROP \
#     --penalties-prop $SUBSET_PENALTIES_PROP \
#     --shots-no-goals-prop $SUBSET_SHOTS_NO_GOALS_PROP
```

**4. Track Players in Videos**
```bash
sbatch scripts/track_videos.sh
```

**5. Extract Tracking Features**
```bash
sbatch scripts/extract_tracking_features.sh
# Or run directly: 
# python src/tracking/extract_tracking_features.py $TRACKING_OUTPUT_DIR $TRACKING_FEATURES_OUTPUT_DIR --device cuda
```

**6. Analyze Crowd Behavior**
```bash
sbatch scripts/crowd_behavior_analysis.sh
# Or run directly:
# python src/tracking/crowd_behavior_analysis.py $TRACKING_OUTPUT_DIR $CROWD_BEHAVIOR_OUTPUT_DIR
```

**7. Encode Statistical Features**
```bash
sbatch scripts/crowd_statistical_encoder.sh
# Or run directly:
# python src/tracking/crowd_statistical_encoder.py $CROWD_BEHAVIOR_OUTPUT_DIR $CROWD_STATISTICAL_OUTPUT_DIR
```

**8. Extract Visual Features**
```bash
sbatch scripts/extract_visual_features.sh
# Or run directly:
# python src/visual_head/extract_visual_features.py $VISUAL_INPUT_DIR $VISUAL_FEATURES_OUTPUT_DIR --device cuda
```

### Training + Evaluation (Steps 9-10)

**9. Train Multi-task Model**
```bash
sbatch scripts/soccer_training_final.sh
# Or run directly:
# python src/multitask_model/soccer_training_final.py $VISUAL_FEATURES_OUTPUT_DIR $CROWD_STATISTICAL_OUTPUT_DIR $TRAINING_OUTPUT_DIR
```

**10. Test Similarity Retrieval**
```bash
sbatch scripts/test_similarity_retrieval.sh
# Or run directly:
# python src/multitask_model/test_similarity_retrieval.py $TRAINING_OUTPUT_DIR/simple_soccer_model.pth $VISUAL_FEATURES_OUTPUT_DIR $CROWD_STATISTICAL_OUTPUT_DIR $TESTING_OUTPUT_DIR
```

## Runtime Prediction Pipeline

Alternatively, you can run the optimized end-to-end pipeline that takes any MKV clip as input and outputs goal probability alongside similar clips using the precomputed model (`simple_soccer_model.pth` at project root):

**Prerequisites:** Complete training pipeline first (steps 1-9)

**1. Precompute Embeddings (one-time setup)**
```bash
sbatch scripts/precompute_embeddings.sh
# Or run directly:
# python src/full_pipeline/precompute_embeddings.py $TRAINING_OUTPUT_DIR/simple_soccer_model.pth $VISUAL_FEATURES_OUTPUT_DIR $CROWD_STATISTICAL_OUTPUT_DIR $EMBEDDINGS_OUTPUT_DIR
```

**2. Run Prediction**
```bash
# Random test clip:
sbatch scripts/run_predict_runtime.sh

# Specific input video:
sbatch scripts/run_predict_runtime.sh /path/to/specific/video.mkv

# Or run directly:
# python src/full_pipeline/predict_runtime.py \
#     $TRAINING_OUTPUT_DIR/simple_soccer_model.pth \
#     $EMBEDDINGS_OUTPUT_DIR \
#     $ORGANIZED_CLIPS_DIR \
#     $PROJECT_ROOT/src/visual_head/extract_visual_features.py \
#     $BYTETRACK_HOME \
#     $PREDICTION_OUTPUT_DIR/prediction_results_$(date +%Y%m%d_%H%M%S) \
#     --tracking_env $TRACKING_ENV_NAME \
#     --visual_env $VISUAL_ENV_NAME
```

## Output

The prediction pipeline generates:
- `prediction_summary.txt` - Human-readable analysis results
- `prediction_report.json` - Detailed JSON report with all metrics
- `similar_clips/` - Directory containing downloaded similar video clips
- Goal probability prediction and model attention weights

## Notes

- All the scripts automatically handle GPU allocation when available
- The test split is excluded from the similarity search to avoid data leakage
- The prediction results include both model-based and similarity-based goal probability estimates
