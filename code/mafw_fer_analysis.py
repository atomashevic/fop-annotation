import tensorflow as tf
import os
import pandas as pd
import cv2
import math
import numpy as np
from fer import Video
from fer import FER
from moviepy.editor import VideoFileClip
import random
from tqdm import tqdm
import logging
import sys
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Using INFO level for production
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fer_analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Configuration
OUTPUT_DIR = 'results/fer_analysis'
TEMP_DIR = 'temp'
DATA_DIR = 'data/mafw_sample'  # Using the sample directory
FRAME_TARGETS = [50, 40, 30, 20, 15]  # Target frame counts to try
MAX_FRAMES = 50  # Strict maximum number of frames

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Initialize FER detector
detector = FER(mtcnn=True)

def get_video_info(video_path):
    """Get video information including frame count and fps."""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps
    
    # Verify frame count by manual counting
    manual_count = 0
    while True:
        ret, _ = cap.read()
        if not ret:
            break
        manual_count += 1
    
    if manual_count != frame_count:
        logging.warning(f"Frame count mismatch - Reported: {frame_count}, Actual: {manual_count}")
        frame_count = manual_count
    
    cap.release()
    return fps, frame_count, duration

def determine_frame_extraction_params(frame_count):
    """Determine the optimal number of frames to extract and the frequency."""
    # Calculate frequency needed to get exactly MAX_FRAMES
    if frame_count > MAX_FRAMES:
        freq = math.ceil(frame_count/MAX_FRAMES)  # Use ceil to ensure we don't exceed MAX_FRAMES
        actual_frames = math.floor(frame_count/freq)
        return freq, actual_frames, MAX_FRAMES
    
    # For shorter videos, try to match one of our target frames
    for target_frames in FRAME_TARGETS:
        if frame_count >= target_frames:
            freq = math.ceil(frame_count/target_frames)  # Use ceil to ensure we don't exceed target
            actual_frames = math.floor(frame_count/freq)
            return freq, actual_frames, target_frames
    
    # If video is very short, just use all frames
    return 1, frame_count, frame_count

def detect_emotions(video_path, video_id):
    """Detect emotions in video frames using adaptive frame extraction."""
    try:
        video = Video(video_path)
        video.first_face_only = False
        
        # Get video information
        fps, frame_count, duration = get_video_info(video_path)
        freq, actual_frames, target_frames = determine_frame_extraction_params(frame_count)
        
        logging.info(f"Processing {video_id} - Frames: {frame_count}, Target: {target_frames}, Freq: {freq}")
        
        # Analyze video
        raw_data = video.analyze(detector, display=False, frequency=freq,
                               save_video=False, annotate_frames=False, zip_images=False)
        
        # Save results
        output_path = os.path.join(OUTPUT_DIR, f'{video_id}_f{actual_frames}.csv')
        df = video.to_pandas(raw_data)
        
        # Add metadata columns
        df['video_id'] = video_id
        df['total_frames'] = frame_count
        df['sampling_freq'] = freq
        df['target_frames'] = target_frames
        df['actual_frames'] = actual_frames
        df['fps'] = fps
        df['duration'] = duration
        df['timestamp'] = datetime.now().isoformat()
        
        df.to_csv(output_path)
        
        return df, {
            'video_id': video_id,
            'video_path': video_path,
            'total_frames': frame_count,
            'sampling_freq': freq,
            'target_frames': target_frames,
            'actual_frames': actual_frames,
            'fps': fps,
            'duration': duration,
            'detected_frames': len(raw_data),
            'success': True
        }
    
    except Exception as e:
        logging.error(f"Error processing video {video_id}: {str(e)}")
        return None, {
            'video_id': video_id,
            'video_path': video_path,
            'success': False,
            'error': str(e)
        }

def main():
    # Check GPU availability
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        logging.warning('GPU device not found, processing may be slower')
    else:
        logging.info(f'Found GPU at: {device_name}')
    
    # Get list of all videos in the sample
    all_videos = []
    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith(('.mp4', '.avi', '.mov')):
                all_videos.append(os.path.join(root, file))
    
    if not all_videos:
        raise ValueError(f"No video files found in {DATA_DIR}")
    
    logging.info(f"Found {len(all_videos)} videos to process")
    
    # Process all videos
    results = []
    for i, video_path in enumerate(tqdm(all_videos)):
        video_id = f"video_{i:04d}"
        
        # Clean temporary directories
        os.system('rm -rf output/')
        os.system(f'rm -rf {TEMP_DIR}/*')
        
        # Process video
        _, metadata = detect_emotions(video_path, video_id)
        results.append(metadata)
        
        # Save intermediate results every 100 videos
        if (i + 1) % 100 == 0:
            pd.DataFrame(results).to_csv(
                os.path.join(OUTPUT_DIR, f'processing_summary_checkpoint_{i+1}.csv'),
                index=False
            )
    
    # Save final processing summary
    summary_df = pd.DataFrame(results)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, 'processing_summary.csv'), index=False)
    
    # Print processing statistics
    successful = summary_df['success'].sum()
    logging.info("\nProcessing complete!")
    logging.info(f"Successfully processed: {successful}/{len(results)} videos")
    
    # Print frame extraction statistics for successful videos
    if successful > 0:
        success_df = summary_df[summary_df['success']]
        logging.info("\nFrame extraction statistics:")
        for target in FRAME_TARGETS:
            count = len(success_df[success_df['target_frames'] == target])
            logging.info(f"Videos using {target} target frames: {count}")
        logging.info(f"Videos using all frames: {len(success_df[success_df['target_frames'] == success_df['total_frames']])}")

if __name__ == "__main__":
    main() 