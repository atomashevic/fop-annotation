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
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Configuration
OUTPUT_DIR = 'results/fer_analysis_debug'
TEMP_DIR = 'temp_debug'
DATA_DIR = 'data/mafw'  # Adjust this to your MAFW dataset location
FRAME_TARGETS = [50, 40, 30, 20, 15]  # Target frame counts to try
MAX_FRAMES = 50  # Strict maximum number of frames

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Initialize FER detector
detector = FER(mtcnn=True)
logging.info("Initialized FER detector with MTCNN")

def cut_video(video, fr, to):
    """Cut video between specified timestamps."""
    logging.debug(f"Attempting to cut video from {fr} to {to}")
    output_path = os.path.join(TEMP_DIR, 'temp_video_edit.mp4')
    video_edit = VideoFileClip(video).subclip(fr, to)
    video_edit.write_videofile(output_path, preset='ultrafast', audio=False)
    logging.debug(f"Video cut successfully, saved to {output_path}")
    return output_path

def get_video_info(video_path):
    """Get video information including frame count and fps."""
    logging.debug(f"Getting video info for: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps
    
    logging.debug(f"Video info - FPS: {fps}, Frames: {frame_count}, Duration: {duration:.2f}s")
    
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
    logging.debug(f"Determining frame extraction parameters for {frame_count} frames")
    
    # Calculate frequency needed to get exactly MAX_FRAMES
    if frame_count > MAX_FRAMES:
        freq = math.ceil(frame_count/MAX_FRAMES)  # Use ceil to ensure we don't exceed MAX_FRAMES
        actual_frames = math.floor(frame_count/freq)
        logging.debug(f"Using frequency {freq} to get {actual_frames} frames (max {MAX_FRAMES})")
        return freq, actual_frames, MAX_FRAMES
    
    # For shorter videos, try to match one of our target frames
    for target_frames in FRAME_TARGETS:
        if frame_count >= target_frames:
            freq = math.ceil(frame_count/target_frames)  # Use ceil to ensure we don't exceed target
            actual_frames = math.floor(frame_count/freq)
            logging.debug(f"Selected target_frames={target_frames}, freq={freq}, expected_frames={actual_frames}")
            return freq, actual_frames, target_frames
    
    # If video is very short, just use all frames
    logging.debug(f"Video too short, using all {frame_count} frames")
    return 1, frame_count, frame_count

def detect_emotions(video_path, video_id):
    """Detect emotions in video frames using adaptive frame extraction."""
    logging.info(f"\nProcessing video {video_id}: {video_path}")
    
    try:
        video = Video(video_path)
        video.first_face_only = False
        
        # Get video information
        fps, frame_count, duration = get_video_info(video_path)
        freq, actual_frames, target_frames = determine_frame_extraction_params(frame_count)
        
        logging.info(f"Video stats for {video_id}:")
        logging.info(f"Duration: {duration:.2f}s, Total frames: {frame_count}, FPS: {fps:.2f}")
        logging.info(f"Target frames: {target_frames}, Sampling frequency: {freq}, Expected frames: {actual_frames}")
        
        # Analyze video
        logging.info("Starting emotion detection...")
        raw_data = video.analyze(detector, display=False, frequency=freq,
                               save_video=False, annotate_frames=False, zip_images=False)
        logging.info(f"Emotion detection complete. Detected {len(raw_data)} frames")
        
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
        
        # Log emotion statistics
        emotion_cols = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        for emotion in emotion_cols:
            if emotion in df.columns:
                mean_val = df[emotion].mean()
                logging.info(f"Average {emotion}: {mean_val:.3f}")
        
        df.to_csv(output_path)
        logging.info(f"Results saved to {output_path}")
        
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
        logging.error(f"Error in detect_emotions: {str(e)}", exc_info=True)
        raise

def main():
    try:
        # Check GPU availability
        device_name = tf.test.gpu_device_name()
        if device_name != '/device:GPU:0':
            logging.warning('GPU device not found, processing may be slower')
        else:
            logging.info(f'Found GPU at: {device_name}')
        
        # Get list of all MAFW videos
        all_videos = []
        for root, _, files in os.walk(DATA_DIR):
            for file in files:
                if file.endswith(('.mp4', '.avi', '.mov')):
                    all_videos.append(os.path.join(root, file))
        
        if not all_videos:
            raise ValueError(f"No video files found in {DATA_DIR}")
        
        logging.info(f"Found {len(all_videos)} videos in total")
        
        # Select one random video
        video_path = random.choice(all_videos)
        logging.info(f"Selected video for processing: {video_path}")
        
        # Clean temporary directories
        os.system('rm -rf output/')
        os.system(f'rm -rf {TEMP_DIR}/*')
        
        # Process video
        metadata = detect_emotions(video_path, "debug_video")
        
        # Save processing summary
        summary_path = os.path.join(OUTPUT_DIR, 'debug_summary.csv')
        pd.DataFrame([metadata[1]]).to_csv(summary_path, index=False)
        logging.info(f"Debug summary saved to {summary_path}")
        
        logging.info("\nDebug run completed successfully!")
        
    except Exception as e:
        logging.error(f"Error in main: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    logging.info("Starting debug run...")
    main() 