#!/bin/bash

# Check if correct number of arguments are provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <model> <labels>"
    echo "model: CLIP, META-CLIP, or APPLE-CLIP"
    echo "labels: E or NE"
    exit 1
fi

MODEL=$1
LABELS=$2

# Determine which script to run
if [ "$MODEL" = "CLIP" ]; then
    SCRIPT_BASE="CLIP-2"
elif [ "$MODEL" = "META-CLIP" ]; then
    SCRIPT_BASE="META-CLIP"
elif [ "$MODEL" = "APPLE-CLIP" ]; then
    SCRIPT_BASE="apple-CLIP"
else
    echo "Invalid model. Use CLIP, META-CLIP, or APPLE-CLIP."
    exit 1
fi

if [ "$LABELS" = "E" ]; then
    SCRIPT="${SCRIPT_BASE}.py"
elif [ "$LABELS" = "NE" ]; then
    SCRIPT="${SCRIPT_BASE}-NE.py"
else
    echo "Invalid labels. Use E or NE."
    exit 1
fi

# TODO: Progress file is not good, define custom for each model

# 'meta_clip_progress.txt'
# 'clip_progress.txt'
# 'meta_clip_ne_progress.txt'
# 'clip_ne_progress.txt'

# Define progress file based on model and label combination
if [ "$MODEL" = "CLIP" ] && [ "$LABELS" = "E" ]; then
    PROGRESS_FILE="clip_progress.txt"
elif [ "$MODEL" = "CLIP" ] && [ "$LABELS" = "NE" ]; then
    PROGRESS_FILE="clip_ne_progress.txt"
elif [ "$MODEL" = "APPLE-CLIP" ] && [ "$LABELS" = "E" ]; then
    PROGRESS_FILE="apple_clip_progress.txt"
elif [ "$MODEL" = "APPLE-CLIP" ] && [ "$LABELS" = "NE" ]; then
    PROGRESS_FILE="apple_clip_ne_progress.txt"
elif [ "$MODEL" = "META-CLIP" ] && [ "$LABELS" = "E" ]; then
    PROGRESS_FILE="meta_clip_progress.txt"
elif [ "$MODEL" = "META-CLIP" ] && [ "$LABELS" = "NE" ]; then
    PROGRESS_FILE="meta_clip_ne_progress.txt"
else
    echo "Invalid combination of model and labels."
    exit 1
fi

echo "Using progress file: $PROGRESS_FILE"



# Total number of images (you may need to adjust this)
TOTAL_IMAGES=514

# Function to run the Python script
run_python_script() {
    python3 "code/$SCRIPT"
}

# Initialize progress file if it doesn't exist
if [ ! -f "$PROGRESS_FILE" ]; then
    echo "0" > "$PROGRESS_FILE"
fi

# Main loop
while true; do
    CURRENT_PROGRESS=$(cat "$PROGRESS_FILE")
    if [ "$CURRENT_PROGRESS" -ge "$TOTAL_IMAGES" ]; then
        echo "All images have been processed. Exiting."
        break
    fi

    echo "Starting batch processing from image $CURRENT_PROGRESS"
    run_python_script

    if [ $? -ne 0 ]; then
        echo "Error occurred while running the Python script. Exiting."
        exit 1
    fi

    echo "Batch processing complete."
    sleep 2
done

# Delete progress file
if [ -f "$PROGRESS_FILE" ]; then
    rm "$PROGRESS_FILE"
    echo "Progress file $PROGRESS_FILE has been deleted."
else
    echo "Progress file $PROGRESS_FILE does not exist."
fi



