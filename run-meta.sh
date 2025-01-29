#!/bin/bash

# File to keep track of progress
PROGRESS_FILE="meta_clip_progress.txt"

# Total number of images (you may need to adjust this)
TOTAL_IMAGES=514

# Function to run the Python script
run_python_script() {
    python3 code/META-CLIP.py
}

# Initialize progress file if it doesn't exist
if [ ! -f "$PROGRESS_FILE" ]; then
    echo "0" > "$PROGRESS_FILE"
fi

# Main loop
while true; do
    # Read current progress
    CURRENT_PROGRESS=$(cat "$PROGRESS_FILE")

    # Check if processing is complete
    if [ "$CURRENT_PROGRESS" -ge "$TOTAL_IMAGES" ]; then
        echo "All images have been processed. Exiting."
        break
    fi

    echo "Starting batch processing from image $CURRENT_PROGRESS"
    
    # Run the Python script
    run_python_script

    # Check if the script ran successfully
    if [ $? -ne 0 ]; then
        echo "Error occurred while running the Python script. Exiting."
        exit 1
    fi

    echo "Batch processing complete."

    # Short pause between runs (optional)
    sleep 2
done

echo "META-CLIP processing completed successfully."