"""
ASL HAND LANDMARK FEATURE EXTRACTOR
====================================
Processes captured images and extracts hand features using MediaPipe AI.
Converts 3,900 images into numerical features for machine learning.
Compatible with MediaPipe 0.10.x
"""

import pickle
from pathlib import Path
from typing import List, Tuple

import cv2
import mediapipe as mp

# Then use:
hands = mp.solutions.hands
drawing_utils = mp.solutions.drawing_utils
drawing_styles = mp.solutions.drawing_styles

#--------------------------------------------------
# CONFIGURATION
#--------------------------------------------------

SOURCE_IMAGE_PATH = './hand_gesture_samples'
FEATURE_OUTPUT_FILE = 'gesture_features.pickle'
DETECTION_THRESHOLD = 0.6


#--------------------------------------------------
# INITIALIZE MEDIAPIPE
#--------------------------------------------------

# MediaPipe: Google's AI for detecting hands
# Finds 21 keypoints on each hand (fingers, joints, wrist)
hand_processor = hands.Hands(
    static_image_mode=True,
    min_detection_confidence=DETECTION_THRESHOLD,
    max_num_hands=1
)


#--------------------------------------------------
# FUNCTION 1: Extract features from one image
#--------------------------------------------------

def compute_normalized_keypoints(image_filepath: Path) -> List[float]:
    """
    Extracts 42 numbers representing hand shape from one image.
    
    Process:
    1. Load image
    2. Use MediaPipe AI to find 21 hand points
    3. Normalize coordinates (make them relative)
    4. Return 42 numbers
    
    Returns:
        List of 42 numbers, or empty list if no hand found
    """
    # Load image from disk
    pixel_data = cv2.imread(str(image_filepath))
    
    if pixel_data is None:
        return []
    
    # Convert BGR to RGB (OpenCV uses BGR, MediaPipe needs RGB)
    rgb_pixel_data = cv2.cvtColor(pixel_data, cv2.COLOR_BGR2RGB)
    
    # Use AI to detect hand
    landmark_results = hand_processor.process(rgb_pixel_data)
    
    # Check if hand was found
    if not landmark_results.multi_hand_landmarks:
        return []
    
    # Get the detected hand
    detected_hand = landmark_results.multi_hand_landmarks[0]
    
    # Collect all x and y coordinates
    all_x_coords = []
    all_y_coords = []
    
    for single_landmark in detected_hand.landmark:
        all_x_coords.append(single_landmark.x)
        all_y_coords.append(single_landmark.y)
    
    # Normalize: subtract minimum values
    # This makes hand position irrelevant (only shape matters)
    minimum_x = min(all_x_coords)
    minimum_y = min(all_y_coords)
    
    # Create feature list
    normalized_coords = []
    
    for single_landmark in detected_hand.landmark:
        normalized_coords.append(single_landmark.x - minimum_x)
        normalized_coords.append(single_landmark.y - minimum_y)
    
    return normalized_coords


#--------------------------------------------------
# FUNCTION 2: Process all images
#--------------------------------------------------

def scan_all_gesture_images() -> Tuple[List[List[float]], List[str]]:
    """
    Processes all 3,900 images and extracts features from each.
    
    Returns:
        Two lists:
        - feature_list: Each item is a list of 42 numbers
        - label_list: Each item is a folder name ('0', '1', '2', etc.)
    """
    feature_list = []
    label_list = []
    
    samples_directory = Path(SOURCE_IMAGE_PATH)
    
    if not samples_directory.exists():
        print(f"❌ ERROR: Cannot find {SOURCE_IMAGE_PATH}")
        print("   Run webcam_collector.py first!")
        return [], []
    
    # Find all letter folders (0, 1, 2, ..., 25)
    letter_folders = sorted([
        folder for folder in samples_directory.iterdir()
        if folder.is_dir()
    ])
    
    if not letter_folders:
        print(f"❌ ERROR: No letter folders found!")
        return [], []
    
    print("\n" + "=" * 70)
    print("PROCESSING HAND GESTURES")
    print("=" * 70)
    
    # Counters for statistics
    total_images_scanned = 0        # Total images found
    total_features_extracted = 0    # Total successful feature extractions
    total_failures = 0              # Total failed extractions
    
    # Process each letter folder
    for letter_folder in letter_folders:
        folder_label = letter_folder.name

        # Convert numeric label to letter (0 → A, 1 → B, etc.)
        letter_character = chr(65 + int(folder_label)) if folder_label.isdigit() else folder_label
        
        print(f"\n📂 Processing Letter '{letter_character}' (folder {folder_label})...")
        
        # Find all JPG images
        image_files = list(letter_folder.glob('*.jpg'))
        images_in_folder = len(image_files)
        successful_extractions = 0
        
        # Process each image
        for image_path in image_files:
            extracted_features = compute_normalized_keypoints(image_path)
            
            if extracted_features:
                feature_list.append(extracted_features)  # Save features
                label_list.append(folder_label)          # Save corresponding label
                successful_extractions += 1
            else:
                total_failures += 1
        
        # Update global counters
        total_images_scanned += images_in_folder
        total_features_extracted += successful_extractions
        
         # Calculate success rate for this folder
        success_rate = (successful_extractions / images_in_folder * 100) if images_in_folder > 0 else 0
        print(f"  ✓ Extracted: {successful_extractions}/{images_in_folder} ({success_rate:.1f}%)")
    
    # Print summary
    print("\n" + "=" * 70)
    print("EXTRACTION SUMMARY")
    print("=" * 70)
    print(f"Total images: {total_images_scanned}")
    print(f"Successful: {total_features_extracted}")
    print(f"Failed: {total_failures}")
    print(f"Success rate: {(total_features_extracted/total_images_scanned*100):.1f}%")
    
    return feature_list, label_list


#--------------------------------------------------
# FUNCTION 3: Save features to file
#--------------------------------------------------

def persist_extracted_features(feature_list: List[List[float]], label_list: List[str]) -> None:
    """
    Saves extracted features to a file using pickle.
    
    Pickle: Python's way of saving complex data structures to disk.
    Creates a binary file containing all features and labels.
    """
    if not feature_list:
        print("\n❌ ERROR: No features to save!")
        return
    
    # Package data into dictionary
    dataset_package = {
        'keypoint_data': feature_list,
        'letter_labels': label_list
    }
    
    # Save using pickle
    with open(FEATURE_OUTPUT_FILE, 'wb') as output_file:
        pickle.dump(dataset_package, output_file)
    
    # Print success
    print(f"\n✅ Features saved!")
    print("=" * 70)
    print(f"Output file: {Path(FEATURE_OUTPUT_FILE).absolute()}")
    print(f"Total samples: {len(feature_list)}")
    print(f"Features per sample: {len(feature_list[0])}")
    print(f"Unique letters: {len(set(label_list))}")
    print(f"\nNext: Run forest_trainer.py")
    print("=" * 70)


#--------------------------------------------------
# FUNCTION 4: Main program
#--------------------------------------------------

def execute_extraction_pipeline():
    """Runs the entire extraction process."""
    print("Starting feature extraction...")
    print("This takes 5-10 minutes.\n")
    
    # Process all images
    feature_data, label_data = scan_all_gesture_images()
    
    # Save if successful
    if feature_data and label_data:
        persist_extracted_features(feature_data, label_data)
        print("\n🎉 Extraction complete!")
    else:
        print("\n❌ No features extracted.")
        print("Check that images exist and hands are visible.")


#--------------------------------------------------
# PROGRAM START
#--------------------------------------------------

if __name__ == "__main__":
    execute_extraction_pipeline()