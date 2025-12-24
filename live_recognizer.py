"""
ASL ALPHABET REAL-TIME RECOGNITION SYSTEM
==========================================
Uses webcam to recognize ASL letters in real-time.
Shows predicted letter with confidence level.
Compatible with MediaPipe 0.10.x and Python 3.11.9
"""

import pickle
from pathlib import Path
from typing import List, Dict

import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe modules
hands = mp.solutions.hands
drawing_utils = mp.solutions.drawing_utils
drawing_styles = mp.solutions.drawing_styles

#--------------------------------------------------
# CONFIGURATION
#--------------------------------------------------

MODEL_FILEPATH = 'asl_forest_model.pkl'
WEBCAM_ID = 0
DETECTION_THRESHOLD = 0.6
APPLICATION_TITLE = 'ASL Alphabet Recognizer'

# Colors (BGR format)
LIME_GREEN = (0, 255, 0)
BRIGHT_RED = (0, 0, 255)
PURE_WHITE = (255, 255, 255)
DARK_GRAY = (50, 50, 50)
CYAN = (255, 255, 0)


#--------------------------------------------------
# INITIALIZE MEDIAPIPE
#--------------------------------------------------

hand_detector = hands.Hands(
    static_image_mode=False,
    min_detection_confidence=DETECTION_THRESHOLD,
    max_num_hands=1,
    min_tracking_confidence=0.5
)


#--------------------------------------------------
# FUNCTION 1: Load model
#--------------------------------------------------

def load_classification_model():
    """
    Loads trained model from file.
    
    Returns:
        Dictionary containing model and metadata, or None if failed
    """
    model_path = Path(MODEL_FILEPATH)
    
    if not model_path.exists():
        print(f"[ERROR] Cannot find {MODEL_FILEPATH}")
        print("   Run forest_trainer.py first!")
        return None
    
    try:
        with open(MODEL_FILEPATH, 'rb') as model_file:
            model_data = pickle.load(model_file)
        
        classifier = model_data['forest_model']
        accuracy = model_data.get('validation_accuracy', 'Unknown')
        
        print("=" * 70)
        print("[SUCCESS] MODEL LOADED")
        print("=" * 70)
        print(f"File: {model_path.absolute()}")
        print(f"Size: {model_path.stat().st_size / 1024:.1f} KB")
        
        if isinstance(accuracy, float):
            print(f"Accuracy: {accuracy * 100:.2f}%")
        
        letter_mapping = model_data.get('label_to_letter', {})
        if letter_mapping:
            letters = sorted([letter_mapping[k] for k in sorted(letter_mapping.keys())])
            print(f"Recognizes: {', '.join(letters)}")
        
        print("=" * 70)
        
        return model_data
        
    except Exception as error:
        print(f"[ERROR] Loading model: {error}")
        return None


#--------------------------------------------------
# FUNCTION 2: Extract hand coordinates
#--------------------------------------------------

def extract_hand_coordinates(hand_landmarks) -> List[float]:
    """
    Extracts 42 normalized coordinates from hand landmarks.
    
    Process:
    1. Get all 21 landmark (x, y) coordinates
    2. Find minimum x and y
    3. Subtract minimum from all coordinates
    4. Return 42 normalized numbers
    
    Returns:
        List of 42 floats
    """
    x_values = []
    y_values = []
    
    # Collect coordinates
    for landmark_point in hand_landmarks.landmark:
        x_values.append(landmark_point.x)
        y_values.append(landmark_point.y)
    
    # Find minimums
    min_x_coord = min(x_values)
    min_y_coord = min(y_values)
    
    # Normalize
    normalized_features = []
    
    for landmark_point in hand_landmarks.landmark:
        normalized_features.append(landmark_point.x - min_x_coord)
        normalized_features.append(landmark_point.y - min_y_coord)
    
    return normalized_features


#--------------------------------------------------
# FUNCTION 3: Calculate bounding box
#--------------------------------------------------

def compute_hand_bounding_box(hand_landmarks, frame_width: int, frame_height: int) -> tuple:
    """
    Calculates rectangle around hand.
    
    Returns:
        (x1, y1, x2, y2) coordinates for bounding box
    """
    x_coords = []
    y_coords = []
    
    for landmark in hand_landmarks.landmark:
        x_coords.append(landmark.x)
        y_coords.append(landmark.y)
    
    # Convert to pixels and add padding
    box_x1 = int(min(x_coords) * frame_width) - 15
    box_y1 = int(min(y_coords) * frame_height) - 15
    box_x2 = int(max(x_coords) * frame_width) + 15
    box_y2 = int(max(y_coords) * frame_height) + 15
    
    # Keep within frame
    box_x1 = max(0, box_x1)
    box_y1 = max(0, box_y1)
    box_x2 = min(frame_width, box_x2)
    box_y2 = min(frame_height, box_y2)
    
    return box_x1, box_y1, box_x2, box_y2


#--------------------------------------------------
# FUNCTION 4: Draw hand skeleton
#--------------------------------------------------

def render_hand_skeleton(frame, hand_landmarks) -> None:
    """
    Draws hand landmarks and connections on frame.
    """
    drawing_utils.draw_landmarks(
        frame,
        hand_landmarks,
        hands.HAND_CONNECTIONS,
        drawing_styles.get_default_hand_landmarks_style(),
        drawing_styles.get_default_hand_connections_style()
    )


#--------------------------------------------------
# FUNCTION 5: Draw prediction
#--------------------------------------------------

def render_prediction_display(frame, x1: int, y1: int, x2: int, y2: int, 
                              predicted_letter: str, confidence: float) -> None:
    """
    Draws bounding box, letter, and confidence meter.
    """
    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), LIME_GREEN, 3)
    
    # Prepare text
    letter_font = cv2.FONT_HERSHEY_DUPLEX
    letter_scale = 2.0
    letter_thickness = 3
    
    (text_width, text_height), baseline = cv2.getTextSize(
        predicted_letter, letter_font, letter_scale, letter_thickness
    )
    
    # Draw background for text
    padding = 10
    bg_x1 = x1
    bg_y1 = y1 - text_height - baseline - (2 * padding)
    bg_x2 = x1 + text_width + (2 * padding)
    bg_y2 = y1
    
    # Adjust if off screen
    if bg_y1 < 0:
        bg_y1 = y2
        bg_y2 = y2 + text_height + baseline + (2 * padding)
        text_y = bg_y2 - padding - baseline
    else:
        text_y = y1 - padding - baseline
    
    # Draw filled rectangle
    cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), LIME_GREEN, -1)
    
    # Draw letter
    cv2.putText(
        frame, predicted_letter, (x1 + padding, text_y),
        letter_font, letter_scale, DARK_GRAY, letter_thickness, cv2.LINE_AA
    )
    
    # Draw confidence meter
    meter_width = x2 - x1
    meter_height = 10
    meter_x = x1
    meter_y = y2 + 5
    
    # Background bar
    cv2.rectangle(
        frame, (meter_x, meter_y),
        (meter_x + meter_width, meter_y + meter_height),
        (100, 100, 100), -1
    )
    
    # Filled portion
    confidence_width = int((confidence / 100.0) * meter_width)
    
    # Color based on confidence
    if confidence >= 80:
        meter_color = LIME_GREEN
    elif confidence >= 50:
        meter_color = CYAN
    else:
        meter_color = BRIGHT_RED
    
    cv2.rectangle(
        frame, (meter_x, meter_y),
        (meter_x + confidence_width, meter_y + meter_height),
        meter_color, -1
    )
    
    # Confidence text
    confidence_text = f"{confidence:.1f}%"
    cv2.putText(
        frame, confidence_text,
        (meter_x, meter_y + meter_height + 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, PURE_WHITE, 1, cv2.LINE_AA
    )


#--------------------------------------------------
# FUNCTION 6: Draw instructions
#--------------------------------------------------

def render_user_instructions(frame) -> None:
    """
    Displays usage instructions on frame.
    """
    instructions = [
        "Show ASL hand signs",
        "Press ESC to exit"
    ]
    
    y_offset = 30
    for instruction in instructions:
        cv2.putText(
            frame, instruction, (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, PURE_WHITE, 2, cv2.LINE_AA
        )
        y_offset += 30


#--------------------------------------------------
# FUNCTION 7: Main recognition loop
#--------------------------------------------------

def execute_realtime_recognition(model_package: Dict) -> None:
    """
    Main loop for real-time recognition.
    
    Process (runs continuously):
    1. Capture frame from webcam
    2. Detect hand with MediaPipe
    3. Extract features
    4. Predict letter with model
    5. Display results
    6. Repeat
    """
    classifier_model = model_package['forest_model']
    letter_mapping = model_package.get('label_to_letter', {})
    
    # Initialize webcam
    webcam_stream = cv2.VideoCapture(WEBCAM_ID)
    
    if not webcam_stream.isOpened():
        print(f"[ERROR] Cannot open camera {WEBCAM_ID}")
        print("   Try changing WEBCAM_ID to 0, 1, or 2")
        return
    
    # Set webcam properties
    webcam_stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    webcam_stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    webcam_stream.set(cv2.CAP_PROP_FPS, 30)
    
    print("\n[CAMERA] Starting real-time recognition...")
    print("=" * 70)
    print("Show your ASL signs to the camera!")
    print("Press ESC to exit")
    print("=" * 70)
    
    frame_counter = 0
    
    # MAIN LOOP
    while True:
        # Capture frame
        frame_success, current_frame = webcam_stream.read()
        
        if not frame_success:
            print("[WARNING] Failed to capture frame")
            break
        
        frame_counter += 1
        frame_height, frame_width, _ = current_frame.shape
        
        # Convert to RGB
        frame_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
        
        # Detect hands
        detection_output = hand_detector.process(frame_rgb)
        
        # If hand detected
        if detection_output.multi_hand_landmarks:
            for hand_landmarks in detection_output.multi_hand_landmarks:
                # Draw skeleton
                render_hand_skeleton(current_frame, hand_landmarks)
                
                # Extract features
                feature_vector = extract_hand_coordinates(hand_landmarks)
                
                # Predict
                prediction = classifier_model.predict([np.asarray(feature_vector)])
                prediction_probs = classifier_model.predict_proba([np.asarray(feature_vector)])
                
                predicted_class = int(prediction[0])
                confidence = np.max(prediction_probs) * 100
                
                predicted_letter = letter_mapping.get(predicted_class, '?')
                
                # Calculate box
                box_x1, box_y1, box_x2, box_y2 = compute_hand_bounding_box(
                    hand_landmarks, frame_width, frame_height
                )
                
                # Draw prediction
                render_prediction_display(
                    current_frame, box_x1, box_y1, box_x2, box_y2,
                    predicted_letter, confidence
                )
        
        # Draw instructions
        render_user_instructions(current_frame)
        
        # Show frame
        cv2.imshow(APPLICATION_TITLE, current_frame)
        
        # Check for ESC
        if cv2.waitKey(1) == 27:
            print("\n[EXIT] Exiting...")
            break
    
    # Cleanup
    webcam_stream.release()
    cv2.destroyAllWindows()
    
    print(f"\n[SUCCESS] Processed {frame_counter} frames")
    print("Thank you for using ASL Recognizer!")


#--------------------------------------------------
# FUNCTION 8: Main program
#--------------------------------------------------

def launch_recognition_system():
    """
    Entry point that starts recognition system.
    """
    print("\n" + "=" * 70)
    print("ASL ALPHABET REAL-TIME RECOGNIZER")
    print("=" * 70)
    
    # Load model
    model_data = load_classification_model()
    
    if model_data is None:
        print("\n[ERROR] Cannot start without trained model")
        print("   Run forest_trainer.py first!")
        return
    
    # Start recognition
    execute_realtime_recognition(model_data)


#--------------------------------------------------
# PROGRAM START
#--------------------------------------------------

if __name__ == "__main__":
    launch_recognition_system()