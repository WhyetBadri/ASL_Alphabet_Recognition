"""
ASL ALPHABET WEBCAM SAMPLE COLLECTOR
=====================================
This script captures hand gesture images from your webcam.
You'll collect 150 images for each of the 26 ASL letters (A-Z).
Total: 3,900 images will be captured.
"""
import cv2
from pathlib import Path
#-------------------------------------------
# CONFIGURATION - Change these if needed
#-------------------------------------------

# Where to save all the captured images
SAMPLE_STORAGE_PATH = './hand_gesture_samples'

# How many different letters to capture (26 for A-Z)
ALPHABET_SIZE = 26

# How many images to capture per letter
# More images = better accuracy, but takes longer
SAMPLES_PER_LETTER = 150

# Which camera to use (0 = built-in, 1 or 2 = external)
# If your camera doesn't work, try changing this to 1 or 2
WEBCAM_ID = 0

# Window title
WINDOW_TITLE = 'ASL Alphabet Sample Collection'

#--------------------------------------------
# FUNCTION 1: Create folders to store images
#--------------------------------------------

def build_storage_structure() -> None:
    """
    Creates folders to organize images by letter.
    
    Creates this structure:
    hand_gesture_samples/
        ├── 0/    (will store images of letter A)
        ├── 1/    (will store images of letter B)
        ├── 2/    (will store images of letter C)
        ...
        └── 25/   (will store images of letter Z)
    """

    # Create main folder
    Path(SAMPLE_STORAGE_PATH).mkdir(parents=True, exist_ok=True)

    # Create 26 Subfolders (one for each letter)
    for letter_index in range(ALPHABET_SIZE):
        letter_folder = Path(SAMPLE_STORAGE_PATH) / str(letter_index)
        letter_folder.mkdir(exist_ok=True)

#--------------------------------------------------
# FUNCTION 2: Show waiting screen before capturing
#--------------------------------------------------
def show_preparation_screen(video_stream, current_letter_num:int) -> bool:
    """
    Shows live video and waits for you to press 'Q' to start capturing.
    
    Parameters:
        video_stream: The webcam connection
        current_letter_num: Which letter we're about to capture (0-25)
    
    Returns:
        True if you pressed Q to start
        False if you closed the window or pressed ESC to cancel
    """
    # Figure out where to save images for this letter
    save_location = Path(SAMPLE_STORAGE_PATH) / str(current_letter_num)

    # Convert number to letter for display
    letter_char = chr(65 + current_letter_num)

    # Keep Showing video until user presses Q
    while True:
        # Get one frame from webcam
        frame_captured, current_frame = video_stream.read()

        # Check if frame was captured successfully
        if not frame_captured:
            print("ERROR: Cannot read from camera!")
            return False
        
        # Create instruction message 
        instruction_message = f'Get ready to show letter "{letter_char}" - Press Q to start!'

        # Draw text on video frame
        cv2.putText(
            current_frame,              # The image to draw on
            instruction_message,        # The text to show
            (30,60),                    # Position (x = 30 px, y = 60 px from top-left)
            cv2.FONT_HERSHEY_DUPLEX,    # Font Style
            0.7,                        # Font size
            (0,255,0),                  # Color in RGB (green)
            2,                          # Text thickness
            cv2.LINE_AA                 # Anti-aliasing (Smoother textures)
        )

        # Show the frame in a window
        cv2.imshow(WINDOW_TITLE, current_frame)


        # Wait 25 milliseconds and check for key press
        pressed_key = cv2.waitKey(25)

        # Check which key was pressed
        if pressed_key == ord('q') or pressed_key == ord('Q'):
            return True # User pressed q or Q, ready to capture
        elif pressed_key == 27: # Esc key
            return False        # User wants to cancel
                
#---------------------------------------------------
# FUNCTION 3: Capture and save images for one letter
#---------------------------------------------------

def record_gesture_samples(video_stream, current_letter_num: int) -> bool:
    """
    Automatically captures 150 images of your hand gesture.
    Saves each image as a JPG file.
    
    Parameters:
        video_stream: The webcam connection
        current_letter_num: Which letter we're capturing (0-25)
    
    Returns:
        True if all images were captured successfully
        False if something went wrong
    """
    # Figure out where to save images for this letter
    save_location = Path(SAMPLE_STORAGE_PATH) / str(current_letter_num)
    
    # Counter to track how many images we've captured
    samples_captured = 0
    
    # Convert number to letter for display
    letter_char = chr(65 + current_letter_num)
    
    # Print status to terminal
    print(f"\n📸 Now capturing {SAMPLES_PER_LETTER} samples for letter '{letter_char}'...")
    
    # Keep capturing until we have enough images
    while samples_captured < SAMPLES_PER_LETTER:
        # Get one frame from webcam
        frame_captured, current_frame = video_stream.read()
        
        # Check if frame was captured
        if not frame_captured:
            print("ERROR: Failed to capture frame!")
            return False
        
        # Create progress message
        progress_message = f'Recording: {samples_captured + 1}/{SAMPLES_PER_LETTER} - Letter {letter_char}'
        
        # Draw progress on video
        cv2.putText(
            current_frame,
            progress_message,
            (30, 60),
            cv2.FONT_HERSHEY_DUPLEX,
            0.9,
            (0, 0, 255),  # Red color (BGR format)
            2,
            cv2.LINE_AA
        )
        
        # Show the frame
        cv2.imshow(WINDOW_TITLE, current_frame)
        cv2.waitKey(25)  # Short delay
        
        # Save this frame as an image file
        image_filename = save_location / f'{samples_captured}.jpg'
        cv2.imwrite(str(image_filename), current_frame)
        
        # Increment counter
        samples_captured += 1
        
        # Print progress update every 30 images
        if samples_captured % 30 == 0:
            print(f"  ✓ Progress: {samples_captured}/{SAMPLES_PER_LETTER} images saved")
    
    # All images captured successfully
    print(f"  ✅ Completed letter '{letter_char}'!\n")
    return True

#------------------------------------------------------
# FUNCTION 4: Main program that coordinates everything
#------------------------------------------------------

def execute_collection_workflow():
    """
    Main function that runs the entire data collection process.
    
    Steps:
    1. Print welcome message
    2. Create folders for storing images
    3. Connect to webcam
    4. For each letter A through Z:
       - Show preparation screen
       - Wait for you to press Q
       - Capture 150 images automatically
       - Save all images
    5. Close webcam and finish
    """
    # Print welcome header
    print("=" * 70)
    print("ASL ALPHABET TRAINING DATA COLLECTION")
    print("=" * 70)
    print(f"\nYou will capture {SAMPLES_PER_LETTER} images for {ALPHABET_SIZE} letters (A-Z).")
    print(f"Total images to capture: {SAMPLES_PER_LETTER * ALPHABET_SIZE}")
    print("\nIMPORTANT TIPS:")
    print("  • Find a spot with good lighting")
    print("  • Keep your hand clearly visible")
    print("  • Vary hand position slightly during each letter's capture")
    print("  • Take breaks between letters to rest your hand")
    print()
    
    # Step 1: Create all folders
    build_storage_structure()
    
    # Step 2: Connect to webcam
    webcam_stream = cv2.VideoCapture(WEBCAM_ID)
    
    # Check if webcam opened successfully
    if not webcam_stream.isOpened():
        print(f"❌ ERROR: Cannot open camera at index {WEBCAM_ID}")
        print("\nTroubleshooting:")
        print("  • Make sure your camera is connected")
        print("  • Try changing WEBCAM_ID to 0, 1, or 2")
        print("  • Close other programs that might be using the camera")
        return
    
    print(f"✓ Camera connected successfully (Device ID: {WEBCAM_ID})\n")
    
    # Step 3: Collect data for all 26 letters
    try:
        for letter_index in range(ALPHABET_SIZE):
            # Convert index to letter
            letter_name = chr(65 + letter_index)
            
            # Print header for this letter
            print(f"\n{'=' * 70}")
            print(f"LETTER {letter_name} (#{letter_index + 1} of {ALPHABET_SIZE})")
            print(f"{'=' * 70}")
            
            # Show preparation screen
            user_ready = show_preparation_screen(webcam_stream, letter_index)
            
            # Check if user cancelled
            if not user_ready:
                print("\n⚠️  Collection cancelled by user")
                break
            
            # Start capturing images
            capture_successful = record_gesture_samples(webcam_stream, letter_index)
            
            # Check if capture was successful
            if not capture_successful:
                print("\n❌ ERROR: Capture failed!")
                break
    
    finally:
        # ALWAYS close the webcam, even if error occurred
        webcam_stream.release()
        cv2.destroyAllWindows()
    
    # Print completion message
    print("\n" + "=" * 70)
    print("✅ DATA COLLECTION COMPLETE!")
    print("=" * 70)
    print(f"Images saved to: {Path(SAMPLE_STORAGE_PATH).absolute()}")
    print("\nNext step: Run feature_extractor.py")
    print("=" * 70)

#-----------------------------------
# PROGRAM START POINT
#-----------------------------------

if __name__ == "__main__":
    # This runs when you execute this script
    execute_collection_workflow()






