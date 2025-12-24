"""
ASL RANDOM FOREST CLASSIFIER TRAINER
=====================================
Trains an AI model to recognize ASL letters using Random Forest algorithm.
Creates 200 decision trees that vote on predictions.
"""

import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#--------------------------------------------------
# CONFIGURATION
#--------------------------------------------------

FEATURES_INPUT_FILE = 'gesture_features.pickle'
TRAINED_MODEL_FILE = 'asl_forest_model.pkl'
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42
TREE_COUNT = 200

# Map numbers to letters
LETTER_MAPPINGS = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
    9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q',
    17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
}


#--------------------------------------------------
# FUNCTION 1: Load features
#--------------------------------------------------

def import_training_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads features from pickle file.
    
    Returns:
        Two arrays: features (42 numbers each) and labels (letter numbers)
    """
    features_file = Path(FEATURES_INPUT_FILE)
    
    if not features_file.exists():
        print(f"❌ ERROR: Cannot find {FEATURES_INPUT_FILE}")
        print("   Run feature_extractor.py first!")
        return None, None
    
    # Load pickle file
    with open(FEATURES_INPUT_FILE, 'rb') as input_file:
        dataset_dictionary = pickle.load(input_file)
    
    # Convert to numpy arrays (required for scikit-learn)
    feature_array = np.asarray(dataset_dictionary['keypoint_data'])
    label_array = np.asarray(dataset_dictionary['letter_labels'])
    
    # Print info
    print("\n" + "=" * 70)
    print("DATASET LOADED")
    print("=" * 70)
    print(f"Total samples: {len(feature_array)}")
    print(f"Features per sample: {feature_array.shape[1]}")
    print(f"Number of letters: {len(np.unique(label_array))}")
    
    # Show samples per letter
    unique_labels, label_counts = np.unique(label_array, return_counts=True)
    print("\nSamples per letter:")
    for label, count in zip(unique_labels, label_counts):
        letter_char = LETTER_MAPPINGS.get(int(label), label)
        print(f"  {letter_char}: {count}")
    
    return feature_array, label_array


#--------------------------------------------------
# FUNCTION 2: Split data
#--------------------------------------------------

def partition_dataset(feature_array: np.ndarray, label_array: np.ndarray) -> Tuple:
    """
    Splits data into training (80%) and testing (20%) sets.
    
    Why split?
    - Training set: Teaches the model
    - Testing set: Checks if model actually learned (unseen data)
    
    Returns:
        X_train, X_val, y_train, y_val
    """
    X_train, X_val, y_train, y_val = train_test_split(
        feature_array,
        label_array,
        test_size=VALIDATION_SPLIT,
        shuffle=True,
        stratify=label_array,  # Keep same letter proportions
        random_state=RANDOM_SEED
    )
    
    print("\n" + "=" * 70)
    print("DATASET SPLIT")
    print("=" * 70)
    print(f"Training samples: {len(X_train)} (80%)")
    print(f"Testing samples: {len(X_val)} (20%)")
    
    return X_train, X_val, y_train, y_val


#--------------------------------------------------
# FUNCTION 3: Train model
#--------------------------------------------------

def build_and_train_forest(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """
    Creates and trains a Random Forest with 200 trees.
    
    Process:
    1. Creates 200 decision trees
    2. Each tree trains on random subset of data
    3. Each tree learns: "If x > 0.5 and y < 0.3, then probably letter A"
    4. Model combines all trees' knowledge
    
    Returns:
        Trained model
    """
    print("\n" + "=" * 70)
    print("TRAINING MODEL")
    print("=" * 70)
    print(f"Algorithm: Random Forest")
    print(f"Number of trees: {TREE_COUNT}")
    print(f"Training samples: {len(X_train)}")
    print("\nTraining... (1-2 minutes)")
    
    # Create model
    forest_classifier = RandomForestClassifier(
        n_estimators=TREE_COUNT,
        random_state=RANDOM_SEED,
        n_jobs=-1,  # Use all CPU cores
        verbose=1   # Show progress
    )
    
    # Train! This is where AI learns!
    forest_classifier.fit(X_train, y_train)
    
    print("\n✅ Training complete!")
    
    return forest_classifier


#--------------------------------------------------
# FUNCTION 4: Test model
#--------------------------------------------------

def assess_model_performance(classifier, X_val: np.ndarray, y_val: np.ndarray) -> float:
    """
    Tests how well the model learned.
    
    Process:
    1. Give model unseen test data
    2. Model predicts letters
    3. Compare predictions to true answers
    4. Calculate accuracy
    
    Returns:
        Accuracy (0.95 = 95% correct)
    """
    # Make predictions
    predicted_labels = classifier.predict(X_val)
    
    # Calculate accuracy
    overall_accuracy = accuracy_score(y_val, predicted_labels)
    
    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE")
    print("=" * 70)
    print(f"\n🎯 Overall Accuracy: {overall_accuracy * 100:.2f}%")
    print(f"   ({int(overall_accuracy * len(y_val))}/{len(y_val)} correct)")
    
    # Detailed report per letter
    print("\n📊 Performance per Letter:")
    print("-" * 70)
    
    label_names = [LETTER_MAPPINGS[int(label)] for label in sorted(set(y_val))]
    
    print(classification_report(
        y_val,
        predicted_labels,
        target_names=label_names,
        digits=3
    ))
    
    # Confusion matrix (which letters get confused)
    print("\n📢 Confusion Matrix:")
    print("(Rows = True Letter, Columns = Predicted Letter)")
    print()
    
    conf_matrix = confusion_matrix(y_val, predicted_labels)
    unique_labels = sorted(set(y_val))
    
    # Print header
    print(f"{'':>8}", end='')
    for label in unique_labels:
        letter = LETTER_MAPPINGS[int(label)]
        print(f"{letter:>5}", end='')
    print()
    
    print("  " + "-" * (5 * len(unique_labels) + 8))
    
    # Print matrix
    for i, true_label in enumerate(unique_labels):
        true_letter = LETTER_MAPPINGS[int(true_label)]
        print(f"{true_letter:>6} |", end='')
        for j in range(len(unique_labels)):
            value = conf_matrix[i][j]
            if i == j:
                print(f"{value:>4}*", end='')  # Correct
            else:
                print(f"{value:>5}", end='')
        print()
    
    # Most confused pairs
    print("\n⚠️  Most Common Mistakes:")
    confusion_pairs = []
    for i, true_label in enumerate(unique_labels):
        for j, pred_label in enumerate(unique_labels):
            if i != j and conf_matrix[i][j] > 0:
                true_letter = LETTER_MAPPINGS[int(true_label)]
                pred_letter = LETTER_MAPPINGS[int(pred_label)]
                confusion_pairs.append((conf_matrix[i][j], true_letter, pred_letter))
    
    confusion_pairs.sort(reverse=True)
    for count, true_letter, pred_letter in confusion_pairs[:5]:
        print(f"   • {true_letter} confused with {pred_letter}: {count} times")
    
    return overall_accuracy


#--------------------------------------------------
# FUNCTION 5: Save model
#--------------------------------------------------

def save_trained_classifier(classifier, accuracy: float) -> None:
    """
    Saves trained model to file.
    
    Why save?
    - Training takes time
    - Can load instantly for real-time detection
    """
    model_package = {
        'forest_model': classifier,
        'validation_accuracy': accuracy,
        'tree_quantity': TREE_COUNT,
        'label_to_letter': LETTER_MAPPINGS
    }
    
    with open(TRAINED_MODEL_FILE, 'wb') as output_file:
        pickle.dump(model_package, output_file)
    
    print("\n" + "=" * 70)
    print("✅ MODEL SAVED")
    print("=" * 70)
    print(f"File: {Path(TRAINED_MODEL_FILE).absolute()}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Size: {Path(TRAINED_MODEL_FILE).stat().st_size / 1024:.1f} KB")
    print(f"\n🚀 Ready for real-time detection!")
    print(f"   Next: Run live_recognizer.py")
    print("=" * 70)


#--------------------------------------------------
# FUNCTION 6: Main program
#--------------------------------------------------

def execute_training_pipeline():
    """Runs complete training process."""
    print("=" * 70)
    print("ASL ALPHABET MODEL TRAINING")
    print("=" * 70)
    
    # Load data
    feature_data, label_data = import_training_dataset()
    
    if feature_data is None:
        return
    
    # Split data
    X_train, X_val, y_train, y_val = partition_dataset(feature_data, label_data)
    
    # Train model
    trained_model = build_and_train_forest(X_train, y_train)
    
    # Test model
    final_accuracy = assess_model_performance(trained_model, X_val, y_val)
    
    # Save model
    save_trained_classifier(trained_model, final_accuracy)
    
    # Final message
    print("\n" + "=" * 70)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 70)
    if final_accuracy >= 0.95:
        print("Excellent! Model is ready.")
    elif final_accuracy >= 0.85:
        print("Good! Should work well.")
    else:
        print("Consider collecting more data.")


#--------------------------------------------------
# PROGRAM START
#--------------------------------------------------

if __name__ == "__main__":
    execute_training_pipeline()