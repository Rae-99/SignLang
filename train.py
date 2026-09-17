import pickle
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def train_classifier(csv_path="custom_asl_landmarks.csv", model_output="asl_model.p"):
    print(f"Loading normalized landmark data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: {csv_path} not found. Run collect_data.py first.")
        return

    # Separate target labels and coordinate features
    X = df.drop(columns=['label'])
    y = df['label']

    print(f"Dataset shape: {X.shape[0]} samples, {X.shape[1]} features per hand.")

    # 80/20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Initialize and train Support Vector Machine (SVM)
    print("\nTraining Support Vector Machine (SVM) Classifier...")
    # C=10 and gamma='scale' are excellent starting hyperparameters for normalized coordinate data
    model = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Save the SVM model object for real-time inference
    with open(model_output, 'wb') as f:
        pickle.dump(model, f)

    print(f"\nSVM Model successfully saved to {model_output}")

if __name__ == "__main__":
    train_classifier()