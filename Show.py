import cv2
import numpy as np
from keras.models import model_from_json
import string
from Image_processing import func2  # Assuming func2 is correctly defined

# Load the pre-trained model
def load_model():
    model_dir = ''  # Change this to the path where your models are stored
    # Load the model architecture from JSON file
    json_file = open(model_dir + 'model-bw.json', 'r')
    model_json = json_file.read()
    json_file.close()

    # Load the model weights
    model = model_from_json(model_json)
    model.load_weights(model_dir + 'model-bw.weights.h5')

    return model

# Prediction function
def predict(model, img):
    # Preprocess the image for prediction
    img = cv2.resize(img, (128, 128))  # Resize image to the model's input size
    img = np.expand_dims(img, axis=-1)  # Add an extra dimension for channels (grayscale)
    img = np.expand_dims(img, axis=0)  # Add an extra dimension for batch size
    
    # Normalize the image
    img = img / 255.0
    
    # Predict using the model
    prediction = model.predict(img)
    
    # Only consider A, B, C for prediction (indices 0, 1, 2)
    alphabet = ['A', 'B', 'C']
    
    # Get the predicted class index (A = 0, B = 1, C = 2)
    pred_index = np.argmax(prediction, axis=1)[0]
    
    # Return the predicted class (A, B, or C)
    return alphabet[pred_index], prediction[0][pred_index] * 100  # Return character and confidence as percentage

# Set up OpenCV windows and frames
def draw_ui(frame, prediction, confidence, gray_hand):
    height, width = frame.shape[:2]
    
    # Calculate frame size and position
    frame_width = int(width * 0.7)  # 70% of screen width
    frame_height = int(height * 0.75)  # 75% of screen height
    frame_x = (width - frame_width) // 2  # Center the frame horizontally
    frame_y = (height - frame_height) // 2  # Center the frame vertically
    
    # Draw frames (Centered frame)
    cv2.rectangle(frame, (frame_x, frame_y), (frame_x + frame_width, frame_y + frame_height), (0, 255, 0), 5)

    # Align the text to center-bottom
    text = f'Prediction: {prediction} | Confidence: {confidence:.2f}%'
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    text_x = (width - text_size[0]) // 2  # Center horizontally
    text_y = height - 20  # Position text 20px from the bottom
    
    # Display the prediction (blue text) and confidence (green text)
    cv2.putText(frame, f'Prediction: {prediction}', (text_x, text_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)  # Blue for prediction
    cv2.putText(frame, f'Confidence: {confidence:.2f}%', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Green for confidence
    
    # Draw a separate hand box in the top-left corner with increased size
    hand_box_size = 250  # Increased size for the hand box
    cv2.rectangle(frame, (20, 20), (20 + hand_box_size, 20 + hand_box_size), (255, 0, 0), 2)  # Blue box for hand
    
    # Resize the grayscale hand to fit in the top-left corner and display it
    gray_hand_resized = cv2.resize(gray_hand, (hand_box_size, hand_box_size))
    frame[20:20 + hand_box_size, 20:20 + hand_box_size] = cv2.cvtColor(gray_hand_resized, cv2.COLOR_GRAY2BGR)

    # Show the frame
    cv2.imshow("Sign Language to Text", frame)

# Main function
def main():
    model = load_model()  # Load the pre-trained model
    
    # OpenCV Video Capture (to simulate live input, e.g., webcam)
    cap = cv2.VideoCapture(0)
    
    # Set the window to full screen
    cv2.namedWindow("Sign Language to Text", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Sign Language to Text", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Convert to grayscale for easier processing (assuming input is a grayscale image)
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Use a region of interest (ROI) for the hand detection area (top-left corner)
        hand_box_size = 300  # Increased hand box size
        roi = frame[20:20 + hand_box_size, 20:20 + hand_box_size]  # Extract ROI from the top-left corner
        
        # Apply func2 only on the hand box (ROI)
        processed_hand = func2(roi)
        
        # Make a prediction on the thresholded image
        prediction, confidence = predict(model, processed_hand)
        
        # Draw the UI frame and display prediction and confidence
        draw_ui(frame, prediction, confidence, processed_hand)  # Pass the original hand ROI for visualization
        
        # Exit on pressing the 'Esc' key
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
