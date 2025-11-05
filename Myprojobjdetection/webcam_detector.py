from ultralytics import YOLO
import cv2
CUSTOM_MODEL_PATH = 'best.pt' 
WEBCAM_SOURCE = 0 

try:
    model = YOLO(CUSTOM_MODEL_PATH)
    print(f" Custom model loaded successfully from: {CUSTOM_MODEL_PATH}")
except Exception as e:
    print(f" Error loading model: {e}")
    print("Ensure 'best.pt' is in the current directory or the path is correct.")
    exit()
cap = cv2.VideoCapture(WEBCAM_SOURCE)

if not cap.isOpened():
    print(f"Error: Could not open webcam at source {WEBCAM_SOURCE}.")
    exit()

print("Starting real-time detection. Press 'q' to exit the window.")

while cap.isOpened():
    
    success, frame = cap.read()

    if success:
       
        results = model.predict(
            source=frame, 
            conf=0.5, 
            verbose=False
        )

        
        annotated_frame = results[0].plot() 

        
        cv2.imshow("Real-Time Object Detection (Custom YOLOv8)", annotated_frame)

        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
       
        break


cap.release()
cv2.destroyAllWindows()