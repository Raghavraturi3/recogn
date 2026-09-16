from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")  # nano model, auto-downloads, ~6MB, fast

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame, classes=[0])  # class 0 = 'person' in COCO
    annotated = results[0].plot()
    cv2.imshow("Person Detection", annotated)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()