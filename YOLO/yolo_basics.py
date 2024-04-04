from ultralytics import YOLO
import cv2
import cvzone
import math



model_1 = YOLO('./Yolo-Weights/yolov8_custom.pt')

model_2 = YOLO('./Yolo-Weights/yolov8n.pt')


cap = cv2.VideoCapture("videos/video_1.mp4")
cap.set(3, 1280)
cap.set(4, 720)

model = model_1

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Draw boxes
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2-x1, y2-y1
            cvzone.cornerRect(img, (x1, y1, w, h))

            conf = math.ceil((box.conf[0] * 100))/100
            print(conf)
            # cvzone.putTextRect(img, f'{conf}', )

    cv2.imshow("Image", img)
    cv2.waitKey(1)
