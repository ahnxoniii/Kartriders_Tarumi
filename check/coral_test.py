import cv2
import numpy as np
import time
from edge_tpu_silva import process_detection

MODEL_PATH = '/home/kart/yolo_test/240_yolov8n_full_integer_quant_edgetpu.tflite'
IMGSZ = 640

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("카메라 열기 실패")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임 읽기 실패")
        continue

    frame_resized = cv2.resize(frame, (IMGSZ, IMGSZ))
    temp_path = "/tmp/live_frame.jpg"
    cv2.imwrite(temp_path, frame_resized)

    outs = process_detection(
        model_path=MODEL_PATH,
        input_path=temp_path,
        imgsz=IMGSZ,
        threshold=0.4,
        verbose=False,
        show=False
    )

    for objs_lst, fps in outs:
        for obj in objs_lst:
            x1, y1, x2, y2 = map(int, obj["bbox"])
            label = f"{obj['label']} {obj['conf']:.2f}"
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_resized, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.putText(frame_resized, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Coral YOLO Detection", frame_resized)
        break  # 하나의 프레임에 대한 예측만 사용

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# from edge_tpu_silva import process_detection

# # Run the object detection process
# outs = process_detection(model_path='/home/kart/yolo_test/NP-converted-best.tflite', input_path='/home/kart/yolo_test/img.png', imgsz=640)

# for _, _ in outs:
#   pass