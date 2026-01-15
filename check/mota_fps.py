import cv2
from ultralytics import YOLO
import time
import numpy as np
from collections import deque
from short import Sort

def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    x1_end, y1_end = x1 + w1, y1 + h1
    x2_end, y2_end = x2 + w2, y2 + h2
    x_intersect = max(x1, x2)
    y_intersect = max(y1, y2)
    x_intersect_end = min(x1_end, x2_end)
    y_intersect_end = min(y1_end, y2_end)
    intersection_area = max(0, x_intersect_end - x_intersect) * max(0, y_intersect_end - y_intersect)
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - intersection_area
    return intersection_area / union_area if union_area > 0 else 0

def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("열기 실패. 종료합니다.")
        exit()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Initialize YOLOv8
    model = YOLO("/home/kart/yolo_test/best_full_integer_quant_edgetpu.tflite", task='detect')
    # Initialize SORT tracker
    tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
    tracked_id = None  # Track a single object ID
    bbox = None
    pTime = None
    fps_list = deque(maxlen=100)  # Store last 100 FPS values
    false_positive_count = 0
    frame_count = 0

    # MOTA calculation
    matches = 0
    misses = 0
    false_positives = 0
    total_gt_objects = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("읽기 실패. 종료.")
                break

            frame_count += 1

            # FPS calculation
            fps = 0  # Initialize fps to 0 for the first frame
            cTime = time.time()
            if pTime is not None:  # Skip FPS for first frame
                delta_time = cTime - pTime
                fps = 1 / delta_time if delta_time > 0 else 0
                fps_list.append(fps)
            pTime = cTime

            # YOLO detection (every frame for MOTA ground truth)
            results = model.predict(source=frame, conf=0.7, iou=0.7, save=False, verbose=False)
            detections = results[0]
            gt_bbox = None
            dets = []  # Detections for SORT

            # Prepare detections for SORT and select ground truth
            if len(detections) > 0:
                boxes = detections.boxes.xyxy.cpu().numpy()
                class_ids = detections.boxes.cls.cpu().numpy()
                confidences = detections.boxes.conf.cpu().numpy()
                max_conf_idx = -1
                max_conf = 0
                for i, (box, class_id, conf) in enumerate(zip(boxes, class_ids, confidences)):
                    if int(class_id) == 0:  # Wheelchair user class
                        dets.append(np.append(box, conf))  # [x1, y1, x2, y2, conf]
                        if conf > max_conf:
                            max_conf = conf
                            max_conf_idx = i
                if max_conf_idx >= 0:
                    xmin, ymin, xmax, ymax = map(int, boxes[max_conf_idx])
                    gt_bbox = (xmin, ymin, xmax - xmin, ymax - ymin)
                    total_gt_objects += 1

            # Convert detections to NumPy array for SORT
            dets = np.array(dets) if dets else np.empty((0, 5))

            # SORT tracking
            tracked_objects = tracker.update(dets)
            bbox = None
            tracking = tracked_id is not None

            # Match tracked objects with ground truth
            if len(tracked_objects) > 0:
                # If not tracking, select the first tracked object
                if not tracking and len(dets) > 0:
                    tracked_id = int(tracked_objects[0, 4])  # Assign the first track ID
                    tracking = True
                    print(f"Tracker initialized with ID {tracked_id}")

                # Find the tracked object with the assigned ID
                for obj in tracked_objects:
                    if int(obj[4]) == tracked_id:
                        x1, y1, x2, y2 = map(int, obj[:4])
                        bbox = (x1, y1, x2 - x1, y2 - y1)
                        break

                if bbox and gt_bbox:
                    iou = calculate_iou(bbox, gt_bbox)
                    if iou >= 0.5:  # IoU threshold for a match
                        matches += 1
                        false_positive_count = 0
                    else:
                        misses += 1
                        false_positive_count = 0
                elif bbox and not gt_bbox:
                    false_positives += 1
                    false_positive_count += 1
                    if false_positive_count > 10:  # Stop tracking after 10 frames without ground truth
                        tracking = False
                        tracked_id = None
                        print("Tracking stopped due to no ground truth for 10 frames.")
                elif not bbox and gt_bbox:
                    misses += 1
                    false_positive_count = 0
            else:
                if gt_bbox:
                    misses += 1
                false_positive_count = 0
                tracking = False
                tracked_id = None

            # Draw ground truth bounding box (if available)
            if gt_bbox:
                xmin, ymin, w, h = map(int, gt_bbox)
                cv2.rectangle(frame, (xmin, ymin), (xmin + w, ymin + h), (0, 255, 0), 1)

            # Draw tracker bounding box (if available)
            if bbox:
                xmin, ymin, w, h = map(int, bbox)
                cv2.rectangle(frame, (xmin, ymin), (xmin + w, ymin + h), (255, 0, 0), 2)
                cv2.putText(frame, f"ID: {tracked_id}", (xmin, ymin - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)

            # Display FPS and MOTA
            avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
            mota = (matches - misses - false_positives) / total_gt_objects if total_gt_objects > 0 else 0
            cv2.putText(frame, f"FPS: {fps:.2f} (Avg: {avg_fps:.2f})", (20, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"MOTA: {mota:.4f}", (20, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

            # Log MOTA components every 100 frames
            if frame_count % 100 == 0:
                print(f"Frame {frame_count}: MOTA={mota:.4f}, Matches={matches}, Misses={misses}, "
                      f"False Positives={false_positives}, Total GT Objects={total_gt_objects}")

            # Show frame
            cv2.imshow("YOLO + SORT Tracking", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exit key pressed.")
                break

    except KeyboardInterrupt:
        print("\n강제 종료.")
    finally:
        # Print final metrics
        avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
        mota = (matches - misses - false_positives) / total_gt_objects if total_gt_objects > 0 else 0
        print(f"Final Average FPS: {avg_fps:.2f}")
        print(f"Final MOTA: {mota:.4f} (Matches: {matches}, Misses: {misses}, "
              f"False Positives: {false_positives}, Total GT Objects: {total_gt_objects})")

        cap.release()
        cv2.destroyAllWindows()
        print("리소스 해제. 프로그램 종료.")

if __name__ == '__main__':
    main()