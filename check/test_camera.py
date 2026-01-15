import cv2

cap = cv2.VideoCapture(2)

if not cap.isOpened():
    print("카메라 열기 실패")
else:
    print("카메라 열기 성공")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Product Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
