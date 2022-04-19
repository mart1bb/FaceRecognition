import cv2

video = cv2.VideoCapture(0)
a = 0
# img = cv2.imread("C:/Users/marti/OneDrive/Bureau/Algo/Code/VisualStudioPy/EyeTracking/oeil.jpg")
# frame = cv2.resize(img, (0, 0), fx = 0.75, fy = 0.75)
while True:
    a = a+1

    check, frame = video.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7 ,7), 0)

    _, thresholdB = cv2.threshold(gray, 35, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresholdB, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 80 and area < 200 :
            cv2.drawContours(frame, [cnt], -1, (0, 0, 255), 3) 


    cv2.imshow("Threshold Black", thresholdB)
    cv2.imshow("Capturing", frame)

    key = cv2.waitKey(1)

    if(key == ord('q')):
        break

video.release()
cv2.destroyAllWindows()