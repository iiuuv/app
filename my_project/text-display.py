import cv2
img = cv2.imread("/app/result.jpg")
cv2.imshow("Test", img)
cv2.waitKey(0)
cv2.destroyAllWindows()