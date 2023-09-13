import cv2
face_cap=cv2.CascadeClassifier("C:/Users/LENOVO/AppData/Roaming/Python/Python311/site-packages/cv2/data/haarcascade_frontalface_default.xml")

video_cap=cv2.VideoCapture(0)
while True:

    ret,video_data=video_cap.read()
    # print(ret)
    col=cv2.cvtColor(video_data,cv2.COLOR_BGR2GRAY)

    faces = face_cap.detectMultiScale(
        col,
            #z=[1,2,3,4]
        scaleFactor=1.1,
        minSize=(20,20),
        flags=cv2.CASCADE_SCALE_IMAGE
        )

    for (x,y,w,h) in faces:
        cv2.rectangle(video_data,(x,y),(x+w,y+h),(0,255,0),3)
    cv2.imshow("Face detector",video_data)
    if cv2.waitKey(10) == ord("c"):
        break
        
video_cap.release()
cv2.destroyAllWindows()

