import cv2

# Loading the haarcascade model from opencv 
faceClassifier = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
video_capture = cv2.VideoCapture(0) # Uses the webcam to detect faces

# Function to detect a face and draw a square around it 
def detect_bound(img):
  
  gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  face = faceClassifier.detectMultiScale(gray_img,scaleFactor=1.1,minNeighbors=5,minSize=(40,40))
  
  for (x,y,w,h) in face:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
  return face

# loop to display the face 
while True:
  result,frame = video_capture.read()
  if result == False:   # Checks if the frame has generates succesfully
    break
  face = detect_bound(frame)
  cv2.imshow("demo",frame)

  if cv2.waitKey(20) & 0xFF == ord("q"):  # Hit 'Q' key to exit from the window 
    break 

# deleting all the instances
video_capture.release()
cv2.destroyAllWindows()

