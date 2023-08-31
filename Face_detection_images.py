import cv2 

# Loading the haarcascade model from opencv 
faceClassifier = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
path = 'images/1.jpg' # Image path
img = cv2.imread(path)

def rescaleFrame(frame, scale = 0.9):
    h = int(frame.shape[0] * scale)
    w = int(frame.shape[1] * scale)
    dim = (w,h)
    
    return cv2.resize(frame,dim,interpolation=cv2.INTER_AREA)


# Function to detect a face and draw a square around it 
def detect_bound(img):
  
  gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  face = faceClassifier.detectMultiScale(gray_img,scaleFactor=1.1,minNeighbors=5,minSize=(20,20))
  
  for (x,y,w,h) in face:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
  return face

face = detect_bound(img)
cv2.imshow(path,rescaleFrame(img,1))   # display the face

cv2.waitKey(0)  # Press any key to exit

