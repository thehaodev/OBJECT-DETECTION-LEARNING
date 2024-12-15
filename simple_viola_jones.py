import cv2
from matplotlib import pyplot as plt

# Initialize cascade classifier with pre-trained haar-like facial features
classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Read an example image
image = cv2.imread("img.png")
# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Detect faces in the image
face = classifier.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags=(cv2.CASCADE_SCALE_IMAGE +
           cv2.CASCADE_DO_CANNY_PRUNING +
           cv2.CASCADE_FIND_BIGGEST_OBJECT +
           cv2.CASCADE_DO_ROUGH_SEARCH)
)
# Draw a rectangle around the faces
for (x, y, w, h) in face:
    cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 2)
# Display image
plt.imshow(gray, 'gray')
plt.show()
