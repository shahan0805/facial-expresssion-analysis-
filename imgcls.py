# We install the FER() library to perform facial recognition
# This installation will also take care of any of the above dependencies if they are missing
#pip install FER

from fer import FER
import matplotlib.pyplot as plt 
#%matplotlib inline

test_image_one = plt.imread(r"C:\Users\optim\Desktop\video interview\istockphoto-1159627027-1024x1024.jpg")
emo_detector = FER(mtcnn=True)
# Capture all the emotions on the image
captured_emotions = emo_detector.detect_emotions(test_image_one)
# Print all captured emotions with the image
print(captured_emotions)
plt.imshow(test_image_one)

# Use the top Emotion() function to call for the dominant emotion in the image
dominant_emotion, emotion_score = emo_detector.top_emotion(test_image_one)
print(dominant_emotion, emotion_score)
