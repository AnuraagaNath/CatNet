import numpy as np
import cv2
import os
import numpy as np
from tensorflow import keras
import video


model = keras.models.load_model('./CatNet/')


cap = cv2.VideoCapture('./cat.mp4')
output_folder = './result_images'

count = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        image = cv2.resize(frame, (100, 100))  
        image = image.astype('float32') / 255.0
        image = np.expand_dims(image, axis=0)

        # Make a prediction
        prediction = model.predict(image)
        catdog = {0:'Cat', 1:'Dog'}
        animal_class = np.argmax(prediction)
        cv2.putText(frame, catdog[animal_class] +' '+ str(prediction[0]), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(output_folder, f"frame_{count}.jpg"), frame)
        count += 1
    else:
        break



# Release everything
cap.release()
cv2.destroyAllWindows()
import re

# Folder containing the saved images
input_folder = './result_images'
output_video = 'output_video.mp4'

# Generate output video
video.generate_output(input_folder, output_video)


