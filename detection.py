import numpy as np
import cv2
import os
import numpy as np
from tensorflow import keras


model = keras.models.load_model('./CatNet/')


cap = cv2.VideoCapture('./cat.mp4')

# Get video width, height and frames per second
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)
print(frame_height, frame_width, fps)
# Define codec and create a VideoWriter object
# out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (100, 100))
output_folder = './result_images'

count = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        image = cv2.resize(frame, (100, 100))  
        image = image.astype('float32') / 255.0
        image = np.expand_dims(image, axis=0)

        # Make a prediction
        prediction = np.argmax(model.predict(image))
        catdog = {0:'Cat', 1:'Dog'}
        cv2.putText(frame, catdog[prediction], (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        # out.write(image.astype(np.uint16))
        cv2.imwrite(os.path.join(output_folder, f"frame_{count}.jpg"), frame)
        count += 1

    else:
        break

# Release everything
cap.release()
# out.release()
cv2.destroyAllWindows()
