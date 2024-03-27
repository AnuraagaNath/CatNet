from tensorflow import keras
import os
from glob import glob
from catnet import CatNet 
from preprocessing import getTrainValidData

filepath = './PetImages'
train_generator, valid_generator = getTrainValidData(filepath)

model = CatNet()
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])
earlystopping = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=20)
print(model.model().summary())


history=model.fit(train_generator, validation_data=valid_generator, epochs=100, batch_size=128, callbacks=earlystopping)


model.save('./CatNet/')