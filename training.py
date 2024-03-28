from tensorflow import keras
import os
from glob import glob
from catnet import CatNet 
from preprocessing import getTrainValidData

# get data
filepath = './PetImages'
train_generator, valid_generator = getTrainValidData(filepath)

# load model
model = CatNet()

# compile model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])
earlystopping = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=20)
print(model.model().summary())

# train model
history=model.fit(train_generator, validation_data=valid_generator, epochs=100, batch_size=128, callbacks=earlystopping)

# save model in a folder. This process saves weights and all settings
model.save('./CatNet/')