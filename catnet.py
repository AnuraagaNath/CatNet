from tensorflow import keras
import tensorflow as tf

print(tf.config.list_physical_devices())

# This model is created using model subclassing method.

# creating custom CNN blocks
class CatBlock(keras.layers.Layer):
    def __init__(self, filter_size, kernel_size=(3,3), activation='relu', pool_size=(2,2)):
        super(CatBlock, self).__init__()
        self.conv = keras.layers.Conv2D(filters=filter_size, kernel_size=kernel_size, activation=activation)
        self.bn = keras.layers.BatchNormalization()
        self.pooling = keras.layers.MaxPooling2D(pool_size=pool_size)

    def call(self, input, training=False):
        x = self.conv(input)
        x = self.bn(x, training=training)
        return self.pooling(x)

# creating CNN model
@keras.saving.register_keras_serializable()
class CatNet(keras.Model):
    def __init__(self, num_classes=1, rate = 0.5, output_activation='sigmoid', activation='relu'):
        super(CatNet, self).__init__()
        self.cnn1 = CatBlock(filter_size=64)
        self.cnn2 = CatBlock(filter_size=128)
        self.cnn3 = CatBlock(filter_size=256)
        self.flatten = keras.layers.Flatten()
        self.connected = keras.layers.Dense(128, activation=activation)
        self.dropout = keras.layers.Dropout(rate=rate)
        self.classifier = keras.layers.Dense(num_classes, activation=output_activation)

    def call(self, input, training=False):
        x = self.cnn1(input, training=training)
        x = self.cnn2(x, training = training)
        x = self.cnn3(x, training = training)
        x = self.flatten(x)
        x = self.connected(x)
        x = self.dropout(x)
        return self.classifier(x)
    
    def model(self):
        x = keras.Input(shape=(100, 100, 3))
        return keras.Model(inputs=[x], outputs=self.call(x))
