
import tensorflow as tf
from tensorflow.keras import layers, models

def build_crnn(input_shape):
    model = models.Sequential([
        #Convolutional Layer 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        # MaxPool 
        layers.MaxPooling2D((1, 2)),
        layers.Dropout(0.2),
        
        #Convolutional Layer 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((1, 2)),
        layers.Dropout(0.2),

        #Adaptive reshape
        layers.Reshape(target_shape=(input_shape[0], -1)), 
        
        #Bidirectional Recurring Block
        layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
        layers.BatchNormalization(),
        layers.Bidirectional(layers.LSTM(32)),
        layers.Dropout(0.3),

        #Dense classifier
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model