# import tensorflow as tf
# from tensorflow.keras import layers, models

# # def build_crnn(input_shape):
# #     model = models.Sequential([
# #         # CNN -> patrones espectrales
# #         layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
# #         layers.BatchNormalization(),
# #         layers.MaxPooling2D((2, 2)),
     
# #         layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
# #         layers.BatchNormalization(),
# #         layers.MaxPooling2D((2, 2)),

# #         # Preparar secuencia temporal
# #         layers.Reshape(target_shape=(input_shape[0] // 4, -1)), 

# #         # RNN (LSTM) -> detectar inestabilidad en el tiempo
# #         layers.LSTM(64, return_sequences=True),
# #         layers.LSTM(32),

# #         layers.Dense(64, activation='relu'),
# #         layers.Dropout(0.4),
# #         layers.Dense(1, activation='sigmoid')
# #     ])


# def build_crnn(input_shape):
#     model = models.Sequential([
#         # Bloque Convolucional 1
#         layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
#         layers.BatchNormalization(),
#         # Reducimos solo frecuencias (eje 1), mantenemos el tiempo (eje 0) intacto
#         layers.MaxPooling2D((1, 2)), 
#         layers.Dropout(0.2),
        
#         # Bloque Convolucional 2
#         layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
#         layers.BatchNormalization(),
#         layers.MaxPooling2D((1, 2)),
#         layers.Dropout(0.2),

#         # En lugar de una fórmula rígida, aplanamos las frecuencias manteniendo el tiempo libre
#         layers.Reshape(target_shape=(input_shape[0], -1)),
        
#         # Capas Recurrentes Bidireccionales (capturan contexto adelante y atrás en el tiempo)
#         layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
#         layers.BatchNormalization(),
#         layers.Bidirectional(layers.LSTM(32)),
#         layers.Dropout(0.3),

#         # Clasificador
#         layers.Dense(64, activation='relu'),
#         layers.BatchNormalization(),
#         layers.Dropout(0.4),
#         layers.Dense(1, activation='sigmoid')
#     ])
    
#     # Usar un learning rate ligeramente más controlado
#     opt = tf.keras.optimizers.Adam(learning_rate=0.0005)
#     model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
#     return model
    
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#     return model


import tensorflow as tf
from tensorflow.keras import layers, models

def build_crnn(input_shape):
    # input_shape esperado ahora: (frames, n_mfcc, 3) debido a los deltas
    model = models.Sequential([
        # Bloque Convolucional 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        # MaxPool estratégico: reduce solo frecuencias (eje 1), mantiene intacto el tiempo (eje 0)
        layers.MaxPooling2D((1, 2)),
        layers.Dropout(0.2),
        
        # Bloque Convolucional 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((1, 2)),
        layers.Dropout(0.2),

        # Reshape adaptativo: mantiene el eje de tiempo libre y colapsa las frecuencias restantes
        layers.Reshape(target_shape=(input_shape[0], -1)), 
        
        # Bloque Recurrente Bidireccional
        layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
        layers.BatchNormalization(),
        layers.Bidirectional(layers.LSTM(32)),
        layers.Dropout(0.3),

        # Clasificador Denso
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Optimizador con un Learning Rate más controlado para evitar saltos drásticos en pérdidas
    opt = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model