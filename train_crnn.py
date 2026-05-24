
from data_processor import load_datasets
from model_architecture import build_crnn
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import numpy as np

# ROUTE SETTINGS
PATH_KCL = r"C:\Users\Hpp\Downloads\parkinsonvoiceapp\26_29_09_2017_KCL\26-29_09_2017_KCL"
PATH_HEALTHY = r"C:\Users\Hpp\Downloads\parkinsonvoiceapp\train-clean-100"
PATH_DB_IT = r"C:\Users\Hpp\Downloads\parkinsonvoiceapp\DB_IT"

# X, y = load_datasets(PATH_PD, PATH_HEALTHY, PATH_DB_IT, max_samples=472)
X, y = load_datasets(PATH_KCL, PATH_HEALTHY, PATH_DB_IT, max_samples=600)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = {0: weights[0], 1: weights[1]}
print(f"Pesos de clase balanceados calculados: {class_weights}")

input_shape = (X.shape[1], X.shape[2], X.shape[3]) # (frames, 40, 3)
model = build_crnn(input_shape)

print("Starting Deep Learning training with dynamic features and class weights...")
model.fit(
    X_train, y_train, 
    validation_data=(X_test, y_test), 
    epochs=40, 
    batch_size=32,
    class_weight=class_weights  
)

#Save model
model.save("parkinson_crnn_model.h5")
print("✅ ¡Nuevo modelo mejorado guardado con éxito como 'parkinson_crnn_model.h5'!")