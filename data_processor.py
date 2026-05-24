# # import librosa
# # import numpy as np
# # import os
# # import tqdm
# # import glob  

# # def extract_features_crnn(file_path, duration=3, n_mfcc=40):
# #     try:
# #         y, sr = librosa.load(file_path, sr=22050, duration=duration)
# #         if len(y) < sr * duration:
# #             y = np.pad(y, (0, sr * duration - len(y)))
        
# #         mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
# #         return mfcc.T 
# #     except Exception as e:

# #         return None

# # def load_datasets(path_pd, path_healthy, max_samples=800):
# #     X, y = [], []
    
# #     # 1. Busca audios Parkinson
# #     print(f"Searching for files in {path_pd}...")
# #     files_pd = glob.glob(os.path.join(path_pd, "**/*.wav"), recursive=True)[:max_samples]
    
# #     print(f"Processing {len(files_pd)} Parkinson's audio recordings...")
# #     for f in tqdm.tqdm(files_pd):
# #         feat = extract_features_crnn(f)
# #         if feat is not None:
# #             X.append(feat)
# #             y.append(1)

# #     print(f"Searching for files in {path_healthy}...")
# #     files_h = glob.glob(os.path.join(path_healthy, "**/*.flac"), recursive=True)[:max_samples]
    
# #     print(f"Processing {len(files_h)} healthy audio recordings...")
# #     for f in tqdm.tqdm(files_h):
# #         feat = extract_features_crnn(f)
# #         if feat is not None:
# #             X.append(feat)
# #             y.append(0)

# #     return np.array(X), np.array(y)

# import librosa
# import numpy as np
# import os
# import tqdm
# import glob

# # def extract_features_crnn(file_path, duration=3, n_mfcc=40):
# #     try:
# #         y, sr = librosa.load(file_path, sr=22050, duration=duration)
# #         if len(y) < sr * duration:
# #             y = np.pad(y, (0, sr * duration - len(y)))
# #         mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
# #         return mfcc.T 
# #     except Exception:
# #         return None

# def extract_features_crnn(file_path, duration=3, n_mfcc=40):
#     try:
#         y, sr = librosa.load(file_path, sr=22050, duration=duration)
#         if len(y) < sr * duration:
#             y = np.pad(y, (0, sr * duration - len(y)))
        
#         # 1. MFCC base
#         mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        
#         # 2. Calcular velocidad (Delta) y aceleración (Delta-Delta)
#         delta_mfcc = librosa.feature.delta(mfcc)
#         delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        
#         # 3. Concatenar para crear una imagen de 3 canales (como una imagen RGB)
#         # Forma resultante: (n_mfcc, frames, 3)
#         features = np.stack([mfcc, delta_mfcc, delta2_mfcc], axis=-1)
        
#         # Retornamos permutando los ejes para que el tiempo sea el primer eje: (frames, n_mfcc, 3)
#         return np.transpose(features, (1, 0, 2))
#     except Exception:
#         return None

# def load_datasets(path_kcl_root, path_librispeech, path_db_it, max_samples=1000):
#     X, y = [], []
    
#     # Search PD 
#     print(f"Searching for Parkinson's cases in subfolders of {path_kcl_root}...")
#     files_pd = glob.glob(os.path.join(path_kcl_root, "**/PD/**/*.wav"), recursive=True)
    
#     # Search HC 
#     print(f"Searching for healthy cases in subfolders of {path_kcl_root}...")
#     files_hc_kcl = glob.glob(os.path.join(path_kcl_root, "**/HC/**/*.wav"), recursive=True)
    
#     # Search on LIBRISPEECH 
#     print(f"Searching for audio files on LibriSpeech: {path_librispeech}")
#     files_ls = glob.glob(os.path.join(path_librispeech, "**/*.flac"), recursive=True)
    
#     # combination of all the healthy ones
#     files_healthy = files_hc_kcl + files_ls

#     print(f"Searching for Parkinson's cases in subfolders of: {path_db_it}")
#     # Buscamos todos los .wav dentro de cualquier subcarpeta de paciente
#     files_db_it = glob.glob(os.path.join(path_db_it, "**", "*.wav"), recursive=True)

#     files_db_it = files_db_it[:200]
#     files_pd = files_pd + files_db_it
#     files_pd = files_pd[:max_samples]
#     files_healthy = files_healthy[:max_samples]

#     print(f"Total Parkinson's disease (PD) cases found: {len(files_pd)}")
#     print(f"Total healthy individuals (HC + Libri) found: {len(files_healthy)}")

#     # label Parkinson (1)
#     for f in tqdm.tqdm(files_pd, desc="Processing PD"):
#         feat = extract_features_crnn(f)
#         if feat is not None:
#             X.append(feat)
#             y.append(1)

#     # label healthy ones (0)
#     for f in tqdm.tqdm(files_healthy, desc="Processing HC"):
#         feat = extract_features_crnn(f)
#         if feat is not None:
#             X.append(feat)
#             y.append(0)

#     return np.array(X), np.array(y)


import librosa
import numpy as np
import os
import tqdm
import glob  

def extract_features_crnn(file_path, duration=3, n_mfcc=40):
    try:
        y, sr = librosa.load(file_path, sr=22050, duration=duration)
        if len(y) < sr * duration:
            y = np.pad(y, (0, sr * duration - len(y)))
        
        # 1. Extraer MFCC base
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        
        # 2. Calcular características dinámicas (Velocidad y Aceleración)
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        
        # 3. Concatenar los 3 canales en el último eje -> (n_mfcc, frames, 3)
        features = np.stack([mfcc, delta_mfcc, delta2_mfcc], axis=-1)
        
        # Permutar ejes para que el tiempo sea la primera dimensión -> (frames, n_mfcc, 3)
        return np.transpose(features, (1, 0, 2))
    except Exception as e:
        return None

def load_datasets(path_kcl_root, path_librispeech, path_db_it, max_samples=1000):
    X, y = [], []
    
    # Buscar Parkinson en KCL
    print(f"Searching for Parkinson's cases in subfolders of {path_kcl_root}...")
    files_pd = glob.glob(os.path.join(path_kcl_root, "**/PD/**/*.wav"), recursive=True)
    
    # Buscar Sanos en KCL
    print(f"Searching for healthy cases in subfolders of {path_kcl_root}...")
    files_hc_kcl = glob.glob(os.path.join(path_kcl_root, "**/HC/**/*.wav"), recursive=True)
    
    # Buscar en LibriSpeech
    print(f"Searching for audio files on LibriSpeech: {path_librispeech}")
    files_ls = glob.glob(os.path.join(path_librispeech, "**/*.flac"), recursive=True)
    files_healthy = files_hc_kcl + files_ls

    # Buscar en DB_IT
    print(f"Searching for Parkinson's cases in subfolders of: {path_db_it}")
    files_db_it = glob.glob(os.path.join(path_db_it, "**", "*.wav"), recursive=True)

    # Combinamos todo el Parkinson
    files_pd = files_pd + files_db_it
    
    # Mezclamos de forma aleatoria antes de recortar por el max_samples
    import random
    random.seed(42)
    random.shuffle(files_pd)
    random.shuffle(files_healthy)

    files_pd = files_pd[:max_samples]
    files_healthy = files_healthy[:max_samples]

    print(f"Total Parkinson's disease (PD) cases to process: {len(files_pd)}")
    print(f"Total healthy individuals (HC + Libri) to process: {len(files_healthy)}")

    # Procesar Parkinson (Etiqueta 1)
    for f in tqdm.tqdm(files_pd, desc="Processing PD"):
        feat = extract_features_crnn(f)
        if feat is not None:
            X.append(feat)
            y.append(1)

    # Enfoque de Sanos (Etiqueta 0)
    for f in tqdm.tqdm(files_healthy, desc="Processing HC"):
        feat = extract_features_crnn(f)
        if feat is not None:
            X.append(feat)
            y.append(0)

    return np.array(X), np.array(y)