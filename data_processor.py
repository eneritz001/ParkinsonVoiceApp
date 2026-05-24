
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
        
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        
        features = np.stack([mfcc, delta_mfcc, delta2_mfcc], axis=-1)
        
        return np.transpose(features, (1, 0, 2))
    except Exception as e:
        return None

def load_datasets(path_kcl_root, path_librispeech, path_db_it, max_samples=1000):
    X, y = [], []
    
    # Search Parkinson at KCL
    print(f"Searching for Parkinson's cases in subfolders of {path_kcl_root}...")
    files_pd = glob.glob(os.path.join(path_kcl_root, "**/PD/**/*.wav"), recursive=True)
    
    # Search Healthy at KCL
    print(f"Searching for healthy cases in subfolders of {path_kcl_root}...")
    files_hc_kcl = glob.glob(os.path.join(path_kcl_root, "**/HC/**/*.wav"), recursive=True)
    
    # Search at LibriSpeech
    print(f"Searching for audio files on LibriSpeech: {path_librispeech}")
    files_ls = glob.glob(os.path.join(path_librispeech, "**/*.flac"), recursive=True)
    files_healthy = files_hc_kcl + files_ls

    # Search at DB_IT
    print(f"Searching for Parkinson's cases in subfolders of: {path_db_it}")
    files_db_it = glob.glob(os.path.join(path_db_it, "**", "*.wav"), recursive=True)

    # Bring together all ones related to Parkinson's
    files_pd = files_pd + files_db_it
    
    import random
    random.seed(42)
    random.shuffle(files_pd)
    random.shuffle(files_healthy)

    files_pd = files_pd[:max_samples]
    files_healthy = files_healthy[:max_samples]

    print(f"Total Parkinson's disease (PD) cases to process: {len(files_pd)}")
    print(f"Total healthy individuals (HC + Libri) to process: {len(files_healthy)}")

    for f in tqdm.tqdm(files_pd, desc="Processing PD"):
        feat = extract_features_crnn(f)
        if feat is not None:
            X.append(feat)
            y.append(1)

    for f in tqdm.tqdm(files_healthy, desc="Processing HC"):
        feat = extract_features_crnn(f)
        if feat is not None:
            X.append(feat)
            y.append(0)

    return np.array(X), np.array(y)