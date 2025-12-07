import librosa
import numpy as np
import torch

def audio_to_mel_tensor(path, n_mels=128, sr=None, norm="zscore"):
    try:
        y, sr = librosa.load(path, sr=sr, mono=True)
    except Exception as e:
        print(f"Erreur de chargement {path}: {e}")
        return None, None
    
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=2048,
        hop_length=512,
        n_mels=n_mels,
        power=2.0
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    
    if norm == "zscore":
        mean = S_db.mean()
        std = S_db.std() + 1e-8
        S_norm = (S_db - mean) / std
    elif norm == "minmax":
        S_min = S_db.min()
        S_max = S_db.max()
        S_norm = (S_db - S_min) / (S_max - S_min + 1e-8)
    else:
        S_norm = S_db  # pas de normalisation

    mel_tensor = torch.tensor(S_norm, dtype=torch.float32).unsqueeze(0)  # (1, 128, T)
    return mel_tensor, sr