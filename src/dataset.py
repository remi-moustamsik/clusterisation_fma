import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from src.audio_processing import audio_to_mel_tensor  # ta fonction qui renvoie (1, 128, T)

class FMADataset(Dataset):
    def __init__(self, file_paths, track_genre_ids, genre_id_to_idx, max_len=3000):
        """
        file_paths        : liste de chemins vers les fichiers audio
        track_genre_ids   : liste des genre_id (comme dans le CSV)
        genre_id_to_idx   : dict {genre_id: index_de_classe}
        max_len           : longueur temporelle fixe (en frames) pour les mels
        """
        self.file_paths = file_paths
        self.track_genre_ids = track_genre_ids
        self.genre_id_to_idx = genre_id_to_idx
        self.max_len = max_len

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        genre_id = self.track_genre_ids[idx]              # ex: 15, 12, 1235, ...
        label_idx = self.genre_id_to_idx[genre_id]        # ex: 0…num_classes-1

        mel, sr = audio_to_mel_tensor(path)               # mel : (1, 128, T)
        mel = mel.float()

        # --- rendre T constant ---
        T = mel.size(2)

        if T > self.max_len:
            mel = mel[:, :, :self.max_len]                # on coupe

        elif T < self.max_len:
            pad_amount = self.max_len - T                 # on pad à droite
            mel = F.pad(mel, (0, pad_amount, 0, 0))

        label = torch.tensor(label_idx, dtype=torch.long)

        return mel, label
