import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import pandas as pd

class PrecomputedFMADataset(Dataset):
    def __init__(self, metadata_csv, max_len=3000):
        self.df = pd.read_csv(metadata_csv)
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]            # assume self.metadata read from CSV in __init__
        feat = torch.load(row["feature_path"])

        # ensure tensor and dtype
        if not isinstance(feat, torch.Tensor):
            feat = torch.tensor(feat)
        feat = feat.float()                      # float32

        # ensure channel dim (1, n_mels, T)
        if feat.dim() == 2:
            feat = feat.unsqueeze(0)
        elif feat.dim() == 3 and feat.size(0) != 1 and feat.size(1) == 128:
            # sometimes saved as (n_mels, T, 1) — try to reorder if needed
            feat = feat.permute(2, 0, 1)

        # pad/truncate to max_len (same logic que FMADataset)
        T = feat.size(2)
        if T > self.max_len:
            feat = feat[:, :, : self.max_len]
        elif T < self.max_len:
            pad_amount = self.max_len - T
            import torch.nn.functional as F
            feat = F.pad(feat, (0, pad_amount, 0, 0))

        # read label safely (CSV can give numpy ints/strings)
        label_idx = int(row["label_idx"])
        label = torch.tensor(label_idx, dtype=torch.long)

        return feat, label