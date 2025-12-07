import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioRNN(nn.Module):
    """
    RNN pour audio, entrées : (B, 1, 128, T)
    - Produit un embedding pour clustering
    - Optionnellement, une tête de classification
    
    Utilisation typique :
        logits, emb = model(x)      # si num_classes n'est pas None
        emb = model(x)[-1]          # récupérer juste les embeddings
    """
    def __init__(
        self,
        n_mels: int = 128,
        hidden_size: int = 128,
        num_layers: int = 2,
        bidirectional: bool = True,
        embedding_dim: int = 128,
        num_classes: int | None = None,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.n_mels = n_mels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        # RNN : on lit la séquence de frames de mels
        # Entrée GRU : (B, T, n_mels)
        self.rnn = nn.GRU(
            input_size=n_mels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        rnn_out_dim = hidden_size * (2 if bidirectional else 1)

        # Projection vers un espace d'embedding plus compact
        self.embedding_proj = nn.Sequential(
            nn.Linear(rnn_out_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Tête de classification optionnelle
        if num_classes is not None:
            self.classifier = nn.Linear(embedding_dim, num_classes)
        else:
            self.classifier = None

        self._softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        x : tensor (B, 1, 128, T)
        Retour :
            - si num_classes est défini : (logits, probs, embedding)
            - sinon : embedding
        """
        # x: (B, 1, 128, T) -> (B, 128, T)
        x = x.squeeze(1)                 # (B, 128, T)
        x = x.transpose(1, 2)           # (B, T, 128) : T = temps, 128 = features

        # RNN
        rnn_out, _ = self.rnn(x)        # rnn_out: (B, T, rnn_out_dim)

        # Pooling temporel global : moyenne + max (plus riche qu'un seul last step)
        mean_pool = rnn_out.mean(dim=1)       # (B, rnn_out_dim)
        max_pool, _ = rnn_out.max(dim=1)      # (B, rnn_out_dim)

        pooled = 0.5 * (mean_pool + max_pool) # (B, rnn_out_dim)

        # Projection vers embedding
        emb = self.embedding_proj(pooled)     # (B, embedding_dim)
        emb = F.normalize(emb, p=2, dim=1)    # normalisation L2, pratique pour clustering

        if self.classifier is not None:
            logits = self.classifier(emb)     # (B, num_classes)
            probs = self._softmax(logits)
            return logits, probs, emb

        return emb
