"""Finetuned K562 AlphaGenome encoder-only model wrapper.

Load + forward recipe mirrors SEAM_validate/single_seq/scripts/predict_test_set.py:
    - EncoderMPRAModel.from_checkpoint(CKPT, device="cuda")
    - each 230bp insert is padded with the 51bp construct suffix (36bp promoter +
      15bp barcode) -> 281bp before scoring
    - model(x) with x shape (B, 281, 4) -> scalar activity per sequence

`predict_var` takes just the 200bp variable region (what the KO/KI swaps modify)
and stitches the fixed adapters + construct suffix back on.
"""

import sys

import numpy as np
import torch

import config as C
import data

sys.path.insert(0, str(C.EI_DIR))
from alphagenome_encoder_ft import EncoderMPRAModel  # noqa: E402


class K562Model:
    def __init__(self, device="cuda"):
        ckpt = torch.load(C.CKPT, map_location="cpu", weights_only=False)
        cc = ckpt["construct_config"]
        suffix = cc["promoter_seq"] + cc["barcode_seq"]
        assert len(suffix) == 51, f"construct suffix is {len(suffix)}bp, expected 51"
        self.suffix = data.onehot(suffix)                      # (51,4)
        self.device = device
        self.model = EncoderMPRAModel.from_checkpoint(C.CKPT, device=device).eval()

    @torch.no_grad()
    def predict_var(self, var_batch, left, right, batch_size=512):
        """var_batch (N,200,4), left (15,4), right (15,4) -> (N,) activity.

        Rebuilds the full 281bp construct: left | var | right | suffix.
        """
        n = var_batch.shape[0]
        left = np.broadcast_to(left, (n, C.VAR_START, 4))
        right = np.broadcast_to(right, (n, C.ENH_LEN - C.VAR_END, 4))
        suffix = np.broadcast_to(self.suffix, (n, 51, 4))
        full = np.concatenate([left, var_batch, right, suffix], axis=1).astype(np.float32)
        assert full.shape[1] == C.ENH_LEN + 51  # 281
        out = np.empty(n, dtype=np.float32)
        for i in range(0, n, batch_size):
            xb = torch.from_numpy(full[i:i + batch_size]).to(self.device)
            out[i:i + batch_size] = self.model(xb).squeeze(-1).detach().cpu().numpy()
        return out
