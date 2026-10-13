"""Adapter exposing the K562 encoder model with the Keras-style `.predict`
interface the vendored two-stage agent expects.

The agent calls `model.predict(x, verbose=0)[head_index]` on (B, 200, 4) input
and reads `[head][0, 0]` / `[head].flatten()`. K562Model instead takes just the
200bp variable region plus that sequence's held-WT left/right adapters. One
adapter is bound per sequence (its own left/right), head_index is always 0.
"""

import numpy as np


class K562Adapter:
    def __init__(self, model, left, right):
        self.model = model          # K562Model (predict_var)
        self.left = left            # (15,4) held-WT left adapter for this seq
        self.right = right          # (15,4) held-WT right adapter for this seq

    def predict(self, x, verbose=0):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 2:             # (L,4) -> (1,L,4)
            x = x[np.newaxis]
        preds = self.model.predict_var(x, self.left, self.right)   # (N,)
        return [preds.reshape(-1, 1)]                              # head 0 -> (N,1)
