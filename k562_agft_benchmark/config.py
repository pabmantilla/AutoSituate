"""Shared paths and constants for the K562 AlphaGenome-encoder annotation benchmark.

Applies AutoSituate's basic annotation methods, systematically, over the existing
SEAM_validate single-sequence foregrounds — scored with motif knock-out (necessity)
and knock-in (sufficiency) against the finetuned K562 encoder-only model.

No ISM anywhere: attributions are the pre-computed SEAM foregrounds; the only model
calls are forward predictions for the KO / KI swaps.
"""

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = Path("/grid/koo/home/pmantill/projects")
SEAM = PROJ / "SEAM_validate"

# --- inputs -----------------------------------------------------------------
FOREGROUNDS = SEAM / "results/single_seq/foregrounds"
FOREGROUND_MAP = "foreground_scaled.npy"          # the map we annotate (Pablo's pick)
SEED_CSVS = [                                     # union gives WT seqs for 84/85 dirs
    SEAM / "single_seq/seq_subsets/seam_seeds.csv",
    SEAM / "single_seq/seq_subsets/seam_seeds_crepool.csv",
]
SKIP = {"rs536864738_WT"}                         # no WT seq row -> not KO/KI-able

# K562 finetuned encoder-only model (same ckpt/recipe as SEAM_ism.py / predict_test_set.py)
EI_DIR = PROJ / "Virtual_Experiments/Hippo_axis/Hippo_dependency_mpra/eigen-interactions"
CKPT = os.environ.get(
    "SEAM_CKPT",
    "/grid/koo/home/shared/models/alphagenome_encoder/torch/mpra_K562/finetuned_encoder.pt",
)
JASPAR_MEME = EI_DIR / "motif_db/JASPAR2026_vertebrates.meme"

# --- sequence geometry (matches SEAM_ism.py) --------------------------------
ENH_LEN = 230                   # insert length
VAR_START, VAR_END = 15, 215    # 200bp variable/attribution window; adapters held WT
L_VAR = VAR_END - VAR_START     # 200

ALPHA = {"A": 0, "C": 1, "G": 2, "T": 3}

# --- KO/KI protocol ---------------------------------------------------------
NUM_SHUFFLES = 100              # dinuc shuffles of the FULL 200bp variable region
SEED = 42
NULL_DRAWS = 100                # window-matched null placements (the null distribution)

# --- TF-MoDISco discovery variants ------------------------------------------
# Default min_metacluster_size=100 finds 0 patterns on this 84-seq panel (only ~83
# positive seqlets). Run two lowered variants as separate methods until there are
# enough sequences to use the default.
MODISCO_MIN_METACLUSTERS = [10]

# --- outputs ----------------------------------------------------------------
RESULTS = HERE / "results"
