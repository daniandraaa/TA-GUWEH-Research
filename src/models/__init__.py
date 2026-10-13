"""Definisi arsitektur yang dipakai bersama oleh kedua skenario."""
from .transformer_encoder import CNNTransformerEncoder
from .patchtst import CNNPatchTST
from .hierarchical_transformer import CNNHierarchicalTransformer

MODELS = {
    "encoder": CNNTransformerEncoder,
    "patchtst": CNNPatchTST,
    "hierarchical": CNNHierarchicalTransformer,
}
MODEL_NAMES = {
    "encoder": "CNN-Transformer Encoder",
    "patchtst": "CNN-PatchTST",
    "hierarchical": "CNN-Hierarchical Transformer",
}


def build_model(name, config):
    return MODELS[name](config)
