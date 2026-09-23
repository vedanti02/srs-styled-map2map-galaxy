"""Standalone conditional flow-matching corrector LF -> HF for CMASS halo-count patches.

No adversarial term, no GAN encoder. Same inputs/outputs/patching/eval as the GAN
(train_patch_cmass.py), so it is a like-for-like ablation of the training objective.
See flow_matching/README.md.
"""
