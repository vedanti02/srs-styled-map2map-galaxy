"""Cluster-specific storage locations, in one place.

Defaults point at NCSA DeltaAI. Override with env vars (scripts/env_delta.sh sets them) to run elsewhere.
Layout under CMASS_ROOT mirrors the old babel /data/.../cmass-ili tree:
  processed/{k:04d}_{input,label}.npy     LR (CHARM) / HR (Quijote) count fields, field (string-sort) order
  models/<tag>/best.pt                    was /data/user_data/vkshirsa/cmass-ili/models/<tag>
  checkpoints/<tag>/best.pt               was the repo's checkpoints/ symlink target
  sr_fields/<tag>/sr_{idx}.npy            generated SR fields
  quijote/nbody/L1000-N128/               Quijote configs/halos (NOT on Delta; only needed to rebuild theta)
"""
import os

CMASS_ROOT = os.environ.get("SRS_CMASS_ROOT", "/work/nvme/bdne/vkshirsagar1/cmass-ili")
PROCESSED = os.environ.get("SRS_PROCESSED", os.path.join(CMASS_ROOT, "processed"))
MODELS = os.path.join(CMASS_ROOT, "models")
SR_FIELDS = os.path.join(CMASS_ROOT, "sr_fields")
NBODY = os.path.join(CMASS_ROOT, "quijote", "nbody", "L1000-N128")
