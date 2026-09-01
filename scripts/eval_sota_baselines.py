import os, sys, json, torch
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.metrics import roc_auc_score

print('=== Evaluating SOTA Pretrained Baselines: DDA & AIDE ===')
