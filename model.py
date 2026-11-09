"""The model behind the demo.

Contract used by app.py:
  MODES            {label: mode} shown in the input-type selector
  GAUSSIANS        {label: count} shown in the Gaussian-count dropdown
  DEFAULTS         default settings
  generate(...)    -> (path of the .ply, the image the model saw), called on the GPU worker
"""
import os
import tempfile
import uuid

from idol import IdolPipeline

MODES = {'Synthetic': 'synthetic', 'Real photo': 'real'}
GAUSSIANS = {'65,536': 65536, '131,072': 131072, '262,144': 262144, '524,288': 524288, '1,048,576': 1048576}
DEFAULTS = {'steps': 50, 'cfg': 3.0, 'rescale_t': 3.0, 'gaussians': '262,144', 'seed': 42}

# Weights load once at import; on ZeroGPU the CUDA placement is deferred to the GPU worker.
pipeline = IdolPipeline.from_pretrained(os.environ.get('IDOL_WEIGHTS', 'speridlabs/idol'), device='cuda')
OUTPUT_DIR = tempfile.mkdtemp(prefix='idol-')


def generate(image, mode, steps, cfg_strength, rescale_t, seed, num_gaussians):
    output = pipeline(image, mode=mode, seed=seed, steps=steps, cfg_strength=cfg_strength, rescale_t=rescale_t,
                      num_gaussians=num_gaussians)
    path = os.path.join(OUTPUT_DIR, f'idol-seed{seed}-{num_gaussians}-{uuid.uuid4().hex[:8]}.ply')
    output.gaussian.save_ply(path)
    return path, output.preprocessed_image
