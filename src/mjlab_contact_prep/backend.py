"""Process-local, version-pinned convex contact fix; never edit site-packages."""
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

_selected = None


def configure_backend(name):
    global _selected
    if name not in ('stock', 'contact-fix'):
        raise ValueError(f'Unknown physics backend: {name}')
    if _selected is not None:
        if name != _selected:
            raise RuntimeError('Use a fresh process to change collision backends (Warp caches kernels).')
        return
    if name == 'contact-fix':
        from mujoco_warp._src import collision_convex
        vendor = Path(__file__).with_name('_vendor')
        provenance = json.loads((vendor / 'provenance.json').read_text())
        actual = hashlib.sha256(Path(collision_convex.__file__).read_bytes()).hexdigest()
        if (version('mujoco-warp') != provenance['compatible_mujoco_warp']
                or actual != provenance['collision_convex_sha256']):
            raise RuntimeError('Contact fix requires the audited MuJoCo-Warp 3.7.0.1 runtime; revalidate before upgrading.')
        for filename, key in [('collision_ccd.py', 'vendor_sha256'), ('collision_kernel.py', 'kernel_sha256')]:
            if hashlib.sha256((vendor / filename).read_bytes()).hexdigest() != provenance[key]:
                raise RuntimeError(f'Unrecorded collision fix change: {filename}; update provenance and revalidate.')
        from ._vendor.collision_ccd import ccd
        # Select the CCD algorithm and pair-local narrowphase before kernel capture.
        # Contact material, manifold algorithm, constraints and solver stay native.
        from ._vendor.collision_kernel import ccd_kernel_builder
        collision_convex.ccd = ccd
        collision_convex.ccd_kernel_builder = ccd_kernel_builder
    _selected = name


def backend_metadata(name):
    result = {'name': name, 'mujoco_warp': version('mujoco-warp')}
    if name == 'contact-fix':
        vendor = Path(__file__).with_name('_vendor')
        result.update(json.loads((vendor / 'provenance.json').read_text()))
        result['loaded_kernel_sha256'] = hashlib.sha256((vendor / 'collision_kernel.py').read_bytes()).hexdigest()
        result['loaded_vendor_sha256'] = hashlib.sha256((vendor / 'collision_ccd.py').read_bytes()).hexdigest()
    return result
