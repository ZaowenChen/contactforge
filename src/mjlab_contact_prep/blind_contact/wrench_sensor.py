"""V6-only F/T reconstruction fix for the pinned MuJoCo-Warp runtime.

This runtime allocates each pyramidal contact row independently with atomic_add.
Its contact_force_fn nevertheless decodes rows as a contiguous block starting
at efc_address[contact, 0]. Read every recorded address instead: otherwise a
neighbour's constraint force is used when GPU row allocation interleaves.
Only sensor/contact-force reconstruction changes; the solver is untouched.
Installed files and the frozen revision-4 execution path are never modified.
"""
import hashlib
from importlib.metadata import version
from pathlib import Path
import warp as wp
from mujoco_warp._src.types import ConeType, vec5

_installed = False


@wp.func
def contact_force_by_address(
    opt_cone: int,
    frame: wp.array[wp.mat33], friction: wp.array[vec5], dim: wp.array[int],
    address: wp.array2d[int], efc_force: wp.array2d[float], njmax: int, nacon: wp.array[int],
    world: int, contact: int, to_world_frame: bool,
) -> wp.spatial_vector:
    result = wp.spatial_vector(0., 0., 0., 0., 0., 0.)
    if contact < 0 or contact >= nacon[0]:
        return result
    count = dim[contact]
    if opt_cone == ConeType.PYRAMIDAL and count > 1:
        for i in range(count-1):
            a = address[contact, 2*i]
            b = address[contact, 2*i+1]
            plus = float(0.)
            minus = float(0.)
            if a >= 0 and a < njmax:
                plus = efc_force[world, a]
            if b >= 0 and b < njmax:
                minus = efc_force[world, b]
            result[0] += plus+minus
            result[i+1] = (plus-minus)*friction[contact][i]
    else:
        for i in range(count):
            a = address[contact, i]
            if a >= 0 and a < njmax:
                result[i] = efc_force[world, a]
    if to_world_frame:
        result = wp.spatial_vector(wp.spatial_top(result)@frame[contact], wp.spatial_bottom(result)@frame[contact])
    return result


def configure_v6_wrench_sensor():
    global _installed
    if _installed:
        return
    from mujoco_warp._src import support, constraint
    expected = {
        "support": "5129be54092602ec72faa5f0260e369f3f5164eea8cdb3dbb6a2697243e7b9f6",
        "constraint": "1f90b0620633cdedf6bf2997b7fad412d956ee996a30d72688a10fed539cf68d",
    }
    if version("mujoco-warp") != "3.7.0.1":
        raise RuntimeError("re-audit the v6 contact-force addressing fix for this MuJoCo-Warp version")
    for name, module in (("support", support), ("constraint", constraint)):
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != expected[name]:
            raise RuntimeError(f"MuJoCo-Warp {name}.py differs from the audited v6 sensor runtime")
    support.contact_force_fn = contact_force_by_address
    _installed = True
