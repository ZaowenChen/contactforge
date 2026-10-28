# Copyright 2025 The Newton Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from typing import Tuple

import warp as wp

from mujoco_warp._src.collision_core import CollisionContext
from mujoco_warp._src.collision_core import Geom
from mujoco_warp._src.collision_core import contact_params
from mujoco_warp._src.collision_core import geom_collision_pair
from mujoco_warp._src.collision_core import write_contact
from .collision_ccd import ccd
from mujoco_warp._src.collision_gjk import multicontact
from mujoco_warp._src.collision_gjk import support
from mujoco_warp._src.collision_primitive import Geom
from mujoco_warp._src.collision_primitive import contact_params
from mujoco_warp._src.collision_primitive import geom_collision_pair
from mujoco_warp._src.collision_primitive import write_contact
from mujoco_warp._src.math import make_frame
from mujoco_warp._src.math import upper_trid_index
from mujoco_warp._src.types import MJ_MAX_EPAFACES
from mujoco_warp._src.types import MJ_MAX_EPAHORIZON
from mujoco_warp._src.types import MJ_MAXCONPAIR
from mujoco_warp._src.types import MJ_MAXVAL
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import EnableBit
from mujoco_warp._src.types import GeomType
from mujoco_warp._src.types import Model
from mujoco_warp._src.types import mat43
from mujoco_warp._src.types import mat63
from mujoco_warp._src.types import vec5
from mujoco_warp._src.warp_util import cache_kernel
from mujoco_warp._src.warp_util import event_scope

# TODO(team): improve compile time to enable backward pass
wp.set_module_options({"enable_backward": False})

vec_maxconpair = wp.types.vector(length=MJ_MAXCONPAIR, dtype=float)
mat_maxconpair = wp.types.matrix(shape=(MJ_MAXCONPAIR, 3), dtype=float)


@cache_kernel
def ccd_kernel_builder(
  geomtype1: int,
  geomtype2: int,
  gjk_iterations: int,
  epa_iterations: int,
  use_multiccd: bool,
  geomgeomid: int,
):
  """Kernel builder for non-heightfield CCD collisions (no hfield args)."""

  @wp.func
  def eval_ccd_write_contact(
    # Model:
    opt_ccd_tolerance: wp.array[float],
    # Data in:
    naconmax_in: int,
    # In:
    epa_vert_in: wp.array2d[wp.vec3],
    epa_vert_index_in: wp.array2d[int],
    epa_face_in: wp.array2d[int],
    epa_pr_in: wp.array2d[wp.vec3],
    epa_norm2_in: wp.array2d[float],
    epa_horizon_in: wp.array2d[int],
    multiccd_polygon_in: wp.array2d[wp.vec3],
    multiccd_clipped_in: wp.array2d[wp.vec3],
    multiccd_pnormal_in: wp.array2d[wp.vec3],
    multiccd_pdist_in: wp.array2d[float],
    multiccd_idx1_in: wp.array2d[int],
    multiccd_idx2_in: wp.array2d[int],
    multiccd_n1_in: wp.array2d[wp.vec3],
    multiccd_n2_in: wp.array2d[wp.vec3],
    multiccd_endvert_in: wp.array2d[wp.vec3],
    multiccd_face1_in: wp.array2d[wp.vec3],
    multiccd_face2_in: wp.array2d[wp.vec3],
    geom1: Geom,
    geom2: Geom,
    geoms: wp.vec2i,
    worldid: int,
    ccdid: int,
    margin: float,
    gap: float,
    condim: int,
    friction: vec5,
    solref: wp.vec2,
    solreffriction: wp.vec2,
    solimp: vec5,
    x1: wp.vec3,
    x2: wp.vec3,
    pairid: wp.vec2i,
    # Data out:
    contact_dist_out: wp.array[float],
    contact_pos_out: wp.array[wp.vec3],
    contact_frame_out: wp.array[wp.mat33],
    contact_includemargin_out: wp.array[float],
    contact_friction_out: wp.array[vec5],
    contact_solref_out: wp.array[wp.vec2],
    contact_solreffriction_out: wp.array[wp.vec2],
    contact_solimp_out: wp.array[vec5],
    contact_dim_out: wp.array[int],
    contact_geom_out: wp.array[wp.vec2i],
    contact_efc_address_out: wp.array2d[int],
    contact_worldid_out: wp.array[int],
    contact_type_out: wp.array[int],
    contact_geomcollisionid_out: wp.array[int],
    nacon_out: wp.array[int],
  ) -> int:
    points = mat43()
    witness1 = mat43()
    witness2 = mat43()
    geom1.margin = margin
    geom2.margin = margin
    is_collision_sensor = pairid[1] >= 0
    if is_collision_sensor:
      cutoff = 1.0e32
    else:
      cutoff = 0.0
    # Keep witnesses, manifold clipping and contact normals in a nearby frame.
    # Adding world positions before subtracting shallow witnesses destroys the
    # normal in float32; only final contact positions go back to world space.
    origin = 0.5 * (geom1.pos + geom2.pos)
    geom1.pos = geom1.pos - origin
    geom2.pos = geom2.pos - origin
    x1 = x1 - origin
    x2 = x2 - origin
    dist, ncollision, w1, w2, multiccd_idx = ccd(
      opt_ccd_tolerance[worldid % opt_ccd_tolerance.shape[0]],
      cutoff,
      gjk_iterations,
      epa_iterations,
      geom1,
      geom2,
      geomtype1,
      geomtype2,
      x1,
      x2,
      epa_vert_in[ccdid],
      epa_vert_index_in[ccdid],
      epa_face_in[ccdid],
      epa_pr_in[ccdid],
      epa_norm2_in[ccdid],
      epa_horizon_in[ccdid],
    )

    if dist >= 0.0 and pairid[1] == -1:
      return 0

    # CCD operates on margin-inflated shapes (support() inflates each geom by
    # 0.5 * margin).  The returned dist is therefore relative to the inflated
    # geometry.  Correct back to the true surface-to-surface distance so that
    # the constraint pipeline (pos = dist - includemargin) works consistently
    # with the primitive narrowphase, which reports un-inflated distances.
    dist += margin

    witness1[0] = w1
    witness2[0] = w2

    if wp.static(use_multiccd or (geomtype1 == GeomType.BOX and geomtype2 == GeomType.BOX)):
      if wp.static(geomtype1 == GeomType.MESH):
        # verify that geom1 mesh data is present for multicontact
        if geom1.mesh_polyadr < 0:
          multiccd_idx = -1

      if wp.static(geomtype2 == GeomType.MESH):
        # verify that geom2 mesh data is present for multicontact
        if geom2.mesh_polyadr < 0:
          multiccd_idx = -1

      if multiccd_idx > -1:
        ncollision, witness1, witness2 = multicontact(
          multiccd_polygon_in[ccdid],
          multiccd_clipped_in[ccdid],
          multiccd_pnormal_in[ccdid],
          multiccd_pdist_in[ccdid],
          multiccd_idx1_in[ccdid],
          multiccd_idx2_in[ccdid],
          multiccd_n1_in[ccdid],
          multiccd_n2_in[ccdid],
          multiccd_endvert_in[ccdid],
          multiccd_face1_in[ccdid],
          multiccd_face2_in[ccdid],
          epa_vert_in[ccdid],
          epa_vert_index_in[ccdid],
          epa_face_in[ccdid, multiccd_idx],
          w1,
          w2,
          geom1,
          geom2,
          geomtype1,
          geomtype2,
        )

    for i in range(ncollision):
      points[i] = 0.5 * (witness1[i] + witness2[i]) + origin
    normal = witness1[0] - witness2[0]
    frame = make_frame(normal)

    # flip if collision sensor
    if pairid[1] >= 0:
      frame *= -1.0
      geoms = wp.vec2i(geoms[1], geoms[0])

    nactive = int(0)  # number of contacts contributing to the physics
    for i in range(ncollision):
      active = write_contact(
        naconmax_in,
        i,
        dist,
        points[i],
        frame,
        margin,
        gap,
        condim,
        friction,
        solref,
        solreffriction,
        solimp,
        geoms,
        pairid,
        worldid,
        contact_dist_out,
        contact_pos_out,
        contact_frame_out,
        contact_includemargin_out,
        contact_friction_out,
        contact_solref_out,
        contact_solreffriction_out,
        contact_solimp_out,
        contact_dim_out,
        contact_geom_out,
        contact_efc_address_out,
        contact_worldid_out,
        contact_type_out,
        contact_geomcollisionid_out,
        nacon_out,
      )
      nactive += active

    return nactive

  # runs convex collision on a set of geom pairs to recover contact info (non-heightfield)
  @wp.kernel(module="unique", enable_backward=False)
  def ccd_kernel(
    # Model:
    opt_ccd_tolerance: wp.array[float],
    geom_type: wp.array[int],
    geom_condim: wp.array[int],
    geom_dataid: wp.array2d[int],
    geom_priority: wp.array[int],
    geom_solmix: wp.array2d[float],
    geom_solref: wp.array2d[wp.vec2],
    geom_solimp: wp.array2d[vec5],
    geom_size: wp.array2d[wp.vec3],
    geom_friction: wp.array2d[wp.vec3],
    geom_margin: wp.array2d[float],
    geom_gap: wp.array2d[float],
    mesh_vertadr: wp.array[int],
    mesh_vertnum: wp.array[int],
    mesh_graphadr: wp.array[int],
    mesh_vert: wp.array[wp.vec3],
    mesh_graph: wp.array[int],
    mesh_polynum: wp.array[int],
    mesh_polyadr: wp.array[int],
    mesh_polynormal: wp.array[wp.vec3],
    mesh_polyvertadr: wp.array[int],
    mesh_polyvertnum: wp.array[int],
    mesh_polyvert: wp.array[int],
    mesh_polymapadr: wp.array[int],
    mesh_polymapnum: wp.array[int],
    mesh_polymap: wp.array[int],
    pair_dim: wp.array[int],
    pair_solref: wp.array2d[wp.vec2],
    pair_solreffriction: wp.array2d[wp.vec2],
    pair_solimp: wp.array2d[vec5],
    pair_margin: wp.array2d[float],
    pair_gap: wp.array2d[float],
    pair_friction: wp.array2d[vec5],
    # Data in:
    geom_xpos_in: wp.array2d[wp.vec3],
    geom_xmat_in: wp.array2d[wp.mat33],
    naconmax_in: int,
    naccdmax_in: int,
    ncollision_in: wp.array[int],
    # In:
    collision_pair_in: wp.array[wp.vec2i],
    collision_pairid_in: wp.array[wp.vec2i],
    collision_worldid_in: wp.array[int],
    epa_vert_in: wp.array2d[wp.vec3],
    epa_vert_index_in: wp.array2d[int],
    epa_face_in: wp.array2d[int],
    epa_pr_in: wp.array2d[wp.vec3],
    epa_norm2_in: wp.array2d[float],
    epa_horizon_in: wp.array2d[int],
    multiccd_polygon_in: wp.array2d[wp.vec3],
    multiccd_clipped_in: wp.array2d[wp.vec3],
    multiccd_pnormal_in: wp.array2d[wp.vec3],
    multiccd_pdist_in: wp.array2d[float],
    multiccd_idx1_in: wp.array2d[int],
    multiccd_idx2_in: wp.array2d[int],
    multiccd_n1_in: wp.array2d[wp.vec3],
    multiccd_n2_in: wp.array2d[wp.vec3],
    multiccd_endvert_in: wp.array2d[wp.vec3],
    multiccd_face1_in: wp.array2d[wp.vec3],
    multiccd_face2_in: wp.array2d[wp.vec3],
    nccd_in: wp.array[int],
    # Data out:
    contact_dist_out: wp.array[float],
    contact_pos_out: wp.array[wp.vec3],
    contact_frame_out: wp.array[wp.mat33],
    contact_includemargin_out: wp.array[float],
    contact_friction_out: wp.array[vec5],
    contact_solref_out: wp.array[wp.vec2],
    contact_solreffriction_out: wp.array[wp.vec2],
    contact_solimp_out: wp.array[vec5],
    contact_dim_out: wp.array[int],
    contact_geom_out: wp.array[wp.vec2i],
    contact_efc_address_out: wp.array2d[int],
    contact_worldid_out: wp.array[int],
    contact_type_out: wp.array[int],
    contact_geomcollisionid_out: wp.array[int],
    nacon_out: wp.array[int],
  ):
    collisionid = wp.tid()
    if collisionid >= ncollision_in[0]:
      return

    geoms = collision_pair_in[collisionid]
    g1 = geoms[0]
    g2 = geoms[1]

    if geom_type[g1] != geomtype1 or geom_type[g2] != geomtype2:
      return

    ccdid = wp.atomic_add(nccd_in, wp.static(geomgeomid), 1)
    if ccdid >= naccdmax_in:
      wp.printf("CCD overflow - please increase naccdmax to %u\n", ccdid)
      return

    worldid = collision_worldid_in[collisionid]

    _, margin, gap, condim, friction, solref, solreffriction, solimp = contact_params(
      geom_condim,
      geom_priority,
      geom_solmix,
      geom_solref,
      geom_solimp,
      geom_friction,
      geom_margin,
      geom_gap,
      pair_dim,
      pair_solref,
      pair_solreffriction,
      pair_solimp,
      pair_margin,
      pair_gap,
      pair_friction,
      collision_pair_in,
      collision_pairid_in,
      collisionid,
      worldid,
    )

    geom1, geom2 = geom_collision_pair(
      geom_type,
      geom_dataid,
      geom_size,
      mesh_vertadr,
      mesh_vertnum,
      mesh_graphadr,
      mesh_vert,
      mesh_graph,
      mesh_polynum,
      mesh_polyadr,
      mesh_polynormal,
      mesh_polyvertadr,
      mesh_polyvertnum,
      mesh_polyvert,
      mesh_polymapadr,
      mesh_polymapnum,
      mesh_polymap,
      geom_xpos_in,
      geom_xmat_in,
      geoms,
      worldid,
    )

    eval_ccd_write_contact(
      opt_ccd_tolerance,
      naconmax_in,
      epa_vert_in,
      epa_vert_index_in,
      epa_face_in,
      epa_pr_in,
      epa_norm2_in,
      epa_horizon_in,
      multiccd_polygon_in,
      multiccd_clipped_in,
      multiccd_pnormal_in,
      multiccd_pdist_in,
      multiccd_idx1_in,
      multiccd_idx2_in,
      multiccd_n1_in,
      multiccd_n2_in,
      multiccd_endvert_in,
      multiccd_face1_in,
      multiccd_face2_in,
      geom1,
      geom2,
      geoms,
      worldid,
      ccdid,
      margin,
      gap,
      condim,
      friction,
      solref,
      solreffriction,
      solimp,
      geom1.pos,
      geom2.pos,
      collision_pairid_in[collisionid],
      contact_dist_out,
      contact_pos_out,
      contact_frame_out,
      contact_includemargin_out,
      contact_friction_out,
      contact_solref_out,
      contact_solreffriction_out,
      contact_solimp_out,
      contact_dim_out,
      contact_geom_out,
      contact_efc_address_out,
      contact_worldid_out,
      contact_type_out,
      contact_geomcollisionid_out,
      nacon_out,
    )

  return ccd_kernel


