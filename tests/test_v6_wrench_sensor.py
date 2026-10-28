"""Regression: contact pyramid rows may interleave with another contact."""
import unittest
import numpy as np
import warp as wp
from mujoco_warp._src.types import vec5
from mjlab_contact_prep.blind_contact.wrench_sensor import contact_force_by_address


@wp.kernel
def decode(frame: wp.array[wp.mat33], friction: wp.array[vec5], dim: wp.array[int],
           address: wp.array2d[int], forces: wp.array2d[float], count: wp.array[int],
           out: wp.array[wp.spatial_vector]):
    i = wp.tid()
    out[i] = contact_force_by_address(0, frame, friction, dim, address, forces, 8, count, 0, i, False)


class WrenchSensorTests(unittest.TestCase):
    def test_interleaved_pyramids_and_invalid_contact(self):
        # Contact 0: (10,2), (7,1), mu .5/.25 -> F=(20,4,1.5).
        # Rows 3,4,6 belong to another contact and must not enter this force.
        wp.init()
        values = np.array([[0,10,7,900,800,2,700,1]], dtype=np.float32)
        frames = wp.array(np.tile(np.eye(3,dtype=np.float32),(2,1,1)), dtype=wp.mat33, device='cpu')
        friction = wp.array(np.array([[.5,.25,0,0,0]]*2,dtype=np.float32), dtype=vec5, device='cpu')
        dims = wp.array([3,3], dtype=int, device='cpu')
        address = wp.array(np.array([[1,5,2,7],[-1,-1,-1,-1]],dtype=np.int32), dtype=int, device='cpu')
        count = wp.array([1], dtype=int, device='cpu')
        out = wp.zeros(2,dtype=wp.spatial_vector,device='cpu')
        wp.launch(decode,dim=2,inputs=[frames,friction,dims,address,wp.array(values,device='cpu'),count],outputs=[out],device='cpu')
        np.testing.assert_allclose(out.numpy(),[[20,4,1.5,0,0,0],[0,0,0,0,0,0]],atol=1e-6)


if __name__ == '__main__':
    unittest.main()
