import math
from dataclasses import replace
from pathlib import Path
import unittest
import torch
from mjlab_contact_prep.blind_contact.config import BlindConfig
from mjlab_contact_prep.blind_contact.study_environment import study_events
from mjlab_contact_prep.blind_contact.cartesian_environment import chamfer_allowance, circle_events, polygon_events
from mjlab_contact_prep.blind_contact.cartesian_control import so3_exp, so3_log, comply, norm_limit
from mjlab_contact_prep.blind_contact.cartesian_networks import CartesianActor, as_tensordict_v6, pose_loss
from mjlab_contact_prep.blind_contact.polygon_geometry import PolygonGeometry

ROOT = Path(__file__).resolve().parents[1]
RADIUS = .017451159417


def ring(center, radius=RADIUS, count=64):
    a = torch.arange(count)*2*math.pi/count
    # Clockwise in the hole XY plane: peg local +Z points down into the hole.
    pts = torch.stack((radius*a.cos(), -radius*a.sin(), torch.zeros_like(a)), -1)
    return (pts+torch.tensor(center))[None]


class V6Tests(unittest.TestCase):
    def setUp(self):
        self.a = BlindConfig.load(ROOT/"configs/blind_v6_cartesian_ft_A_05mm.json")
        self.b = BlindConfig.load(ROOT/"configs/blind_v6_cartesian_ft_B_005mm.json")

    def test_configs_differ_only_in_declared_curriculum(self):
        da, db = self.a.to_dict(), self.b.to_dict()
        self.assertEqual({k for k in da if da[k] != db[k]}, {"clearance_per_side_m", "seed", "learning_rate"})
        from mjlab_contact_prep.blind_contact.clearance_validation import check_initialization_contract
        check_initialization_contract(da, self.b)
        with self.assertRaises(ValueError):
            replace(self.a, asset_variant="rigid", arm_gravity_compensation=False,
                    joint_frictionloss_scale=1., joint_armature_scale=1.).validate()

    def test_default_job_has_only_final_evaluation(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("v6_job_schedule", ROOT / "scripts/blind_study_job.py")
        job = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(job)
        for cfg in (self.a, self.b):
            budget, evaluations = job.resolve_budget(cfg.to_dict())
            self.assertEqual(budget, 393216000)
            stages = job.stage_plan(cfg.to_dict(), budget, evaluations)
            self.assertEqual(len(stages), 1)
            self.assertEqual(stages[0]["end_iteration"], 1000)

    def test_chamfer_allowance_profile(self):
        z = torch.tensor([.001, 0., -.0005, -.001, -.02, -.0395, -.03975, -.04, -.041], dtype=torch.float64)
        expected = torch.tensor([.001, .001, .0005, 0., 0., 0., .00025, .0005, .0005], dtype=torch.float64)
        torch.testing.assert_close(chamfer_allowance(z), expected, atol=1e-9, rtol=0)
        # Float32 subtracts two ~40 mm values. Its rounding budget is separate
        # from the unchanged 25 um physical legality tolerance.
        torch.testing.assert_close(chamfer_allowance(z.float()), expected.float(), atol=5e-9, rtol=0)

    def test_exit_chamfer_seat_rejected_by_v4_and_accepted_by_v6(self):
        # A legal 40 mm seat with the rim 5 um outside the straight bore while
        # the rim lies inside the exit chamfer (the observed revision-4 miss).
        cfg = replace(self.a, study_revision=4)
        points = ring([cfg.clearance_per_side_m+5e-6, 0., -.0399])
        rotation = torch.diag(torch.tensor([1., -1., -1.]))[None]
        error = torch.tensor([-1e-5])
        _, seated_v4 = study_events(points, RADIUS, cfg, error)
        _, seated_v6 = circle_events(points, RADIUS, rotation, error, self.a)
        self.assertFalse(bool(seated_v4[0]))
        self.assertTrue(bool(seated_v6[0]))
        # Outside by more than chamfer + tolerance is still rejected.
        far = ring([cfg.clearance_per_side_m+.0006, 0., -.0399])
        self.assertFalse(bool(circle_events(far, RADIUS, rotation, error, self.a)[1][0]))
        # An illegal whole shaft is rejected regardless of the rim.
        self.assertFalse(bool(circle_events(points, RADIUS, rotation, torch.tensor([1e-4]), self.a)[1][0]))
        # Straight-bore capture keeps only the 25 um tolerance.
        shallow = ring([cfg.clearance_per_side_m+3e-5, 0., -.002])
        self.assertFalse(bool(circle_events(shallow, RADIUS, rotation, error, self.a)[0][0]))

    def test_polygon_exit_chamfer_seat(self):
        geometry = PolygonGeometry("square", .0346, .00005, "cpu")
        points = geometry.points[None].clone()
        points[..., 0] += .00005+5e-6
        points[..., 2] = -.0399
        rotation = torch.diag(torch.tensor([1., -1., -1.]))[None]
        error = torch.tensor([-1e-5])
        cfg = replace(self.b, study_revision=4)
        self.assertFalse(bool(geometry.events(points, rotation, error, cfg)[1][0]))
        self.assertTrue(bool(polygon_events(geometry, points, rotation, error, self.b)[1][0]))

    def test_seating_transfer_is_explicit_and_v4_only(self):
        from mjlab_contact_prep.blind_contact.seating_ablation import seating_transfer_contract
        source = replace(self.a, study_revision=4, seating_protocol="legacy_v4").to_dict()
        target = replace(self.a, study_revision=4)
        with self.assertRaises(ValueError):
            seating_transfer_contract(source, target, False)
        self.assertTrue(seating_transfer_contract(source, target, True)["seating_only_ablation"])
        with self.assertRaises(ValueError):
            seating_transfer_contract(source, self.a, True)

    def test_through_hole_and_tilted_rim_are_not_seated(self):
        rotation = torch.diag(torch.tensor([1., -1., -1.]))[None]
        for z in (-.002, -.0389, -.0411, -.06):
            self.assertFalse(bool(circle_events(ring([0., 0., z]), RADIUS, rotation,
                                               torch.zeros(1), self.a)[1][0]))
        points = ring([0., 0., -.04])
        points[0, 0, 2] = -.0389
        self.assertFalse(bool(circle_events(points, RADIUS, rotation, torch.zeros(1), self.a)[1][0]))

    def test_governor_direction_on_all_load_axes(self):
        for axis in range(3):
            for sign in (-1., 1.):
                load = torch.zeros(1, 3)
                load[0, axis] = sign*30
                lead = -load/10000
                unloaded = comply(torch.zeros_like(lead), load, 20., 5., 2e-6)
                self.assertGreater(float(unloaded[0, axis])*sign, 0)
                blocked = comply(lead, load, 20., 5., 2e-6)
                self.assertGreaterEqual(float(blocked[0, axis])*sign, 0)

    def test_so3_maps_round_trip(self):
        torch.manual_seed(0)
        w = torch.randn(64, 3)*.3
        torch.testing.assert_close(so3_log(so3_exp(w)), w, atol=1e-5, rtol=0)
        r = so3_exp(w)
        torch.testing.assert_close(r@r.transpose(1, 2), torch.eye(3).expand(64, 3, 3), atol=1e-5, rtol=0)
        torch.testing.assert_close(so3_exp(torch.zeros(2, 3)), torch.eye(3).expand(2, 3, 3))

    def test_governor_stops_pushing_and_retreats_under_load(self):
        lead = torch.tensor([[0., 0., .002]])          # pushing down the peg axis
        threshold = torch.tensor([10., 10., 20.])
        unloaded = comply(lead, torch.tensor([[0., 0., -5.]]), threshold, 5., 2e-6)
        torch.testing.assert_close(unloaded, lead)
        loaded = comply(lead, torch.tensor([[0., 0., -30.]]), threshold, 5., 2e-6)
        self.assertLessEqual(float(loaded[0, 2]), 0.)   # pushing removed, retreat applied
        lateral = comply(torch.tensor([[.001, 0., 0.]]), torch.tensor([[12., 0., 0.]]), threshold, 5., 2e-6)
        self.assertGreater(float(lateral[0, 0]), .001)  # moving with the load is not reduced
        torch.testing.assert_close(norm_limit(torch.tensor([[.003, .004, 0.]]), .001).norm(), torch.tensor(.001))

    def test_actor_is_isolated_from_privileged_inputs(self):
        n = 8
        obs = dict(history=torch.randn(n, 30, 36), critic=torch.randn(n, 248),
                   labels=torch.randint(0, 3, (n, 64)), pose=torch.randn(n, 7))
        td = as_tensordict_v6(obs)
        actor = CartesianActor(td)
        actor.refresh_normalization(td)
        before = actor(td).clone()
        other = td.clone()
        other["critic"][:] = 999.
        other["pose"][:] = 5.
        other["labels"][:] = 2
        torch.testing.assert_close(actor(other), before, atol=0, rtol=0)
        logits, pose = actor.auxiliary(td)
        self.assertEqual(tuple(logits.shape), (n, 64, 3))
        self.assertEqual(tuple(pose.shape), (n, 6))
        target = torch.zeros(n, 7)
        target[:, 5] = .5
        self.assertEqual(float(pose_loss(torch.zeros(n, 6), target)), 0.)   # circle yaw masked
        target[:, 6] = 1.
        self.assertGreater(float(pose_loss(torch.zeros(n, 6), target)), 0.)

    def test_pose_yaw_is_periodic(self):
        target = torch.zeros(1, 7)
        target[:, 5:] = torch.tensor([.99, 1.])
        prediction = torch.zeros(1, 6)
        prediction[:, 5] = -.99
        self.assertAlmostEqual(float(pose_loss(prediction, target)), .02**2/6, places=8)

    def test_load_fault_wins_over_same_sample_seating(self):
        from mjlab_contact_prep.blind_contact.cartesian_environment import terminal_flags
        running = torch.tensor([True, True, True, True, False])
        seated = torch.ones(5, dtype=torch.bool)
        hit, fault, timeout, code = terminal_flags(seated, torch.tensor([0., 51., 0., 51., 99.]),
            torch.tensor([0., 0., 2.1, 2.1, 9.]), torch.zeros(5, dtype=torch.bool),
            torch.ones(5, dtype=torch.bool), running, self.a)
        self.assertEqual(hit.tolist(), [True, False, False, False, False])
        self.assertEqual(code.tolist(), [0, 1, 2, 3, 0])
        self.assertFalse(timeout.any())

    def test_potential_acts_only_after_capture_and_telescopes(self):
        from mjlab_contact_prep.blind_contact.cartesian_environment import alignment_potential
        tilt = torch.tensor([.01, .02, .005])
        yaw = torch.tensor([.02, -.01, .0])
        self.assertFalse(alignment_potential(tilt, yaw, torch.zeros(3, dtype=torch.bool), self.a).any())
        phi = alignment_potential(tilt, yaw, torch.ones(3, dtype=torch.bool), self.a)
        sequence = torch.cat((phi, torch.zeros(1)))
        rewards = self.a.gamma*sequence[1:]-sequence[:-1]
        discount = self.a.gamma**torch.arange(3)
        torch.testing.assert_close((rewards*discount).sum(), -sequence[0])


if __name__ == "__main__":
    unittest.main()
