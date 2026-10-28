"""Controller probes in Cartesian action units; no learned policy or IK actions."""
from dataclasses import replace
from pathlib import Path
import math
import time
import numpy as np
import torch
from .assets import NOMINAL_TIP, solve_ik
from .cartesian_environment import make_cartesian_shape_environment
from .training import freeze_run, save_json


def cartesian_control_probe(folder, cfg):
    torch.set_num_threads(4)
    if len(cfg.assembly_shapes) > 1:
        return {shape:cartesian_control_probe(Path(folder)/shape, replace(cfg, assembly_shapes=[shape]))
                for shape in cfg.assembly_shapes}
    cfg = replace(cfg, num_envs=5, reset_bank_size=16, tilt_deg=0., fixture_tilt_deg=0.)
    env = make_cartesian_shape_environment(cfg, auto_reset=False)
    q = solve_ik(env.model, NOMINAL_TIP, env.nominal_rotation)
    env.data.qpos[:] = torch.tensor(q, device=env.device, dtype=torch.float32)
    env.data.qvel[:] = 0
    env.data.mocap_pos[:, env.hole_mocap] = torch.tensor([*NOMINAL_TIP[:2], .1], device=env.device)
    env.data.mocap_pos[:, env.hole_mocap, 0] += torch.tensor([0., .002, .020, .15, .15], device=env.device)
    env.data.mocap_quat[:, env.hole_mocap] = torch.tensor([1., 0., 0., 0.], device=env.device)
    env.sim.forward()
    env.control.reset(torch.arange(5, device=env.device), env.data)
    tool = env.read_wrench()
    for name in ("wrench_tool", "filtered_wrench", "interval_mean_wrench"):
        getattr(env, name).copy_(tool)
    env.history[:] = env.proprioception()[:, None]
    labels = ["aligned", "edge_2mm", "plate_20mm", "free_hold", "free_motion"]
    folder = Path(folder); freeze_run(folder, cfg, env, "cartesian_control_probe")
    rows, trace = {}, []
    started = time.monotonic()
    with torch.no_grad():
        for step in range(math.ceil(cfg.horizon_s/cfg.policy_dt)):
            # Tool +Z points into the plate. These are Cartesian increments,
            # with no joint-space inversion or privileged force cancellation.
            speed = .003 if step*cfg.policy_dt < 1.5 else .016
            p, r = env.control.pose(env.data)
            lead_z = (r.transpose(1, 2)@(env.control.ref_pos-p)[:, :, None]).squeeze(-1)[:, 2]
            mapped = torch.zeros(5, 6, device=env.device)
            mapped[:, 2] = (.5*lead_z+speed*cfg.policy_dt)/cfg.cart_translation_step_m
            mapped[3] = 0
            _, _, done, info = env.step(torch.atanh(mapped.clamp(-.999, .999)))
            trace.append(torch.stack((info["depth"], info["shaft_error"], info["force_peak_n"],
                                      info["moment_peak_nm"], info["elapsed_s"], info["fault_code"].float()), -1).cpu().numpy())
            for i in done.nonzero().flatten().tolist():
                rows[i] = dict(case=labels[i], success=bool(info["success"][i]),
                               geometry_success=bool(info["geometry_success"][i]),
                               depth_m=float(info["depth"][i]), shaft_error_m=float(info["shaft_error"][i]),
                               force_peak_n=float(info["force_peak_n"][i]), moment_peak_nm=float(info["moment_peak_nm"][i]),
                               force_exceed_s=float(info["force_exceed_s"][i]), moment_exceed_s=float(info["moment_exceed_s"][i]),
                               fault_code=int(info["fault_code"][i]), elapsed_s=float(info["elapsed_s"][i]))
            if not env.active.any():
                break
    assert len(rows) == 5
    np.savez_compressed(folder/"trajectory.npz", states=np.stack(trace),
                        columns=np.array(["depth_m", "shaft_error_m", "force_peak_n", "moment_peak_nm", "elapsed_s", "fault_code"]))
    report = dict(cases=[rows[i] for i in range(5)], controller_only=True, learned_policy=False,
                  action_units="tool-frame translation/rotation; atanh of bounded Cartesian increment",
                  force_sample_hz=1/cfg.physics_dt, trajectory_sample_hz=1/cfg.policy_dt,
                  terminal_detection_hz=1/cfg.physics_dt, wall_s=time.monotonic()-started,
                  max_contacts=int(env.max_contacts[0]), max_constraints=int(env.max_constraints[0]),
                  buffer_overflow=bool(env.buffer_overflow[0]))
    save_json(folder/"results.json", report); print(report, flush=True)
    env.close()
    return report


def transition_checks(env):
    """Inject known measurements/events into the real 1 ms transition loop.

    These verify timing and priority, not physical seating or force accuracy.
    The independent Cartesian probes exercise the unmodified physical inputs.
    """
    group = env.envs[0]
    state, auto = group.snapshot(), group.auto_reset
    read, sample = group.read_wrench, group._sample
    cancel = group.control.cancel_reference
    group.auto_reset = False
    counter = dict(wrench=0, sample=0, cancelled=0)
    first_q = []
    def measured():
        tool = read()
        tool[:5] = 0
        if counter['wrench'] < 2:
            tool[1, 2] = -51.
            tool[2, 3] = 2.1
            tool[3, 2] = -51.
        else:
            # Post-terminal spikes must not overwrite the terminal episode.
            tool[:4, 2] = -1000.
        counter['wrench'] += 1
        return tool
    def geometry():
        values = sample()
        values['far'][:5] = False
        values['seated'][:5] = False
        values['capture_now'][:5] = False
        if counter['sample'] == 0:
            values['seated'][:3] = True
            values['capture_now'][:3] = True
            values['depth'][:3] = .04
            values['shaft_error'][:3] = 0.
            first_q.append(group.data.qpos[:5].clone())
        counter['sample'] += 1
        return values
    def cancelled(data, mask):
        cancel(data, mask)
        p, r = group.control.pose(data)
        torch.testing.assert_close(group.control.ref_pos[mask], p[mask], atol=0, rtol=0)
        torch.testing.assert_close(group.control.ref_rot[mask], r[mask], atol=0, rtol=0)
        counter['cancelled'] += int(mask[:5].sum())
    try:
        group.read_wrench, group._sample = measured, geometry
        group.control.cancel_reference = cancelled
        group.physics_steps[4] = math.ceil(group.cfg.horizon_s/group.cfg.physics_dt)-1
        _, reward, done, info = group.step(torch.ones(group.num_envs, 6, device=group.device)*.25)
        assert done[:5].all()
        assert info['success'][:5].tolist() == [True, False, False, False, False]
        assert info['fault_code'][:5].tolist() == [0, 1, 2, 1, 0]
        assert info['timeout'][:5].tolist() == [False, False, False, False, True]
        torch.testing.assert_close(info['elapsed_s'][:4], torch.full((4,), .001, device=group.device))
        torch.testing.assert_close(info['trajectory_qpos'][:5], first_q[0], atol=0, rtol=0)
        assert info['geometry_success'][:3].all()
        assert float(info['force_peak_n'][1]) == 51.
        assert float(info['force_peak_n'][0]) == 0.
        assert info['force_exceed_s'][1] == group.cfg.physics_dt
        assert counter['cancelled'] == 5
        # A second action cannot revive the ended worlds or pay the seat twice.
        _, reward2, done2, info2 = group.step(torch.ones(group.num_envs, 6, device=group.device))
        assert not done2[:5].any() and not reward2[:5].any()
        torch.testing.assert_close(info2['fault_code'][:5], info['fault_code'][:5], atol=0, rtol=0)
    finally:
        group.read_wrench, group._sample = read, sample
        group.control.cancel_reference = cancel
        group.auto_reset = auto
        group.restore_snapshot(state)
    return dict(one_ms_seat_latch=True, load_fault_precedes_same_sample_seat=True,
                force_and_moment_faults_latch=True, terminal_reference_cancelled=True,
                first_terminal_qpos_preserved=True, post_terminal_peaks_excluded=True,
                ended_worlds_cannot_resume_or_repeat_reward=True,
                injected_transition_checks_are_not_physical_seating_tests=True)
