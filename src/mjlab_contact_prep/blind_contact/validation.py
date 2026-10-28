"""Small physical probes and held-out policy evaluation with saved trajectories."""
from dataclasses import replace
import hashlib
import math
from pathlib import Path
import time
import numpy as np
import torch
from .assets import NOMINAL_TIP, solve_ik
from .environment import BlindInsertionEnv
from .cartesian_environment import make_task_environment as make_environment
from .cartesian_networks import models_for, tensordict_for
from .training import freeze_run, save_json


def control_probe(folder, cfg):
    if cfg.study_revision == 6:
        from .cartesian_validation import cartesian_control_probe
        return cartesian_control_probe(folder, cfg)
    if cfg.study_revision >= 3 and len(cfg.assembly_shapes)>1:
        return {shape:control_probe(Path(folder)/shape,replace(cfg,assembly_shapes=[shape]))
                for shape in cfg.assembly_shapes}
    count=5 if cfg.method=="study" else 4
    cfg = replace(cfg,num_envs=count,xy_min_m=0.,xy_max_m=0.,tilt_deg=0.,
                  world_xy_jitter_m=0.,horizon_s=5. if cfg.method=="study" else 12.,reset_bank_size=16)
    if cfg.study_revision == 4:
        from .polygon_environment import make_shape_environment
        env = make_shape_environment(cfg,auto_reset=False)
    else:
        env = make_environment(cfg,auto_reset=False)
    if cfg.method == "study":
        # Privileged diagnostics deliberately share a known aligned trajectory.
        q=solve_ik(env.model,NOMINAL_TIP,env.nominal_rotation)
        env.data.qpos[:]=torch.tensor(q,device=env.device,dtype=torch.float32)
        env.data.qvel[:]=0
        env.data.mocap_pos[:,env.hole_mocap]=torch.tensor(
            [*NOMINAL_TIP[:2],.1],device=env.device,dtype=torch.float32)
        env.data.mocap_quat[:,env.hole_mocap]=torch.tensor([1.,0,0,0],device=env.device)
        env.sim.forward()
        env.control.target[:]=env.data.qpos[:,:6]
        env.history[:]=env.proprioception()[:,None,:]
    folder=Path(folder);freeze_run(folder,cfg,env,"control_probe")
    # Same robot trajectory, four distinct contact situations.
    offsets=torch.tensor([0.,.002,.020,.15,.15] if cfg.method=="study" else [0.,.002,.020,0.],device=env.device)
    env.data.mocap_pos[:,env.hole_mocap,0] += offsets
    env.sim.forward()
    labels=["aligned","edge_2mm","plate_20mm","free_hold"]+(["free_motion"] if cfg.method=="study" else [])
    traces=[];rows={};timer=time.monotonic()
    with torch.no_grad():
        initial=env.data.qpos.clone()
        qseed=initial[0,:6].cpu().numpy()
        for step in range(int(cfg.horizon_s/cfg.policy_dt)):
            p=NOMINAL_TIP.copy()
            t=step*cfg.policy_dt
            travel=(.003*min(t,1.5)+.016*max(t-1.5,0.)) if cfg.method=="study" else t*.006
            p[2]-=min(travel,.0465)
            if cfg.study_revision>=3:
                # Diagnostic-only correction for the measured +X tracking bias
                # (~0.4--0.6 mm) of the existing joint impedance trajectory.
                # The learned policy must infer its own action from history.
                p[0]-=.0004*min(t/1.5,1.)
            kwargs={"target_rotation":env.nominal_rotation} if cfg.method=="study" else {}
            command=solve_ik(env.model,p,initial=qseed,**kwargs)
            desired_velocity=(command[:6]-qseed)/cfg.policy_dt
            qseed=command[:6]
            command_target=command[:6].copy()
            if cfg.method=="study":
                # Diagnostic trajectory feedforward compensates the known
                # viscous lag. The learned policy receives no such trajectory.
                command_target += ((np.asarray(cfg.kd)+env.model.dof_damping[env.control.vids])
                                   /np.asarray(cfg.kp))*desired_velocity
            targets=torch.tensor(command_target,device=env.device,dtype=torch.float32).repeat(count,1)
            targets[3]=initial[3,:6]
            action=(targets-env.data.qpos[:,:6])/cfg.action_scale
            if cfg.action_transform == "tanh":
                # Invert the execution map for this privileged target probe;
                # unreachable targets remain bounded by the same policy limit.
                action=torch.atanh(action.clamp(-.999,.999))
            _,_,done,info=env.step(action)
            traces.append(torch.cat((env.data.qpos[:,:6],env.data.qvel[:,:6],
                env.control.target-env.data.qpos[:,:6],
                info["depth"][:,None],info["shaft_error"][:,None],
                env.peak_force[:,None],info["fault_code"][:,None].float()),-1).cpu().numpy())
            for i in done.nonzero().flatten().tolist():
                rows[i]=dict(case=labels[i],success=bool(info["success"][i]),
                    depth_m=float(info["depth"][i]),force_peak_n=float(env.peak_force[i]),
                    fault_code=int(info["fault_code"][i]),elapsed_s=(step+1)*cfg.policy_dt)
            if step%100==0:
                print("probe",step,info["depth"].tolist(),env.peak_force.tolist(),flush=True)
        for i in range(count):
            if i not in rows:
                rows[i]=dict(case=labels[i],success=False,depth_m=float(info["depth"][i]),
                    force_peak_n=float(env.peak_force[i]),fault_code=int(info["fault_code"][i]))
    np.savez_compressed(folder/"trajectory.npz",states=np.stack(traces),
                       columns=np.array([f"q{i}" for i in range(6)]+[f"dq{i}" for i in range(6)]
                           +[f"tracking_error{i}" for i in range(6)]
                           +["depth_m","shaft_error_m","force_peak_n","fault_code"]))
    report=dict(cases=list(rows.values()),wall_s=time.monotonic()-timer,
                max_contacts=int(env.max_contacts[0]),max_constraints=int(env.max_constraints[0]),
                buffer_overflow=bool(env.buffer_overflow[0]),
                controller_only=True,learned_policy=False)
    report["force_sample_hz"]=1/cfg.policy_dt
    if cfg.method=="study":
        report["diagnostic_viscous_feedforward"]=True
        report["diagnostic_trajectory"]="3 mm/s for first 1.5 s, then 16 mm/s; known target only"
        report["diagnostic_lateral_compensation_m"]=-.0004 if cfg.study_revision>=3 else 0.
        a=np.stack(traces)[25:100]
        report["matched_command_contact_signal"]={
            "plate_tracking_error_rms_rad":float(np.sqrt(np.mean(a[:,2,12:18]**2))),
            "free_tracking_error_rms_rad":float(np.sqrt(np.mean(a[:,4,12:18]**2))),
            "plate_vs_free_joint_difference_rms_rad":float(np.sqrt(np.mean((a[:,2,:6]-a[:,4,:6])**2)))}
    save_json(folder/"results.json",report)
    print(report,flush=True)
    return report


def study_integrity_probe(folder,cfg):
    """Real GPU checks of the study reset, observation, reward and replay."""
    if cfg.study_revision in (4,6) or (cfg.study_revision==3 and len(cfg.assembly_shapes)>1):
        return polygon_integrity_probe(folder,cfg)
    cfg=replace(cfg,num_envs=32,reset_bank_size=128)
    torch.manual_seed(cfg.seed)
    env=make_environment(cfg)
    folder=Path(folder);freeze_run(folder,cfg,env,"study_gpu_integrity")
    with torch.no_grad():
        points,_,rotation=env.local_geometry()
        _,initial_success=env.events(points,rotation,env.truth()[2])
        assert not initial_success.any()
        assert env.reset_audit["all_starts_have_actual_contact"]
        initial=env.snapshot()
        if cfg.study_revision >= 2:
            assert not env.captured.any()
            before=env.observations()
            env.captured[0]=True
            after=env.observations()
            torch.testing.assert_close(before["history"],after["history"],atol=0,rtol=0)
            torch.testing.assert_close(before["critic"][:,:-1],after["critic"][:,:-1],atol=0,rtol=0)
            assert after["critic"].shape[-1]==235 and after["critic"][0,-1]==1
            env.restore_snapshot(initial)
        actor_before=env.proprioception().clone()
        env.data.mocap_pos[:,:,0]+=.005
        env.sim.forward()
        torch.testing.assert_close(env.proprioception(),actor_before,atol=1e-7,rtol=0)
        env.restore_snapshot(initial)
        action=torch.randn(32,6,device=env.device)*.25
        env.step(action)
        snapshot=env.snapshot()
        out1=env.step(action)
        q1=env.data.qpos.clone();h1=env.history.clone()
        env.restore_snapshot(snapshot)
        torch.testing.assert_close(env.data.qvel.clone(),snapshot["data"]["qvel"],atol=0,rtol=0)
        torch.testing.assert_close(env.data.qacc_warmstart.clone(),snapshot["data"]["qacc_warmstart"],atol=0,rtol=0)
        out2=env.step(action)
        error=float((env.data.qpos-q1).abs().max())
        history_error=float((env.history-h1).abs().max())
        torch.testing.assert_close(env.data.qpos.clone(),q1,atol=2e-6,rtol=0)
        # Contact solver replay is bounded, not bitwise deterministic. In this
        # raw frame, unscaled TCP twist dominates the difference after contact.
        torch.testing.assert_close(env.history,h1,atol=2e-4,rtol=0)
        torch.testing.assert_close(out1[1],out2[1],atol=0,rtol=0)
        other=env.history[1:].clone()
        env.reset(torch.tensor([0],device=env.device))
        torch.testing.assert_close(env.history[1:],other,atol=0,rtol=0)
        torch.testing.assert_close(env.history[0],env.history[0,:1].expand_as(env.history[0]),atol=0,rtol=0)
        fault_before=env.fault_code.clone()
        env.data.sensordata[:,env.force_adr]=500
        env.monitor_physics()
        torch.testing.assert_close(env.fault_code,fault_before,atol=0,rtol=0)
        fr=env.sim.model.geom_friction[:,env.hole_geoms[0],0].clone()
        tc=env.sim.model.geom_solref[:,env.hole_geoms[0],0].clone()
        assert float(fr.min())>=cfg.plate_friction_range[0] and float(fr.max())<=cfg.plate_friction_range[1]
        assert float(tc.min())>=cfg.plate_timeconst_range_s[0] and float(tc.max())<=cfg.plate_timeconst_range_s[1]
        assert fr.std()>0 and tc.std()>0
    report=dict(reset_audit=env.reset_audit,initial_success_count=0,
                hidden_hole_does_not_enter_actor=True,partial_reset_isolated=True,
                snapshot_replay_max_qpos_error_rad=error,force_is_diagnostic_only=True,
                snapshot_replay_max_observation_error=history_error,
                replay_observation_atol=2e-4,
                study_revision=cfg.study_revision,critic_dim=env.critic_dim,
                capture_flag_is_critic_only=cfg.study_revision>=2,
                friction_range_observed=[float(fr.min()),float(fr.max())],
                timeconst_range_observed_s=[float(tc.min()),float(tc.max())],
                buffer_overflow=bool(env.buffer_overflow[0]))
    save_json(folder/"results.json",report);print(report,flush=True)
    return report


def polygon_integrity_probe(folder,cfg):
    """Check grouped reset, critic-only geometry identity, replay and true contact."""
    groups=len(cfg.assembly_shapes)
    cfg=replace(cfg,num_envs=16*groups,reset_bank_size=64*groups)
    torch.set_num_threads(4);torch.manual_seed(cfg.seed)
    env=make_environment(cfg)
    folder=Path(folder);freeze_run(folder,cfg,env,"polygon_gpu_integrity")
    with torch.no_grad():
        assert env.reset_audit["all_starts_have_actual_contact"]
        assert not env.captured.any()
        for group in env.envs:
            points,_,rotation=group.local_geometry()
            assert not group.events(points,rotation,group.truth()[2])[1].any()
            before=group.proprioception().clone()
            original=group.data.mocap_pos.clone()
            group.data.mocap_pos[:,:,0]+=.005;group.sim.forward()
            torch.testing.assert_close(group.proprioception(),before,atol=1e-7,rtol=0)
            group.data.mocap_pos.copy_(original);group.sim.forward()
        as_tensordict=tensordict_for(cfg)
        raw=env.observations();obs=as_tensordict(raw)
        assert obs["critic"].shape==(cfg.num_envs,env.critic_dim)
        code_width=len(cfg.shape_vocabulary)
        torch.testing.assert_close(obs["critic"][:,-code_width:],env.shape_code,atol=0,rtol=0)
        actor,critic=models_for(cfg,obs)
        actor=actor.to(env.device)
        critic=critic.to(env.device)
        assert critic(obs).shape==(cfg.num_envs,1)
        parameters={k:v.clone() for k,v in actor.state_dict().items()}
        before=actor(obs).clone();other=obs.clone()
        other["critic"][:]=999;other["labels"][:]=2
        torch.testing.assert_close(actor(other),before,atol=0,rtol=0)
        action=torch.randn(cfg.num_envs,6,device=env.device)*.25
        env.step(action);state=env.snapshot()
        out1=env.step(action);q1=env.data.qpos.clone();h1=env.history.clone()
        env.restore_snapshot(state)
        if cfg.study_revision == 6:
            # Persisted environment/controller/filter/reward state must restore
            # exactly; only subsequent floating-point simulation has a tolerance.
            again = env.snapshot()
            for original, restored in zip(state["groups"], again["groups"]):
                for section in ("data", "buffers", "v6", "v6_control"):
                    for key in original[section]:
                        torch.testing.assert_close(restored[section][key], original[section][key], atol=0, rtol=0)
        out2=env.step(action)
        q_error=float((env.data.qpos-q1).abs().max())
        h_error=float((env.history-h1).abs().max())
        torch.testing.assert_close(env.data.qpos,q1,atol=2e-6,rtol=0)
        torch.testing.assert_close(env.history,h1,atol=2e-4,rtol=0)
        torch.testing.assert_close(out1[1],out2[1],atol=0,rtol=0)
        before=env.history.clone();ids=torch.arange(groups,device=env.device)*16
        env.reset(ids);mask=torch.ones(cfg.num_envs,device=env.device,dtype=torch.bool);mask[ids]=False
        torch.testing.assert_close(env.history[mask],before[mask],atol=0,rtol=0)
        torch.testing.assert_close(env.history[ids],env.history[ids,:1].expand(-1,30,-1),atol=0,rtol=0)
        for group in env.envs:
            group.steps[:]=math.ceil(cfg.horizon_s/cfg.policy_dt)-1
            if cfg.study_revision == 6:
                group.physics_steps[:] = math.ceil(cfg.horizon_s/cfg.physics_dt)-cfg.decimation
        _,_,done,info=env.step(torch.zeros(cfg.num_envs,6,device=env.device))
        assert done.all() and env.active.all()
        assert all((e.steps==0).all() for e in env.envs)
        assert info["terminal"]["critic"].shape==(cfg.num_envs,env.critic_dim)
        for i,shape in enumerate(env.shapes):
            assert (info["yaw_alignment_required"][i*16:(i+1)*16]==(shape!="circle")).all()
        for key,value in actor.state_dict().items():
            torch.testing.assert_close(value,parameters[key],atol=0,rtol=0)
    report=dict(reset_audit=env.reset_audit,actor_geometry_hidden=True,critic_dim=env.critic_dim,
                shared_actor=True,shape_counts={s:e.num_envs for s,e in zip(env.shapes,env.envs)},
                partial_reset_across_groups_isolated=True,snapshot_replay_max_qpos_error_rad=q_error,
                snapshot_replay_max_observation_error=h_error,initial_success_count=0,
                timeout_autoreset_all_groups=True,circle_yaw_is_not_applicable=True,
                optimizer_updates=0,actor_parameters_unchanged=True,
                buffer_overflow=bool(env.buffer_overflow[0]))
    if cfg.study_revision == 6:
        from .cartesian_validation import transition_checks
        with torch.no_grad():
            report.update(transition_checks(env))
        report["snapshot_persisted_state_exact"] = True
        report["snapshot_history_tolerance_unchanged"] = 2e-4
    save_json(folder/"results.json",report);print(report,flush=True)
    env.close()
    return report


def current_keys(cfg):
    return list(cfg.to_dict())


def wilson(successes,total):
    p=successes/total;z=1.96;den=1+z*z/total
    center=(p+z*z/(2*total))/den
    half=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return [center-half,center+half]


def integrity_probe(folder, cfg):
    """Exercise real GPU reset/replay and a one-sample protection injection."""
    cfg=replace(cfg,num_envs=4)
    env=BlindInsertionEnv(cfg)
    folder=Path(folder);freeze_run(folder,cfg,env,"gpu_integrity")
    with torch.no_grad():
        actor_before=env.proprioception().clone()
        original_hole=env.data.mocap_pos.clone()
        env.data.mocap_pos[:,:,0]+=.005
        env.sim.forward()
        torch.testing.assert_close(env.proprioception(),actor_before,atol=1e-7,rtol=0)
        env.data.mocap_pos.copy_(original_hole);env.sim.forward()
        action=torch.tensor([.08,-.12,.1,.03,-.1,.02],device=env.device).repeat(4,1)
        env.step(action)
        state=env.snapshot()
        out1=env.step(action)
        q1=env.data.qpos.clone();h1=env.history.clone()
        env.restore_snapshot(state)
        out2=env.step(action)
        error=float((env.data.qpos-q1).abs().max())
        torch.testing.assert_close(env.data.qpos.clone(),q1,atol=2e-6,rtol=0)
        torch.testing.assert_close(env.history,h1,atol=2e-5,rtol=0)
        history_other=env.history[1:].clone()
        env.reset(torch.tensor([0],device=env.device))
        torch.testing.assert_close(env.history[1:],history_other,atol=0,rtol=0)
        torch.testing.assert_close(env.history[0],env.history[0,:1].expand_as(env.history[0]),atol=0,rtol=0)
        sensor_before=env.data.sensordata.clone()
        env.data.sensordata[0,env.force_adr]=500.
        env.monitor_physics()
        env.data.sensordata.copy_(sensor_before)
        env.monitor_physics()
        assert env.fault_code[0].item() & 1
        env.reset(torch.tensor([0],device=env.device))
        assert env.fault_code[0].item()==0
    report=dict(hidden_hole_does_not_enter_actor=True,reset_history_is_fresh=True,
                partial_reset_preserves_other_histories=True,
                snapshot_replay_max_qpos_error_rad=error,one_sample_overload_latches=True)
    save_json(folder/"results.json",report);print(report,flush=True)
    return report


def evaluate(folder, checkpoint, cfg, *, stochastic=False, allow_clearance_transfer=False, allow_seating_transfer=False):
    torch.set_num_threads(4);torch.manual_seed(cfg.seed)
    blob=torch.load(checkpoint,map_location="cpu",weights_only=False)
    if (blob["config"]["method"] != cfg.method or
        blob["config"].get("study_revision",1) != cfg.study_revision):
        raise ValueError("evaluation requires the checkpoint's task revision")
    from .clearance_validation import clearance_transfer_contract,check_asset_contract,summarize_cases
    check_asset_contract(blob["config"],cfg)
    from .seating_ablation import seating_transfer_contract
    seating = seating_transfer_contract(blob["config"],cfg,allow_seating_transfer)
    transfer=clearance_transfer_contract(blob["config"]["clearance_per_side_m"],
                                        cfg.clearance_per_side_m,allow_clearance_transfer)
    if cfg.method=="study" and cfg.study_revision>=2:
        fields=("action_transform","action_scale","action_clip","kp","kd","torque_limits",
                "friction_compensation","friction_deadband_rad","clearance_per_side_m",
                "capture_depth_m","seating_tolerance_m","geometry_tolerance_m",
                "goal_depth_m","horizon_s","physics_dt","decimation")
        if cfg.study_revision >= 3:
            fields += ("assembly_shapes","polygon_peg_width_m")
        if cfg.study_revision == 6:
            fields += tuple(k for k in current_keys(cfg) if k.startswith("cart_") or k.endswith("_threshold_n")
                            or k in ("force_band_n","force_retreat_m_per_ns","moment_threshold_nm","moment_band_nm",
                                     "moment_comply_rad_per_nms","governor_wrench_tau_s",
                                     "wrench_force_scale_n","wrench_moment_scale_nm","force_abort_n","moment_abort_nm"))
        current=cfg.to_dict()
        changed=[key for key in fields if blob["config"].get(key)!=current[key]]
        if allow_clearance_transfer:
            changed=[key for key in changed if key!="clearance_per_side_m"]
        if changed:
            raise ValueError(f"evaluation control/task contract differs: {changed}")
    env=make_environment(cfg,auto_reset=False)
    folder=Path(folder);freeze_run(folder,cfg,env,"held_out_evaluation")
    as_tensordict=tensordict_for(cfg)
    obs=as_tensordict(env.observations())
    actor,critic=models_for(cfg,obs)
    actor=actor.to(env.device)
    actor.load_state_dict(blob["algorithm"]["actor_state_dict"])
    actor.eval()
    rows=[dict(case_id=i,initial_radius_m=float(env.reset_radius[i]),outcome="pending")
          for i in range(cfg.num_envs)]
    if cfg.study_revision >= 3:
        shape_ids=(env.shape_ids.cpu().tolist() if hasattr(env,"shape_ids") else [0]*cfg.num_envs)
        for row,shape_id in zip(rows,shape_ids):
            row["shape"]=cfg.assembly_shapes[shape_id]
    initial_q=env.data.qpos.cpu().numpy().copy()
    initial_hole=env.data.mocap_pos.cpu().numpy().copy()
    initial_quat=env.data.mocap_quat.cpu().numpy().copy()
    trajectories=[]
    sample_times=[]
    sample_valid=[]
    maximum_depth=torch.full((cfg.num_envs,),-float("inf"),device=env.device)
    with torch.no_grad():
        for step in range(math.ceil(cfg.horizon_s/cfg.policy_dt)):
            action=actor(obs,stochastic_output=stochastic)
            was_active=env.active.clone()
            next_obs,_,done,info=env.step(action)
            if cfg.study_revision == 6:
                sample_times.append(info["elapsed_s"].cpu().numpy())
                sample_valid.append(was_active.cpu().numpy())
            maximum_depth=torch.where(was_active,torch.maximum(maximum_depth,info["depth"]),maximum_depth)
            obs=as_tensordict(next_obs)
            if cfg.method=="study":
                for i in info["new_milestones"].nonzero().flatten().tolist():
                    rows[i].update(capture_time_s=(step+1)*cfg.policy_dt,
                                   capture_xy_m=float(info["xy_error"][i]),
                                   capture_tilt_deg=float(info["tilt_rad"][i])*180/math.pi,
                                   capture_depth_m=float(info["depth"][i]),
                                   capture_shaft_error_m=float(info["shaft_error"][i]),
                                   capture_full_ring_inside=bool(info["full_ring_inside"][i]))
                    if cfg.study_revision == 6:
                        rows[i].update(capture_time_s=float(info["capture_time_s"][i]),
                                       capture_xy_m=float(info["capture_xy_m"][i]),
                                       capture_tilt_deg=float(info["capture_tilt_rad"][i])*180/math.pi,
                                       capture_depth_m=float(info["capture_depth_m"][i]),
                                       capture_shaft_error_m=float(info["capture_shaft_error_m"][i]),
                                       capture_full_ring_inside=bool(info["capture_full_ring_inside"][i]))
            trajectories.append(torch.cat((info.get("trajectory_qpos",env.data.qpos).clone(), info["depth"][:,None],
                                           info["shaft_error"][:,None],env.peak_force[:,None]),-1).cpu().numpy())
            for i in done.nonzero().flatten().tolist():
                rows[i].update(outcome="success" if info["success"][i] else
                               ("protection" if info["fault"][i] else "timeout"),
                               depth_m=float(info["depth"][i]),shaft_error_m=float(info["shaft_error"][i]),
                               max_depth_m=float(maximum_depth[i]),
                               elapsed_s=(step+1)*cfg.policy_dt,force_peak_n=float(info["force_peak_n"][i]),
                               fault_code=int(info["fault_code"][i]))
                if cfg.study_revision == 6:
                    rows[i].update(elapsed_s=float(info["elapsed_s"][i]),
                                   geometry_success=bool(info["geometry_success"][i]),
                                   force_qualified_success=bool(info["force_qualified_success"][i]),
                                   moment_peak_nm=float(info["moment_peak_nm"][i]),
                                   force_exceed_s=float(info["force_exceed_s"][i]),
                                   moment_exceed_s=float(info["moment_exceed_s"][i]),
                                   max_legal_depth_m=float(info["legal_depth_m"][i]))
                if cfg.method=="study":
                    rows[i].update(terminal_xy_m=float(info["xy_error"][i]),
                                   terminal_tilt_deg=float(info["tilt_rad"][i])*180/math.pi)
                if "yaw_error_rad" in info:
                    required=bool(info["yaw_alignment_required"][i])
                    rows[i]["yaw_alignment_required"]=required
                    rows[i]["terminal_symmetry_yaw_error_deg"]=(float(info["yaw_error_rad"][i])*180/math.pi
                                                               if required else None)
            if not env.active.any():
                break
    if any(r["outcome"]=="pending" for r in rows):
        raise RuntimeError("incomplete evaluation; pending cases cannot be scored")
    count=sum(r["outcome"]=="success" for r in rows)
    report=dict(successes=count,total=len(rows),success_rate=count/len(rows),
                wilson_95=wilson(count,len(rows)),checkpoint=str(Path(checkpoint).resolve()),
                checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                deterministic=not stochastic,reset_seed=cfg.seed,
                initial_distribution=dict(xy_min_m=cfg.xy_min_m,xy_max_m=cfg.xy_max_m,tilt_deg=cfg.tilt_deg),
                goal_depth_m=cfg.goal_depth_m,hold_s=cfg.success_hold_s,
                method=cfg.method,
                study_revision=cfg.study_revision,
                force_sample_hz=1/(cfg.physics_dt if cfg.study_revision==6 else cfg.policy_dt),
                trajectory_sample_hz=1/cfg.policy_dt,
                terminal_detection_hz=1/(cfg.physics_dt if cfg.study_revision==6 else cfg.policy_dt),
                seating_tolerance_m=cfg.seating_tolerance_m if cfg.method=="study" else None,
                max_contacts=int(env.max_contacts[0]),max_constraints=int(env.max_constraints[0]),
                buffer_overflow=bool(env.buffer_overflow[0]),cases=rows)
    report.update(transfer)
    report.update(seating)
    if allow_seating_transfer:
        report["allowed_contract_changes"].append("seating_protocol")
    if cfg.study_revision == 6:
        report.update(geometry_successes=sum(r["geometry_success"] for r in rows),
                      force_qualified_successes=sum(r["force_qualified_success"] for r in rows),
                      force_abort_n=cfg.force_abort_n, moment_abort_nm=cfg.moment_abort_nm,
                      episode_peak_moment_nm_quantiles=dict(zip(("p50","p95","max"),
                          np.quantile([r["moment_peak_nm"] for r in rows],[.5,.95,1]).tolist())),
                      force_exceed_s_total=sum(r["force_exceed_s"] for r in rows),
                      moment_exceed_s_total=sum(r["moment_exceed_s"] for r in rows),
                      terminal_protocol="first 1 ms event; load violation wins over geometric seat; no post-seat hold",
                      sensor_reconstruction="v6 per-row contact addresses; static payload gravity compensation")
    report.update(summarize_cases(rows,cfg))
    report["geometrically_legal_successes"]=sum(
        r["outcome"]=="success" and r["shaft_error_m"]<=cfg.geometry_tolerance_m for r in rows)
    report["geometrically_legal_success_rate"]=report["geometrically_legal_successes"]/len(rows)
    if cfg.method=="study":
        captured_rows=[r for r in rows if "capture_time_s" in r]
        report["new_captures"]=len(captured_rows)
        report["success_after_new_capture"]=sum(r["outcome"]=="success" for r in captured_rows)
        report["capture_to_success_rate"]=(report["success_after_new_capture"]/len(captured_rows)
                                           if captured_rows else None)
        report["episode_peak_force_n_quantiles"]=dict(zip(
            ["p50","p95","max"],np.quantile([r["force_peak_n"] for r in rows],[.5,.95,1]).tolist()))
    if cfg.method=="study":
        report["initial_distribution"]=dict(
            xy_uniform_square_halfwidth_m=cfg.reset_xy_halfwidth_m,
            relative_rotation_per_axis_deg=cfg.tilt_deg,
            fixture_rotation_per_axis_deg=cfg.fixture_tilt_deg,
            fixture_xy_jitter_m=cfg.fixture_xy_jitter_m,
            fixture_z_jitter_m=cfg.fixture_z_jitter_m,
            reset_bank_size=cfg.reset_bank_size,reset_bank_seed=cfg.seed,
            plate_friction_range=cfg.plate_friction_range,
            plate_timeconst_range_s=cfg.plate_timeconst_range_s)
    if cfg.study_revision >= 3:
        report["by_shape"]={}
        for shape in cfg.assembly_shapes:
            selected=[r for r in rows if r["shape"]==shape]
            successes=sum(r["outcome"]=="success" for r in selected)
            report["by_shape"][shape]=dict(total=len(selected),successes=successes,
                success_rate=successes/len(selected),wilson_95=wilson(successes,len(selected)),
                geometrically_illegal_successes=sum(r["outcome"]=="success" and
                    r["shaft_error_m"]>cfg.geometry_tolerance_m for r in selected))
            report["by_shape"][shape].update(summarize_cases(selected,cfg))
    timing = dict(sample_time_s=np.stack(sample_times),sample_valid=np.stack(sample_valid)) if cfg.study_revision==6 else {}
    np.savez_compressed(folder/"trajectory.npz",qpos_and_metrics=np.stack(trajectories), **timing,
                       initial_qpos=initial_q,initial_mocap_pos=initial_hole,initial_mocap_quat=initial_quat)
    save_json(folder/"results.json",report)
    print({k:v for k,v in report.items() if k!="cases"},flush=True)
    env.close()
    return report
