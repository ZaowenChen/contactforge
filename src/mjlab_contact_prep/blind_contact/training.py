"""Contact-supervised PPO extension using the existing RSL-RL storage and GAE."""
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import random
import shutil
import subprocess
import time

import mujoco
import numpy as np
import torch
from rsl_rl.algorithms import PPO
from rsl_rl.storage import RolloutStorage
from .config import BlindConfig
from .cartesian_environment import make_task_environment
from .cartesian_networks import models_for, tensordict_for, pose_loss

ROOT = Path(__file__).resolve().parents[3]


def implementation_fingerprint():
    paths=list(Path(__file__).parent.glob("*.py"))
    paths += [ROOT/"src/mjlab_contact_prep"/name
              for name in ["assets.py","backend.py","hole_collision.py","wrench_torch.py"]]
    # Validation/reporting edits do not change the training transition function.
    paths=[p for p in paths if p.name!="validation.py"]
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


class ContactPPO(PPO):
    def __init__(self, actor, critic, storage, cfg, device):
        super().__init__(actor,critic,storage,
                         num_learning_epochs=cfg.epochs, num_mini_batches=cfg.minibatches,
                         learning_rate=cfg.learning_rate, clip_param=cfg.clip_ratio,
                         gamma=cfg.gamma, lam=cfg.gae_lambda, entropy_coef=cfg.entropy_coef,
                         max_grad_norm=cfg.max_grad_norm, schedule="fixed", desired_kl=None,
                         device=device)
        self.contact_coef = cfg.contact_coef
        self.class_weights = torch.tensor(cfg.contact_weights,device=device)
        self.pose_coef = cfg.pose_coef if cfg.study_revision==6 else 0.
        self.to_td = tensordict_for(cfg)

    def update(self):
        totals = torch.zeros(8,device=self.device)
        count = 0
        for batch in self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs):
            self.actor(batch.observations, stochastic_output=True)
            logp = self.actor.get_output_log_prob(batch.actions)
            ratio = (logp-batch.old_actions_log_prob.flatten()).exp()
            approximate_kl = ((ratio-1)-(logp-batch.old_actions_log_prob.flatten())).mean()
            clip_fraction = ((ratio-1).abs()>self.clip_param).float().mean()
            advantage = batch.advantages.flatten()
            surrogate = -torch.minimum(ratio*advantage,
                ratio.clamp(1-self.clip_param,1+self.clip_param)*advantage).mean()
            values = self.critic(batch.observations)
            clipped_values = batch.values+(values-batch.values).clamp(-self.clip_param,self.clip_param)
            value_loss = torch.maximum((values-batch.returns).square(),
                                      (clipped_values-batch.returns).square()).mean()
            if self.pose_coef:
                logits,pose = self.actor.auxiliary(batch.observations)
                pose_term = pose_loss(pose,batch.observations["pose"])
            else:
                logits = self.actor.contact_logits(batch.observations)
                pose_term = torch.zeros((),device=self.device)
            contact_loss = torch.nn.functional.cross_entropy(logits.flatten(0,1),
                batch.observations["labels"].long().flatten(),weight=self.class_weights)
            entropy = self.actor.output_entropy.mean()
            loss = surrogate+self.value_loss_coef*value_loss-self.entropy_coef*entropy+self.contact_coef*contact_loss+self.pose_coef*pose_term
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite PPO loss")
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                list(self.actor.parameters())+list(self.critic.parameters()),self.max_grad_norm,
                error_if_nonfinite=True)
            self.optimizer.step()
            totals += torch.stack((surrogate.detach(),value_loss.detach(),contact_loss.detach(),
                                   entropy.detach(),norm.detach(),approximate_kl.detach(),clip_fraction.detach(),pose_term.detach()))
            count += 1
        # Keep normalization fixed while computing old/new likelihood ratios.
        if hasattr(self.actor,"refresh_normalization"):
            flat = self.storage.observations.flatten(0,1)
            self.actor.refresh_normalization(flat)
            self.critic.refresh_normalization(flat)
        self.storage.clear()
        return dict(zip(["policy_loss","value_loss","contact_loss","entropy","gradient_norm","approximate_kl","clip_fraction","pose_loss"],
                        (totals/count).tolist()))

    def record_step(self, obs, reward, done, info):
        # Bootstrap the FINAL state of a truncated episode, never its reset or
        # preceding state. RSL-RL's GAE then stops at the episode boundary.
        terminal_value = self.critic(self.to_td(info["terminal"])).flatten().detach()
        bootstrapped = reward+self.gamma*terminal_value*info["timeout"].float()
        self.process_env_step(obs,bootstrapped,done,{})


def save_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+"\n")


def freeze_run(folder, cfg, env, kind):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    save_json(folder/"config.json",cfg.to_dict())
    if hasattr(env,"reset_audit"):
        save_json(folder/"reset_audit.json",env.reset_audit)
    # Copy actual source (including uncommitted implementation) into each run.
    shutil.copytree(ROOT/"src",folder/"source/src")
    shutil.copy2(ROOT/"scripts/blind_contact_ppo.py",folder/"source/entrypoint.py")
    hashes = {str(p.relative_to(folder/"source")):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted((folder/"source").rglob("*")) if p.is_file()}
    asset_hashes = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted((ROOT/"assets").rglob("*")) if p.is_file()}
    models = ({shape:group.model for shape,group in zip(env.shapes,env.envs)}
              if hasattr(env,"envs") else {"assembly":env.model})
    model_hashes={}
    for shape,model in models.items():
        name=f"model_{shape}.mjb" if len(models)>1 else "model.mjb"
        mujoco.mj_saveModel(model,str(folder/name))
        model_hashes[name]=hashlib.sha256((folder/name).read_bytes()).hexdigest()
    save_json(folder/"manifest.json",dict(kind=kind,
        git_head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        source_sha256=hashes,assets_sha256=asset_hashes,
        runtime={name:version(name) for name in ["torch","mujoco","mujoco-warp","mjlab","rsl-rl-lib"]},
        model_sha256=next(iter(model_hashes.values())) if len(models)==1 else model_hashes,
        assembly_shapes=cfg.assembly_shapes,
        critic_shape_vocabulary=list(cfg.shape_vocabulary),
        physics_backend=cfg.physics_backend,
        actor_inputs=("q,dq,encoder-derived TCP pose and twist, previous Cartesian action, interval-mean compensated tip F/T"
                      if cfg.study_revision==6 else "q,dq,encoder-derived TCP pose and twist, previous action"),
        privileged_inputs="critic, contact auxiliary loss, reward and evaluation only"))


def checkpoint(path, algorithm, env, cfg, iteration, fingerprint):
    state = dict(format="blind-contact-ppo-v1", config=cfg.to_dict(), iteration=iteration,
                 implementation_sha256=fingerprint,
                 algorithm=algorithm.save(), environment=env.snapshot(),
                 torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                 numpy_rng=np.random.get_state(), python_rng=random.getstate())
    temp = Path(path).with_suffix(".tmp")
    torch.save(state,temp)
    temp.replace(path)


def train(folder, cfg, iterations, *, initialize_from=None, resume=None):
    if cfg.study_revision == 4 and cfg.seating_protocol != "legacy_v4":
        raise ValueError("revision-4 seating ablation is evaluation-only")
    random.seed(cfg.seed);np.random.seed(cfg.seed);torch.manual_seed(cfg.seed)
    torch.set_num_threads(4)
    env = make_task_environment(cfg)
    as_tensordict = tensordict_for(cfg)
    obs = as_tensordict(env.observations())
    actor, critic = models_for(cfg,obs)
    storage = RolloutStorage("rl",cfg.num_envs,cfg.rollout_steps,obs,[6],env.device)
    alg = ContactPPO(actor,critic,storage,cfg,env.device)
    if hasattr(actor,"refresh_normalization"):
        actor.refresh_normalization(obs)
        critic.refresh_normalization(obs)
    folder = Path(folder)
    freeze_run(folder,cfg,env,"training")
    fingerprint=implementation_fingerprint()
    start = 0
    source = resume or initialize_from
    if source:
        blob = torch.load(source,map_location=env.device,weights_only=False)
        if blob["format"] != "blind-contact-ppo-v1":
            raise ValueError("incompatible checkpoint")
        if (blob["config"]["method"] != cfg.method or
            blob["config"].get("study_revision",1) != cfg.study_revision):
            raise ValueError("checkpoint task revision differs; train the revised contract from scratch")
        from .clearance_validation import check_asset_contract,check_initialization_contract
        check_asset_contract(blob["config"],cfg)
        if initialize_from:
            check_initialization_contract(blob["config"],cfg)
            if blob.get("implementation_sha256") != fingerprint:
                raise ValueError("A-to-B initialization requires identical frozen implementation")
        alg.load(blob["algorithm"],
                 {"actor":True,"critic":True,"optimizer":bool(resume),"iteration":bool(resume)},True)
        save_json(folder/"checkpoint_source.json",dict(path=str(Path(source).resolve()),
            sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),mode="resume" if resume else "curriculum_initialization"))
        if resume:
            if blob["config"] != cfg.to_dict():
                raise ValueError("exact resume requires the same environment and training contract")
            if blob.get("implementation_sha256") != fingerprint:
                raise ValueError("exact resume requires identical training source; use initialize-from for an intentional new curriculum")
            env.restore_snapshot(blob["environment"])
            torch.set_rng_state(blob["torch_rng"].cpu())
            torch.cuda.set_rng_state_all([x.cpu() for x in blob["cuda_rng"]])
            np.random.set_state(blob["numpy_rng"]);random.setstate(blob["python_rng"])
            start=blob["iteration"]
            obs=as_tensordict(env.observations())
    checkpoint(folder/f"checkpoint_{start:05d}.pt",alg,env,cfg,start,fingerprint)
    timer=time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    records=[]
    try:
        for iteration in range(start,start+iterations):
            if iteration == cfg.profile_start_iteration:
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStart()
            tick=time.monotonic()
            counters=torch.zeros(8,device=env.device)
            study_counters=torch.zeros(7,device=env.device)
            shape_counters=torch.zeros(len(cfg.assembly_shapes),3,device=env.device)
            v6_counters=torch.zeros(10,device=env.device)
            max_depth=torch.full((),-float("inf"),device=env.device)
            torch.cuda.nvtx.range_push("rollout")
            with torch.no_grad():
                for _ in range(cfg.rollout_steps):
                    actions=alg.act(obs)
                    next_obs,reward,done,info=env.step(actions)
                    next_obs=as_tensordict(next_obs)
                    alg.record_step(next_obs,reward,done,info)
                    counters += torch.stack((done.sum(),(done & info["success"]).sum(),
                        (done & info["fault"]).sum(),info["new_milestones"].sum(),
                        (done & ((info["fault_code"] & 1)!=0)).sum(),
                        (done & ((info["fault_code"] & 4)!=0)).sum(),
                        (info["return_"]*done).sum(),(info["depth"]*done).sum()))
                    if cfg.method=="study":
                        first=info["new_milestones"].bool()
                        cx = info["capture_xy_m"] if cfg.study_revision==6 else info["xy_error"]
                        ct = info["capture_tilt_rad"] if cfg.study_revision==6 else info["tilt_rad"]
                        cd = info["capture_depth_m"] if cfg.study_revision==6 else info["depth"]
                        ce = info["capture_shaft_error_m"] if cfg.study_revision==6 else info["shaft_error"]
                        ci = info["capture_full_ring_inside"] if cfg.study_revision==6 else info["full_ring_inside"]
                        study_counters += torch.stack((
                            (cx*first).sum(),(ct*first).sum(),(cd*first).sum(),
                            (first & (ce<=cfg.geometry_tolerance_m) & ci).sum(),
                            (done & info["captured"]).sum(),(info["force_peak_n"]*done).sum(),
                            (done & info["success"] & (info["shaft_error"]>cfg.geometry_tolerance_m)).sum()))
                    if "shape_id" in info:
                        for shape_id in range(len(cfg.assembly_shapes)):
                            selected=done & (info["shape_id"]==shape_id)
                            shape_counters[shape_id] += torch.stack((selected.sum(),
                                (selected&info["success"]).sum(),(selected&info["captured"]).sum()))
                    if cfg.study_revision==6:
                        v6_counters += torch.stack((info["preload"].float().sum(),info["overload_cost"].sum(),
                            (done & (info["force_peak_n"]>50.)).sum(),(info["moment_peak_nm"]*done).sum(),
                            (info["legal_depth_m"]*done).sum(),
                            (done & ((info["fault_code"] & 2)!=0)).sum(),
                            (done & info["geometry_success"]).sum(),
                            (info["force_exceed_s"]*done).sum(),(info["moment_exceed_s"]*done).sum(),
                            (done & info["force_qualified_success"]).sum()))
                    max_depth=torch.maximum(max_depth,info["depth"].max())
                    obs=next_obs
                alg.compute_returns(obs)
            torch.cuda.nvtx.range_pop()
            rollout_s=time.monotonic()-tick
            torch.cuda.nvtx.range_push("ppo_update")
            metrics=alg.update()
            torch.cuda.nvtx.range_pop()
            completed,successes,faults,milestones,force_faults,pose_faults,returns,depths=counters.tolist()
            row=dict(iteration=iteration+1,completed=completed,successes=successes,faults=faults,
                     new_milestones=milestones,max_depth_m=float(max_depth),
                     force_faults=force_faults,pose_faults=pose_faults,
                     episode_return=returns/completed if completed else None,
                     terminal_depth_m=depths/completed if completed else None,
                     samples_per_s=cfg.num_envs*cfg.rollout_steps/(time.monotonic()-tick),
                     peak_torch_memory_bytes=torch.cuda.max_memory_allocated(),
                     max_contacts=int(env.max_contacts[0]),max_constraints=int(env.max_constraints[0]),
                     rollout_s=rollout_s,iteration_s=time.monotonic()-tick,
                     elapsed_s=time.monotonic()-timer,**metrics)
            if cfg.method=="study":
                cx,ct,cd,legal,captured_completed,force_sum,illegal_success=study_counters.tolist()
                row.update(study_revision=cfg.study_revision,
                           capture_xy_mean_m=cx/milestones if milestones else None,
                           capture_tilt_mean_deg=ct*180/np.pi/milestones if milestones else None,
                           capture_depth_mean_m=cd/milestones if milestones else None,
                           capture_legal_fraction=legal/milestones if milestones else None,
                           completed_captured=captured_completed,
                           completed_peak_force_mean_n=force_sum/completed if completed else None,
                           geometrically_illegal_successes=illegal_success)
            if cfg.study_revision==6:
                preload,overload,over50,moment_sum,legal_depth,moment_faults,geometric,force_time,moment_time,qualified=v6_counters.tolist()
                steps=cfg.num_envs*cfg.rollout_steps
                row.update(preload_step_fraction=preload/steps,overload_cost_per_step=overload/steps,
                           completed_force_over_50n=over50, moment_faults=moment_faults,
                           geometry_successes=geometric, force_qualified_successes=qualified,
                           completed_force_exceed_s=force_time, completed_moment_exceed_s=moment_time,
                           completed_peak_moment_mean_nm=moment_sum/completed if completed else None,
                           completed_legal_depth_mean_m=legal_depth/completed if completed else None)
            if hasattr(env,"shape_ids"):
                for shape,(total,success,capture) in zip(cfg.assembly_shapes,shape_counters.tolist()):
                    row.update({f"{shape}/completed":total,f"{shape}/successes":success,
                                f"{shape}/success_rate":success/total if total else None,
                                f"{shape}/capture_rate":capture/total if total else None})
            with (folder/"progress.jsonl").open("a") as f:
                f.write(json.dumps(row)+"\n")
            records.append(row)
            if (iteration+1)%10==0 or iteration==start or iteration==start+iterations-1:
                print(json.dumps(row),flush=True)
            if iteration == cfg.profile_stop_iteration:
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStop()
            if (iteration+1)%cfg.checkpoint_interval==0 or iteration==start+iterations-1:
                checkpoint(folder/f"checkpoint_{iteration+1:05d}.pt",alg,env,cfg,iteration+1,fingerprint)
        save_json(folder/"complete.json",dict(iterations=iterations,final_iteration=start+iterations,
                  transitions=iterations*cfg.num_envs*cfg.rollout_steps,
                  elapsed_s=time.monotonic()-timer,
                  note="Training episode counts are not held-out success rates."))
    except Exception as exc:
        save_json(folder/"failure.json",dict(error=repr(exc),completed_iterations=len(records)))
        raise
    finally:
        env.close()
    return records[-1]
