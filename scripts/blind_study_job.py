#!/usr/bin/env python3
"""Launch a bounded study run from frozen sources and evaluate its final policy."""
import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo


def save(path,data):
    temporary=path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n")
    temporary.replace(path)


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def stage_plan(cfg, transitions, evaluate_at=()):
    batch=cfg["num_envs"]*cfg["rollout_steps"]
    final=math.ceil(transitions/batch)
    ends=sorted({math.ceil(n/batch) for n in evaluate_at if 0<n<transitions}|{final})
    stages=[]
    start=0
    for index,end in enumerate(ends):
        stages.append(dict(start_iteration=start,end_iteration=end,
            training_dir="training" if index==0 else f"training_stage_{index+1}",
            evaluation_dir="evaluation" if end==final else f"evaluation_{end:05d}"))
        start=end
    return stages


def evaluation_episodes(cfg):
    """Keep 2048 held-out cases per trained shape, independent of group count."""
    return 2048*len(cfg.get("assembly_shapes",["circle"]))


def transfer_evaluation_config(cfg, clearance):
    from mjlab_contact_prep.blind_contact.clearance_validation import clearance_transfer_contract
    clearance_transfer_contract(cfg["clearance_per_side_m"],clearance,True)
    if cfg.get("study_revision") not in (4,6) or cfg.get("asset_variant")!="passive_xyz_v1":
        raise ValueError("transfer requires the passive-asset revision 4 or 6 contract")
    return dict(cfg,clearance_per_side_m=clearance)


def resolve_budget(cfg, transitions=None, evaluate_at=None):
    """Explicit CLI budgets override the configured long-run schedule."""
    batch=cfg["num_envs"]*cfg["rollout_steps"]
    configured=transitions is None and cfg.get("job_iterations") is not None
    if cfg.get("asset_variant")=="passive_xyz_v1" and transitions is None and not configured:
        raise ValueError("asset-only training requires an explicit budget")
    if transitions is None:
        transitions=cfg["job_iterations"]*batch if configured else 250_000_000
    if transitions <= 0:
        raise ValueError("positive transitions required")
    if evaluate_at is None:
        evaluate_at=([n*batch for n in cfg.get("job_evaluation_iterations",[])]
                     if configured else [])
    return transitions,evaluate_at


def log_progress(writer,path,cfg,offset=0):
    if not path.exists():
        return offset
    with path.open() as stream:
        stream.seek(offset)
        while True:
            position=stream.tell()
            line=stream.readline()
            if not line or not line.endswith("\n"):
                offset=position
                break
            row=json.loads(line)
            step=row["iteration"]*cfg["num_envs"]*cfg["rollout_steps"]
            for key,value in row.items():
                if isinstance(value,(int,float)) and math.isfinite(value):
                    writer.add_scalar(f"train/{key}",value,step)
            if row["completed"]:
                writer.add_scalar("train/success_rate",row["successes"]/row["completed"],step)
                if "completed_captured" in row:
                    writer.add_scalar("train/capture_rate",row["completed_captured"]/row["completed"],step)
    writer.flush()
    return offset


def worker(folder):
    from torch.utils.tensorboard import SummaryWriter

    spec=json.loads((folder/"job.json").read_text())
    source=folder/"frozen"
    env=dict(os.environ,PYTHONPATH=str(source/"src"),PYTHONDONTWRITEBYTECODE="1")
    entry=str(source/"scripts/blind_contact_ppo.py")
    cfg=json.loads((folder/"config.json").read_text())
    status=dict(state="training",started_at=now(),worker_pid=os.getpid())
    save(folder/"status.json",status)
    # A new bank and RNG seed; no selection of cases from the training bank.
    eval_count=spec.get("evaluation_episodes",evaluation_episodes(cfg))
    evaluation_cfg=dict(cfg,num_envs=eval_count,seed=10007,reset_bank_size=eval_count*4,
                        profile_start_iteration=-1,profile_stop_iteration=-1)
    save(folder/"evaluation_config.json",evaluation_cfg)
    writer=SummaryWriter(str(folder/"tensorboard"/f"study_v{cfg.get('study_revision',1)}"),flush_secs=5)
    writer.add_text("configuration",json.dumps(cfg,indent=2),0)
    writer.add_text("metrics","train/success_rate is the current rollout's completed-episode rate; "
                    f"evaluation/success_rate uses {eval_count} independent validation starts. X axis is policy transitions.",0)
    checkpoint=None
    status["evaluations"]=[]
    try:
        for stage in spec["stages"]:
            training=folder/stage["training_dir"]
            end=stage["end_iteration"]
            train=[sys.executable,entry,"train",str(training),"--config",str(folder/"config.json"),
                   "--iterations",str(end-stage["start_iteration"])]
            if checkpoint is not None:
                train += ["--resume",str(checkpoint)]
            elif spec.get("initialize_from"):
                train += ["--initialize-from",spec["initialize_from"]]
            status.update(state="training",phase="training",stage=stage)
            with (folder/f"{stage['training_dir']}.log").open("x") as log:
                p=subprocess.Popen(train,env=env,cwd=source,stdout=log,stderr=subprocess.STDOUT)
                status["training_pid"]=p.pid;save(folder/"status.json",status)
                offset=0
                while True:
                    try:
                        code=p.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        offset=log_progress(writer,training/"progress.jsonl",cfg,offset)
                        continue
                    log_progress(writer,training/"progress.jsonl",cfg,offset)
                    break
            status.pop("training_pid",None)
            if code:
                status.update(state="failed",exit_code=code,finished_at=now())
                save(folder/"status.json",status)
                return code
            checkpoint=training/f"checkpoint_{end:05d}.pt"
            status.update(state="evaluating",phase="evaluation",latest_checkpoint=str(checkpoint))
            save(folder/"status.json",status)
            output=folder/stage["evaluation_dir"]
            evaluate=[sys.executable,entry,"evaluate",str(output),
                      "--config",str(folder/"evaluation_config.json"),"--checkpoint",str(checkpoint)]
            with (folder/f"{stage['evaluation_dir']}.log").open("x") as log:
                p=subprocess.Popen(evaluate,env=env,cwd=source,stdout=log,stderr=subprocess.STDOUT)
                status["evaluation_pid"]=p.pid;save(folder/"status.json",status)
                code=p.wait()
            status.pop("evaluation_pid",None)
            if code:
                status.update(state="failed",exit_code=code,finished_at=now())
                save(folder/"status.json",status)
                return code
            result=json.loads((output/"results.json").read_text())
            step=end*cfg["num_envs"]*cfg["rollout_steps"]
            for key in ("success_rate","successes","total","capture_to_success_rate",
                        "geometrically_legal_success_rate"):
                if isinstance(result.get(key),(int,float)):
                    writer.add_scalar(f"evaluation/{key}",result[key],step)
            for key,value in result.get("episode_peak_force_n_quantiles",{}).items():
                writer.add_scalar(f"evaluation/peak_force_n_{key}",value,step)
            if result.get("study_revision") == 6:
                for key,value in result.get("episode_peak_moment_nm_quantiles",{}).items():
                    writer.add_scalar(f"evaluation/peak_moment_nm_{key}",value,step)
                for key in ("geometry_successes","force_qualified_successes","force_exceed_s_total","moment_exceed_s_total"):
                    writer.add_scalar(f"evaluation/{key}",result[key],step)
            for key,value in zip(("wilson_95_low","wilson_95_high"),result["wilson_95"]):
                writer.add_scalar(f"evaluation/{key}",value,step)
            for shape,values in result.get("by_shape",{}).items():
                for key,value in values.items():
                    if isinstance(value,(int,float)):
                        writer.add_scalar(f"evaluation/{shape}/{key}",value,step)
                for key,value in zip(("wilson_95_low","wilson_95_high"),values["wilson_95"]):
                    writer.add_scalar(f"evaluation/{shape}/{key}",value,step)
            writer.flush()
            status["evaluation"]={key:result[key] for key in ("successes","total","success_rate","wilson_95")}
            if "by_shape" in result:
                status["evaluation"]["by_shape"]=result["by_shape"]
            status["evaluations"].append(dict(iteration=end,transitions=step,
                                               output=str(output),**status["evaluation"]))
            save(folder/"status.json",status)
        if spec.get("test_clearance_per_side_m") is not None:
            transfer_cfg=transfer_evaluation_config(evaluation_cfg,spec["test_clearance_per_side_m"])
            save(folder/"clearance_transfer_config.json",transfer_cfg)
            output=folder/"evaluation_clearance_005mm"
            status.update(state="evaluating",phase="clearance_transfer_evaluation")
            command=[sys.executable,entry,"evaluate",str(output),
                     "--config",str(folder/"clearance_transfer_config.json"),
                     "--checkpoint",str(checkpoint),"--allow-clearance-transfer"]
            with (folder/"evaluation_clearance_005mm.log").open("x") as log:
                p=subprocess.Popen(command,env=env,cwd=source,stdout=log,stderr=subprocess.STDOUT)
                status["evaluation_pid"]=p.pid;save(folder/"status.json",status)
                code=p.wait()
            status.pop("evaluation_pid",None)
            if code:
                status.update(state="failed",exit_code=code,finished_at=now())
                save(folder/"status.json",status)
                return code
            baseline=json.loads((folder/"evaluation/results.json").read_text())
            result=json.loads((output/"results.json").read_text())
            if baseline["checkpoint_sha256"]!=result["checkpoint_sha256"]:
                raise ValueError("clearance comparison must use identical checkpoint weights")
            comparison=dict(zero_shot=True,checkpoint_sha256=result["checkpoint_sha256"],
                training_clearance_per_side_m=cfg["clearance_per_side_m"],
                no_fine_tuning=True,by_clearance={
                    "0.5mm":{k:v for k,v in baseline.items() if k!="cases"},
                    "0.05mm":{k:v for k,v in result.items() if k!="cases"}})
            save(folder/"clearance_comparison.json",comparison)
            status["clearance_transfer_evaluation"]={key:result[key] for key in
                ("successes","total","success_rate","wilson_95","by_shape","failure_reasons",
                 "strict_geometry_success_rate","episode_peak_force_n_quantiles")}
            for shape,values in result["by_shape"].items():
                for key in ("success_rate","strict_geometry_success_rate"):
                    writer.add_scalar(f"clearance_005mm/{shape}/{key}",values[key],step)
            writer.flush()
        status.update(state="completed",exit_code=0,finished_at=now())
        save(folder/"status.json",status)
        return 0
    except Exception as exc:
        status.update(state="failed",error=repr(exc),finished_at=now())
        save(folder/"status.json",status)
        raise
    finally:
        writer.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    parser.add_argument("--config")
    parser.add_argument("--transitions",type=int,help="override config job_iterations; legacy default: 250M")
    parser.add_argument("--eval-at-transitions",type=int,nargs="*",help="override config evaluation schedule")
    parser.add_argument("--test-clearance-per-side-m",type=float,
                        help="final evaluation-only bore clearance; authorized transfer: 0.00005")
    parser.add_argument("--initialize-from",help="initialize actor and critic from a checkpoint; use a fresh optimizer")
    parser.add_argument("--evaluation-per-shape",type=int,default=2048)
    parser.add_argument("--worker",action="store_true")
    args=parser.parse_args()
    folder=Path(args.output).resolve()
    if args.worker:
        return worker(folder)
    if not args.config:
        parser.error("--config required")
    root=Path(__file__).resolve().parents[1]
    cfg=json.loads(Path(args.config).read_text())
    initialize_from=Path(args.initialize_from).resolve() if args.initialize_from else None
    if initialize_from is not None and not initialize_from.is_file():
        parser.error("initialization checkpoint does not exist")
    try:
        if args.test_clearance_per_side_m is not None:
            transfer_evaluation_config(cfg,args.test_clearance_per_side_m)
        args.transitions,args.eval_at_transitions=resolve_budget(cfg,args.transitions,args.eval_at_transitions)
    except ValueError as exc:
        parser.error(str(exc))
    if args.evaluation_per_shape < 1:
        parser.error("positive evaluation count required")
    if cfg["method"]!="study":
        parser.error("this launcher only accepts the study contract")
    # Fail before launching if logging dependencies are unavailable.
    from torch.utils.tensorboard import SummaryWriter
    folder.mkdir(parents=True,exist_ok=False)
    frozen=folder/"frozen"
    shutil.copytree(root/"src",frozen/"src",ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    shutil.copytree(root/"assets",frozen/"assets")
    (frozen/"scripts").mkdir()
    for name in ("blind_contact_ppo.py","blind_study_job.py"):
        shutil.copy2(root/"scripts"/name,frozen/"scripts"/name)
    save(folder/"config.json",cfg)
    iterations=math.ceil(args.transitions/(cfg["num_envs"]*cfg["rollout_steps"]))
    hashes={str(p.relative_to(frozen)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in frozen.rglob("*") if p.is_file()}
    spec=dict(created_at=now(),source_git_head=subprocess.check_output(
        ["git","rev-parse","HEAD"],cwd=root,text=True).strip(),
        requested_transitions=args.transitions,iterations=iterations,
        actual_transitions=iterations*cfg["num_envs"]*cfg["rollout_steps"],
        stages=stage_plan(cfg,args.transitions,args.eval_at_transitions),
        frozen_sha256=hashes,evaluation_seed=10007,
        evaluation_episodes=args.evaluation_per_shape*len(cfg.get("assembly_shapes",["circle"])),
        test_clearance_per_side_m=args.test_clearance_per_side_m,
        initialize_from=str(initialize_from) if initialize_from else None,
        initialize_from_sha256=hashlib.sha256(initialize_from.read_bytes()).hexdigest() if initialize_from else None)
    save(folder/"job.json",spec)
    with (folder/"worker.log").open("x") as log:
        process=subprocess.Popen([sys.executable,str(frozen/"scripts/blind_study_job.py"),
                                  str(folder),"--worker"],cwd=frozen,
                                 stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                                 stdin=subprocess.DEVNULL)
    save(folder/"launch.json",dict(worker_pid=process.pid,launched_at=now()))
    print(json.dumps(dict(output=str(folder),worker_pid=process.pid,
                          iterations=iterations,transitions=spec["actual_transitions"]),ensure_ascii=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
