#!/usr/bin/env python3
"""Revision 6 clearance curriculum: 0.5 mm stage A, zero-shot test, then 0.05 mm stage B from A."""
import argparse
import fcntl
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from zoneinfo import ZoneInfo


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def validate_plan(a, b, transitions):
    from mjlab_contact_prep.blind_contact.config import BlindConfig
    for config in (a,b):
        cfg=BlindConfig(**config).validate()
        if cfg.study_revision != 6 or cfg.asset_variant != "passive_xyz_v1":
            raise ValueError("A and B must use passive-asset revision 6")
    if a["clearance_per_side_m"] != .0005 or b["clearance_per_side_m"] != .00005:
        raise ValueError("A must train at 0.5 mm and B at 0.05 mm")
    permitted = {"clearance_per_side_m", "seed", "learning_rate"}
    changed = {key for key in set(a) | set(b) if a.get(key) != b.get(key)}
    if changed - permitted:
        raise ValueError(f"A/B control or training schedule differs: {sorted(changed-permitted)}")
    if a["assembly_shapes"] != ["circle", "square", "hexagon"]:
        raise ValueError("the A/B experiment requires all three shapes")
    batch = a["num_envs"] * a["rollout_steps"]
    if transitions <= 0 or transitions % batch:
        raise ValueError("equal sample budgets must contain whole PPO rollouts")


def launch_job(entry, output, config, budget, evaluations, *, initialize_from=None, transfer=False, evaluation_per_shape=2048):
    source = entry.parents[1]
    env = dict(os.environ, PYTHONPATH=str(source / "src"), PYTHONDONTWRITEBYTECODE="1")
    command = [sys.executable, str(entry), str(output), "--config", str(config),
               "--transitions", str(budget), "--evaluation-per-shape", str(evaluation_per_shape),
               "--eval-at-transitions", *map(str, evaluations)]
    if transfer:
        command += ["--test-clearance-per-side-m", ".00005"]
    if initialize_from is not None:
        command += ["--initialize-from", str(initialize_from)]
    result = subprocess.run(command, cwd=source, env=env, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"job launcher failed: {result.stdout}\n{result.stderr}")
    return json.loads(result.stdout.splitlines()[-1])


def wait_for_job(folder, parent, status, label):
    previous = None
    while True:
        path = folder / "status.json"
        if path.exists():
            child = json.loads(path.read_text())
            status.update(state=f"{child['state']}_{label}", active_model=label,
                          updated_at=now(), child_status=child)
            save(parent / "status.json", status)
            phase = (child["state"], child.get("phase"), child.get("stage", {}).get("end_iteration"))
            if phase != previous:
                print(now(), label, phase, flush=True)
                previous = phase
            if child["state"] == "failed":
                raise RuntimeError(f"model {label} failed; see {folder}/status.json")
            if child["state"] == "completed":
                return child
        time.sleep(5)


def compare_results(a, b, source_sha):
    if a["checkpoint_sha256"] == b["checkpoint_sha256"]:
        raise ValueError("B must be an independently trained checkpoint")
    if a["checkpoint_sha256"] != source_sha:
        raise ValueError("B initialization is not the zero-shot A checkpoint")
    for result in (a, b):
        if result["test_clearance_per_side_m"] != .00005:
            raise ValueError("the A/B comparison requires the same narrow clearance")
    indexed_a = {(r["shape"], r["case_id"]): r for r in a["cases"]}
    indexed_b = {(r["shape"], r["case_id"]): r for r in b["cases"]}
    if indexed_a.keys() != indexed_b.keys():
        raise ValueError("A/B evaluation case IDs or shapes differ")
    by_shape = {}
    for shape in a["by_shape"]:
        keys = [key for key in indexed_a if key[0] == shape]
        outcomes = [(indexed_a[k]["outcome"] == "success", indexed_b[k]["outcome"] == "success") for k in keys]
        by_shape[shape] = dict(
            A=a["by_shape"][shape], B=b["by_shape"][shape],
            success_rate_change=b["by_shape"][shape]["success_rate"]-a["by_shape"][shape]["success_rate"],
            paired_outcomes=dict(both_success=sum(x and y for x, y in outcomes),
                improved=sum(not x and y for x, y in outcomes),
                lost=sum(x and not y for x, y in outcomes),
                both_failure=sum(not x and not y for x, y in outcomes)))
    return dict(created_at=now(), training_samples_A=a.get("training_samples"),
                training_samples_B=b.get("training_samples"), test_clearance_per_side_m=.00005,
                A_checkpoint_sha256=a["checkpoint_sha256"], B_checkpoint_sha256=b["checkpoint_sha256"],
                B_initialized_from_A_sha256=source_sha, by_shape=by_shape)


def write_report(parent, comparison, baseline, a, b):
    lines=["# revision 6 课程：0.5 mm A → 0.05 mm B", "",
           "共享圆/方/六边形策略；轴尖笛卡尔动作 + F/T 限力调节器、36 维观测、位姿回归辅助头、稠密奖励。",
           "A 的同一权重先完成两间隙评估，B 再从 A 初始化，以新优化器训练相同新增样本。",
           "峰值力为 1 kHz 补偿腕部 F/T 换算到名义轴尖的合力（含惯性残差），不是接触法向力之和。", "",
           "| 模型/间隙 | 形状 | 成功/总数 | 成功率 | 末端深度中位数 mm | 最大深度中位数 mm | 采样力峰值 p95 N |",
           "|---|---|---:|---:|---:|---:|---:|"]
    for label,result in (("A/0.5 mm",baseline),("A/0.05 mm 零微调",a),("B/0.05 mm",b)):
        for shape,row in result["by_shape"].items():
            lines.append(f"| {label} | {shape} | {row['successes']}/{row['total']} | {row['success_rate']:.2%} | "
                         f"{row['depth_at_end_m_quantiles']['p50']*1000:.3f} | "
                         f"{row['episode_max_depth_m_quantiles']['p50']*1000:.3f} | "
                         f"{row['episode_peak_force_n_quantiles']['p95']:.3f} |")
    lines += ["", "成功：1 ms采样下底边位于40 mm±1 mm座深且整轴合法（25 μm），底边使用倒角裕度；本回合无合力/力矩越限。同拍越限优先判故障，首次终态取消旧参考；无保持计时。",
              "窄间隙另列终态5 μm几何合格率，仅作补充指标。三形状均参与训练，不作为未知形状泛化证据。",
              f"A/B各新增样本：{comparison['training_samples_A']:,}。",
              f"A SHA256：`{comparison['A_checkpoint_sha256']}`。",
              f"B SHA256：`{comparison['B_checkpoint_sha256']}`。",
              f"B初始化来源：`{comparison['B_initialized_from_A_sha256']}`。",
              f"评估初态配对核验：{comparison['paired_initial_states']}。",
              "逐形状置信区间、几何合格率、失败原因及逐回合数据见 JSON。", ""]
    (parent/"REPORT.md").write_text("\n".join(lines))


def worker(parent):
    import numpy as np
    plan = json.loads((parent / "plan.json").read_text())
    lock = open("/tmp/mjlab_contact_20n_v6_curriculum.lock", "a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status = dict(state="starting_A", started_at=now(), supervisor_pid=os.getpid())
    save(parent / "status.json", status)
    try:
        afolder = parent / "A_05mm"
        status["A_launch"] = launch_job(Path(plan["source_repo"]) / "scripts/blind_study_job.py", afolder,
            parent / "config_A.json", plan["samples_per_model"], plan["evaluation_samples"], transfer=True, evaluation_per_shape=plan["evaluation_per_shape"])
        astatus = wait_for_job(afolder, parent, status, "A")
        a = json.loads((afolder / "evaluation_clearance_005mm/results.json").read_text())
        baseline = json.loads((afolder / "evaluation/results.json").read_text())
        if a["checkpoint_sha256"] != baseline["checkpoint_sha256"]:
            raise ValueError("A zero-shot test must use the final baseline checkpoint")
        checkpoint = Path(astatus["latest_checkpoint"])
        bfolder = parent / "B_005mm"
        # Reuse A's frozen implementation and assets even if the working tree changes later.
        status["B_launch"] = launch_job(afolder / "frozen/scripts/blind_study_job.py", bfolder,
            parent / "config_B.json", plan["samples_per_model"], plan["evaluation_samples"],
            initialize_from=checkpoint, evaluation_per_shape=plan["evaluation_per_shape"])
        wait_for_job(bfolder, parent, status, "B")
        b = json.loads((bfolder / "evaluation/results.json").read_text())
        bspec = json.loads((bfolder / "job.json").read_text())
        comparison = compare_results(a, b, bspec["initialize_from_sha256"])
        initial_differences = {}
        with np.load(afolder / "evaluation_clearance_005mm/trajectory.npz") as x, np.load(bfolder / "evaluation/trajectory.npz") as y:
            for key in ("initial_qpos", "initial_mocap_pos", "initial_mocap_quat"):
                initial_differences[key] = float(np.max(np.abs(x[key]-y[key])))
        comparison.update(training_samples_A=plan["samples_per_model"], training_samples_B=plan["samples_per_model"],
                          initial_state_max_abs_differences=initial_differences,
                          paired_initial_states=all(v <= 1e-7 for v in initial_differences.values()))
        if not comparison["paired_initial_states"]:
            for row in comparison["by_shape"].values():
                row.pop("paired_outcomes")
        save(parent / "A_vs_B_clearance_005mm.json", comparison)
        write_report(parent, comparison, baseline, a, b)
        status.update(state="completed", finished_at=now(), exit_code=0,
                      report=str(parent / "REPORT.md"), comparison=str(parent / "A_vs_B_clearance_005mm.json"))
        save(parent / "status.json", status)
        print(now(), "A_B_ALL_TRAINING_AND_VALIDATION_COMPLETED", flush=True)
        return 0
    except Exception as exc:
        status.update(state="failed", error=repr(exc), finished_at=now())
        save(parent / "status.json", status)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    parser.add_argument("--config-a")
    parser.add_argument("--config-b")
    parser.add_argument("--samples-per-model", type=int)
    parser.add_argument("--evaluation-per-shape", type=int, default=2048)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    parent = Path(args.output).resolve()
    if args.worker:
        return worker(parent)
    if not args.config_a or not args.config_b:
        parser.error("--config-a and --config-b required")
    a = json.loads(Path(args.config_a).read_text())
    b = json.loads(Path(args.config_b).read_text())
    if args.samples_per_model is None:
        args.samples_per_model=a["num_envs"]*a["rollout_steps"]*a["job_iterations"]
    if args.evaluation_per_shape < 1:
        parser.error("positive evaluation count required")
    validate_plan(a, b, args.samples_per_model)
    parent.mkdir(parents=True, exist_ok=False)
    save(parent / "config_A.json", a)
    save(parent / "config_B.json", b)
    supervisor = parent / "supervisor.py"
    shutil.copy2(__file__, supervisor)
    plan = dict(created_at=now(), source_repo=str(Path(__file__).resolve().parents[1]),
                samples_per_model=args.samples_per_model, samples_per_shape=args.samples_per_model // 3,
                evaluation_samples=[n*a["num_envs"]*a["rollout_steps"]
                    for n in a["job_evaluation_iterations"]
                    if n*a["num_envs"]*a["rollout_steps"] < args.samples_per_model],
                evaluation_per_shape=args.evaluation_per_shape,
                A_training_clearance_per_side_m=.0005, B_training_clearance_per_side_m=.00005,
                B_initialize_from="A final checkpoint after both A evaluations complete",
                supervisor_sha256=hashlib.sha256(supervisor.read_bytes()).hexdigest())
    save(parent / "plan.json", plan)
    with (parent / "supervisor.log").open("x") as log:
        process = subprocess.Popen([sys.executable, str(supervisor), str(parent), "--worker"],
            cwd=parent, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, stdin=subprocess.DEVNULL)
    launch = dict(output=str(parent), supervisor_pid=process.pid, launched_at=now())
    save(parent / "launch.json", launch)
    print(json.dumps(launch, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
