#!/usr/bin/env python3
"""Dedicated Blind Contact PPO CLI; every invocation uses a new output directory."""
import argparse
import json
from mjlab_contact_prep.blind_contact.config import BlindConfig


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["geometry","probe","integrity","train","evaluate"])
    parser.add_argument("output")
    parser.add_argument("--config")
    parser.add_argument("--num-envs",type=int)
    parser.add_argument("--seed",type=int)
    parser.add_argument("--iterations",type=int,default=100)
    parser.add_argument("--goal-depth-m",type=float)
    parser.add_argument("--xy-min-m",type=float)
    parser.add_argument("--xy-max-m",type=float)
    parser.add_argument("--tilt-deg",type=float)
    parser.add_argument("--checkpoint")
    parser.add_argument("--allow-clearance-transfer",action="store_true")
    parser.add_argument("--allow-seating-transfer",action="store_true")
    group=parser.add_mutually_exclusive_group()
    group.add_argument("--initialize-from")
    group.add_argument("--resume")
    args=parser.parse_args()
    if (args.allow_clearance_transfer or args.allow_seating_transfer) and args.command!="evaluate":
        parser.error("clearance transfer is evaluation-only")
    cfg=BlindConfig.load(args.config,num_envs=args.num_envs,seed=args.seed,
        goal_depth_m=args.goal_depth_m,xy_min_m=args.xy_min_m,xy_max_m=args.xy_max_m,tilt_deg=args.tilt_deg)
    if args.command=="geometry":
        from mjlab_contact_prep.blind_contact.assets import generate_assets
        print(json.dumps(generate_assets(args.output,cfg),indent=2))
    elif args.command=="probe":
        from mjlab_contact_prep.blind_contact.validation import control_probe
        control_probe(args.output,cfg)
    elif args.command=="integrity":
        from mjlab_contact_prep.blind_contact.validation import integrity_probe,study_integrity_probe
        (study_integrity_probe if cfg.method=="study" else integrity_probe)(args.output,cfg)
    elif args.command=="train":
        from mjlab_contact_prep.blind_contact.training import train
        train(args.output,cfg,args.iterations,initialize_from=args.initialize_from,resume=args.resume)
    else:
        if not args.checkpoint:
            parser.error("evaluate requires --checkpoint")
        from mjlab_contact_prep.blind_contact.validation import evaluate
        evaluate(args.output,args.checkpoint,cfg,allow_clearance_transfer=args.allow_clearance_transfer,
                 allow_seating_transfer=args.allow_seating_transfer)


if __name__=="__main__":
    main()
