"""Evaluation-only v4 seat repair: actor, controller, capture and timing unchanged."""
from types import MethodType
from .seating import circle_events, polygon_events


def apply_seating_ablation(env):
    for group in getattr(env, 'envs', [env]):
        old_events = group.events
        def events(self, points, rotation, error, original=old_events):
            captured, _ = original(points, rotation, error)
            if self.shape == 'circle':
                _, seated = circle_events(points, self.radius, rotation, error, self.cfg)
            else:
                _, seated = polygon_events(self.geometry, points, rotation, error, self.cfg)
            return captured, seated
        group.events = MethodType(events, group)
    return env


def seating_transfer_contract(source, cfg, enabled):
    previous = source.get('seating_protocol', 'legacy_v4')
    changed = previous != cfg.seating_protocol
    if enabled:
        if not (source.get('study_revision') == cfg.study_revision == 4
                and previous == 'legacy_v4' and cfg.seating_protocol == 'chamfer_v1'):
            raise ValueError('seating ablation requires v4 legacy_v4 -> chamfer_v1')
    elif changed:
        raise ValueError('evaluation seating protocol differs; explicit seating transfer required')
    return dict(seating_protocol=cfg.seating_protocol, seating_only_ablation=bool(enabled),
                training_seating_protocol=previous)
