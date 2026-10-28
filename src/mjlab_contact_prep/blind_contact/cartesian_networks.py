"""Revision 6 actor: dual-stream frame encoder, contact and pose auxiliary heads."""
import torch
from torch import nn
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from .networks import RunningNormalizer, StudyCritic, as_tensordict, make_models

PROPRIO_DIM = 30
FORCE_DIM = 6


def as_tensordict_v6(observation):
    return TensorDict(dict(history=observation["history"].flatten(1), critic=observation["critic"],
                           labels=observation["labels"], pose=observation["pose"]),
                      batch_size=[observation["history"].shape[0]])


class CartesianActor(MLPModel):
    """Same TCN/actor sizes as the study actor; encoder-F/T streams are summed.

    Actor input is only the 30-frame history of encoder values and the measured
    tip wrench. The contact and pose decoders are training-only auxiliaries.
    """
    def __init__(self, obs):
        self.frame_dim = obs["history"].shape[-1]//30
        if self.frame_dim != PROPRIO_DIM+FORCE_DIM:
            raise ValueError("revision 6 actor expects 36 values per frame")
        super().__init__(obs, {"actor": ["history"]}, "actor", 6,
                         hidden_dims=[256, 256, 128], activation="elu", obs_normalization=False,
                         distribution_cfg={"class_name": "GaussianDistribution",
                                           "init_std": .5, "std_type": "scalar"})
        self.obs_normalizer = RunningNormalizer(30*self.frame_dim)
        self.proprio_fc = nn.Linear(PROPRIO_DIM, 32)
        self.force_fc = nn.Linear(FORCE_DIM, 32)
        self.frame_mlp = nn.Sequential(nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        self.encoder = nn.Sequential(nn.Conv1d(32, 32, 9, stride=2), nn.ReLU(),
                                     nn.Conv1d(32, 32, 5), nn.ReLU(),
                                     nn.Conv1d(32, 32, 5), nn.ReLU(),
                                     nn.AdaptiveAvgPool1d(3), nn.Flatten(), nn.Linear(96, 32))
        self.contact_decoder = nn.Sequential(nn.Linear(32, 128), nn.ELU(),
                                             nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 192))
        self.pose_decoder = nn.Sequential(nn.Linear(32, 64), nn.ELU(), nn.Linear(64, 6))
        nn.init.orthogonal_(self.mlp[-1].weight, gain=.01)
        nn.init.zeros_(self.mlp[-1].bias)

    def _get_latent_dim(self):
        return self.frame_dim+32

    def frame_transform(self, history):
        return self.frame_mlp(self.proprio_fc(history[..., :PROPRIO_DIM])+self.force_fc(history[..., PROPRIO_DIM:]))

    def normalized(self, obs):
        return self.obs_normalizer(obs["history"]).reshape(-1, 30, self.frame_dim)

    def encode_normalized(self, history):
        return self.encoder(self.frame_transform(history[:, 1:]).transpose(1, 2))

    def get_latent(self, obs, masks=None, hidden_state=None):
        history = self.normalized(obs)
        return torch.cat((history[:, -1], self.encode_normalized(history)), -1)

    def contact_logits(self, obs):
        return self.contact_decoder(self.encode_normalized(self.normalized(obs))).reshape(-1, 64, 3)

    def auxiliary(self, obs):
        z = self.encode_normalized(self.normalized(obs))
        return self.contact_decoder(z).reshape(-1, 64, 3), self.pose_decoder(z)

    def refresh_normalization(self, obs):
        self.obs_normalizer.update(obs["history"])


def pose_loss(prediction, target):
    """Pose MSE with symmetry-periodic yaw (normalized period 2), circle masked."""
    mask = torch.cat((torch.ones_like(target[:, :5]), target[:, 6:7]), -1)
    delta = prediction-target[:, :6]
    delta = torch.cat((delta[:, :5], (delta[:, 5:6]+1).remainder(2)-1), -1)
    return (delta.square()*mask).sum()/mask.sum()


def models_for(cfg, obs):
    if cfg.study_revision == 6:
        return CartesianActor(obs), StudyCritic(obs)
    return make_models(obs, cfg.history_steps, cfg.method)


def tensordict_for(cfg):
    return as_tensordict_v6 if cfg.study_revision == 6 else as_tensordict
