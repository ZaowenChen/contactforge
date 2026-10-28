"""RSL-RL actor with a 30-frame TCN and training-only contact decoder."""
import torch
from torch import nn
from tensordict import TensorDict
from rsl_rl.models import MLPModel


def as_tensordict(observation):
    return TensorDict(dict(history=observation["history"].flatten(1),
                           critic=observation["critic"],
                           labels=observation["labels"]),
                      batch_size=[observation["history"].shape[0]])


class BlindActor(MLPModel):
    def __init__(self, obs, history_steps=30):
        self.history_steps = history_steps
        super().__init__(obs, {"actor":["history"]}, "actor", 6,
                         hidden_dims=[256,256,128], activation="elu",
                         obs_normalization=False,
                         distribution_cfg={"class_name":"GaussianDistribution",
                                           "init_std":.5, "std_type":"scalar"})
        # Valid convolutions see the complete past window, never future samples.
        self.encoder = nn.Sequential(nn.Conv1d(30,64,5,stride=2),nn.ELU(),
                                     nn.Conv1d(64,64,3,stride=2),nn.ELU(),
                                     nn.AdaptiveAvgPool1d(1),nn.Flatten(),
                                     nn.Linear(64,32),nn.ELU())
        self.contact_decoder = nn.Sequential(nn.Linear(32,128),nn.ELU(),
                                            nn.Linear(128,128),nn.ELU(),nn.Linear(128,64*3))
        # Near-zero mean avoids a randomly chosen permanent motion direction
        # before PPO has observed a positive milestone.
        nn.init.orthogonal_(self.mlp[-1].weight,gain=.01)
        nn.init.zeros_(self.mlp[-1].bias)

    def _get_latent_dim(self):
        return 30+32

    def encode(self, history):
        return self.encoder(history.reshape(-1,self.history_steps,30).transpose(1,2))

    def get_latent(self, obs, masks=None, hidden_state=None):
        history = obs["history"].reshape(-1,self.history_steps,30)
        return torch.cat((history[:,-1],self.encode(history)), -1)

    def contact_logits(self, obs):
        return self.contact_decoder(self.encode(obs["history"])).reshape(-1,64,3)


class RunningNormalizer(nn.Module):
    """Running moments, frozen within each PPO rollout and its optimization."""
    def __init__(self, size):
        super().__init__()
        self.register_buffer("mean",torch.zeros(1,size))
        self.register_buffer("var",torch.ones(1,size))
        self.register_buffer("count",torch.zeros(()))

    @torch.no_grad()
    def update(self, x):
        variance, mean = torch.var_mean(x,dim=0,unbiased=False,keepdim=True)
        total = self.count+x.shape[0]
        delta = mean-self.mean
        fraction = x.shape[0]/total
        self.var.copy_((self.var*self.count+variance*x.shape[0]
                        +delta.square()*self.count*x.shape[0]/total)/total)
        self.mean.add_(delta*fraction)
        self.count.copy_(total)

    def forward(self,x):
        return ((x-self.mean)*torch.rsqrt(self.var+1e-8)).clamp(-10.,10.)


class StudyActor(MLPModel):
    """Architecture verified against the official public inference metadata.

    Six joints replace seven. The duplicate current-frame normalization has
    identical statistics to the newest history row and is shared with it.
    """
    def __init__(self,obs):
        super().__init__(obs,{"actor":["history"]},"actor",6,
                         hidden_dims=[256,256,128],activation="elu",
                         obs_normalization=False,
                         distribution_cfg={"class_name":"GaussianDistribution",
                                           "init_std":.5,"std_type":"scalar"})
        self.obs_normalizer=RunningNormalizer(30*30)
        self.frame_transform=nn.Sequential(nn.Linear(30,32),nn.ReLU(),
                                          nn.Linear(32,32),nn.ReLU())
        self.encoder=nn.Sequential(nn.Conv1d(32,32,9,stride=2),nn.ReLU(),
                                   nn.Conv1d(32,32,5),nn.ReLU(),
                                   nn.Conv1d(32,32,5),nn.ReLU(),
                                   nn.AdaptiveAvgPool1d(3),nn.Flatten(),nn.Linear(96,32))
        self.contact_decoder=nn.Sequential(nn.Linear(32,128),nn.ELU(),
                                           nn.Linear(128,128),nn.ELU(),nn.Linear(128,192))

    def _get_latent_dim(self):
        return 62

    def encode_normalized(self,history):
        return self.encoder(self.frame_transform(history[:,1:]).transpose(1,2))

    def get_latent(self,obs,masks=None,hidden_state=None):
        history=self.obs_normalizer(obs["history"]).reshape(-1,30,30)
        return torch.cat((history[:,-1],self.encode_normalized(history)),-1)

    def contact_logits(self,obs):
        history=self.obs_normalizer(obs["history"]).reshape(-1,30,30)
        return self.contact_decoder(self.encode_normalized(history)).reshape(-1,64,3)

    def refresh_normalization(self,obs):
        self.obs_normalizer.update(obs["history"])


class StudyCritic(MLPModel):
    def __init__(self,obs):
        super().__init__(obs,{"critic":["critic"]},"critic",1,
                         hidden_dims=[256,256,128],activation="elu",obs_normalization=False)
        self.obs_normalizer=RunningNormalizer(obs["critic"].shape[-1])

    def refresh_normalization(self,obs):
        self.obs_normalizer.update(obs["critic"])


def make_models(obs, history_steps=30, method="legacy"):
    if method == "study":
        return StudyActor(obs),StudyCritic(obs)
    actor = BlindActor(obs, history_steps)
    critic = MLPModel(obs, {"critic":["critic"]}, "critic", 1,
                      hidden_dims=[256,256,128], activation="elu", obs_normalization=False)
    return actor, critic
