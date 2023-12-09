import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from redq.algos.core import TanhGaussianPolicy, Mlp, soft_update_model1_with_model2, mbpo_target_entropy_dict

from cpprb import ReplayBuffer
from customexperiencereplays.HERforRoboticsEnvs import HindsightReplayBufferForRobotics

def get_probabilistic_num_min(num_mins):
    # allows the number of min to be a float
    floored_num_mins = np.floor(num_mins)
    if num_mins - floored_num_mins > 0.001:
        prob_for_higher_value = num_mins - floored_num_mins
        if np.random.uniform(0, 1) < prob_for_higher_value:
            return int(floored_num_mins+1)
        else:
            return int(floored_num_mins)
    else:
        return num_mins

class REDQSACAgent(object):
    """
    Naive SAC: num_Q = 2, num_min = 2
    REDQ: num_Q > 2, num_min = 2
    MaxMin: num_mins = num_Qs
    """
    def __init__(self, env_name, obs_dim, act_dim, act_limit, device,
                 hidden_sizes=(256, 256), replay_size=int(1e6), batch_size=256,
                 lr=3e-4, gamma=0.99, polyak=0.995,
                 alpha=0.2, auto_alpha=True, target_entropy='mbpo',
                 start_steps=5000, delay_update_steps='auto',
                 utd_ratio=20, num_Q=10, num_min=2,
                 policy_update_delay=20,
                 # added by TH20211206
                 target_drop_rate = 0.0, layer_norm = False,
                 # added by TH20230626
                 hindsight_experience_replay = False,

                 # variables related to goal conditioned envs. added by TH20230627
                 is_goal_conditioned_env=False,
                 agoal_dim=0,
                 dgoal_dim=0,
                 reward_function=None,
                 max_ep_len=1000,

                 # added by TH20230705
                 additional_goals=4,
                 # added by TH20230726
                 boundq=False
                 ):
        # set up networks
        self.policy_net = TanhGaussianPolicy(obs_dim + dgoal_dim, act_dim, hidden_sizes, action_limit=act_limit).to(device)
        self.q_net_list, self.q_target_net_list = [], []
        for q_i in range(num_Q):
            new_q_net = Mlp(obs_dim + dgoal_dim + act_dim, 1, hidden_sizes, target_drop_rate=target_drop_rate, layer_norm=layer_norm).to(device)
            self.q_net_list.append(new_q_net)
            new_q_target_net = Mlp(obs_dim + dgoal_dim + act_dim, 1, hidden_sizes, target_drop_rate=target_drop_rate, layer_norm=layer_norm).to(device)
            new_q_target_net.load_state_dict(new_q_net.state_dict())
            self.q_target_net_list.append(new_q_target_net)
        # set up optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.q_optimizer_list = []
        for q_i in range(num_Q):
            self.q_optimizer_list.append(optim.Adam(self.q_net_list[q_i].parameters(), lr=lr))
        # set up adaptive entropy (SAC adaptive)
        self.auto_alpha = auto_alpha
        if auto_alpha:
            if target_entropy == 'auto':
                self.target_entropy = - act_dim
            if target_entropy == 'mbpo':
                # add 20211206
                mbpo_target_entropy_dict['AntTruncatedObs-v2'] = -4
                mbpo_target_entropy_dict['HumanoidTruncatedObs-v2'] = -2

                self.target_entropy = mbpo_target_entropy_dict[env_name]
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optim = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.cpu().exp().item()
        else:
            self.alpha = alpha
            self.target_entropy, self.log_alpha, self.alpha_optim = None, None, None
        # set up replay buffer. modified by TH 20230626
        if is_goal_conditioned_env:
            env_dict = {"obs1": {"shape":obs_dim, "dtype": np.float32},
                        "acts": {"shape":act_dim, "dtype": np.float32},
                        "obs2": {"shape":obs_dim, "dtype": np.float32},
                        "rews": {"dtype": np.float32}, #
                        "done": {"dtype": np.float32},
                        "agoal1": {"shape": agoal_dim, "dtype": np.float32},
                        "agoal2": {"shape": agoal_dim, "dtype": np.float32},
                        "dgoal1": {"shape": dgoal_dim, "dtype": np.float32},
                        "dgoal2": {"shape": dgoal_dim, "dtype": np.float32},
                       }
        else:
            env_dict = {"obs1": {"shape":obs_dim, "dtype": np.float32},
                        "acts": {"shape":act_dim, "dtype": np.float32},
                        "obs2": {"shape":obs_dim, "dtype": np.float32},
                        "rews": {"dtype": np.float32},
                        "done": {"dtype": np.float32},
                       }

        if hindsight_experience_replay:
            self.replay_buffer = HindsightReplayBufferForRobotics(replay_size, env_dict, max_ep_len, reward_function,
                                                                  additional_goals=additional_goals,
                                                                  prioritized=False,
                                                                  gamma=gamma)
        else:
            self.replay_buffer = ReplayBuffer(replay_size, env_dict)
        # set up other things
        self.mse_criterion = nn.MSELoss()

        # store other hyperparameters
        self.start_steps = start_steps
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_limit = act_limit
        self.lr = lr
        self.hidden_sizes = hidden_sizes
        self.gamma = gamma
        self.polyak = polyak
        self.replay_size = replay_size
        self.alpha = alpha
        self.batch_size = batch_size
        self.num_min = num_min
        self.num_Q = num_Q
        self.utd_ratio = utd_ratio
        self.delay_update_steps = self.start_steps if delay_update_steps == 'auto' else delay_update_steps
        self.policy_update_delay = policy_update_delay
        self.device = device

        # for reset. TH 20230625
        self.target_drop_rate = target_drop_rate
        self.layer_norm = layer_norm

        # for goal conditioned env. TH 20230627
        self.is_goal_conditioned_env = is_goal_conditioned_env
        self.agoal_dim = agoal_dim
        self.dgoal_dim = dgoal_dim
        self.reward_function = reward_function

        # Th 20230728
        self.boundq = boundq

    def __get_current_num_data(self):
        # used to determine whether we should get action from policy or take random starting actions
        return self.replay_buffer.get_stored_size()

    # added by TH. 20230628.
    def _get_obs_goal_torch_tensor_from_unpacked_obs(self, unpacked_np_obs, unsqueeze=True):
        return self._get_obs_goal_torch_tensor(unpacked_np_obs["observation"], unpacked_np_obs["desired_goal"], unsqueeze=unsqueeze)

    def _get_obs_goal_torch_tensor(self, np_obs, np_dgoal, unsqueeze=True):
        if unsqueeze:
            obs_tensor = torch.Tensor(np_obs).unsqueeze(0).to(self.device)
            goal_tensor = torch.Tensor(np_dgoal).unsqueeze(0).to(self.device)
        else:
            obs_tensor = torch.Tensor(np_obs).to(self.device)
            goal_tensor = torch.Tensor(np_dgoal).to(self.device)

        cat_tensor = torch.cat( (obs_tensor, goal_tensor), dim=-1)
        return cat_tensor

    def get_exploration_action(self, obs, env):
        # given an observation, output a sampled action in numpy form
        with torch.no_grad():
            if self.__get_current_num_data() > self.start_steps:
                if self.is_goal_conditioned_env:  # TH20230628
                    obs_tensor = self._get_obs_goal_torch_tensor_from_unpacked_obs(obs)
                else:
                    obs_tensor = torch.Tensor(obs).unsqueeze(0).to(self.device)
                action_tensor = self.policy_net.forward(obs_tensor, deterministic=False, return_log_prob=False)[0]
                action = action_tensor.cpu().numpy().reshape(-1)
            else:
                action = env.action_space.sample()
        return action



    def get_test_action(self, obs):
        # given an observation, output a deterministic action in numpy form
        with torch.no_grad():
            if self.is_goal_conditioned_env: # TH20230628
                obs_tensor = self._get_obs_goal_torch_tensor_from_unpacked_obs(obs)
            else:
                obs_tensor = torch.Tensor(obs).unsqueeze(0).to(self.device)
            action_tensor = self.policy_net.forward(obs_tensor, deterministic=True, return_log_prob=False)[0]
            action = action_tensor.cpu().numpy().reshape(-1)
        return action

    def get_action_and_logprob_for_bias_evaluation(self, obs):
        # given an observation, output a sampled action in numpy form
        with torch.no_grad():
            if self.is_goal_conditioned_env: # TH20230628
                obs_tensor = self._get_obs_goal_torch_tensor_from_unpacked_obs(obs)
            else:
                obs_tensor = torch.Tensor(obs).unsqueeze(0).to(self.device)
            action_tensor, _, _, log_prob_a_tilda, _, _, = self.policy_net.forward(obs_tensor, deterministic=False,
                                         return_log_prob=True)
            action = action_tensor.cpu().numpy().reshape(-1)
        return action, log_prob_a_tilda

    def get_ave_q_prediction_for_bias_evaluation(self, obs_tensor, acts_tensor):
        # given obs_tensor and act_tensor, output Q prediction
        q_prediction_list = []
        for q_i in range(self.num_Q):
            q_prediction = self.q_net_list[q_i](torch.cat([obs_tensor, acts_tensor], 1))
            q_prediction_list.append(q_prediction)
        q_prediction_cat = torch.cat(q_prediction_list, dim=1)
        average_q_prediction = torch.mean(q_prediction_cat, dim=1)
        return average_q_prediction

    def store_data(self, o, a, r, o2, d):
        # store one transition to the buffer
        if self.is_goal_conditioned_env:    # added by TH20230628
            self.replay_buffer.add(obs1=o["observation"], acts=a, rews=r, obs2=o2["observation"], done=d,
                                   agoal1=o["achieved_goal"], agoal2=o2["achieved_goal"],
                                   dgoal1=o["desired_goal"], dgoal2=o2["desired_goal"])
        else:
            self.replay_buffer.add(obs1=o, acts=a, rews=r, obs2=o2, done=d)

    def sample_data(self, batch_size):
        # sample data from replay buffer
        batch = self.replay_buffer.sample(batch_size)
        if self.is_goal_conditioned_env:
            obs_tensor = self._get_obs_goal_torch_tensor(batch["obs1"], batch["dgoal1"], unsqueeze=False)
            obs_next_tensor = self._get_obs_goal_torch_tensor(batch["obs2"], batch["dgoal2"], unsqueeze=False)
        else:
            obs_tensor = Tensor(batch['obs1']).to(self.device)
            obs_next_tensor = Tensor(batch['obs2']).to(self.device)
        acts_tensor = Tensor(batch['acts']).to(self.device)
        rews_tensor = Tensor(batch['rews']).to(self.device)
        done_tensor = Tensor(batch['done']).to(self.device)

        return obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor

    def get_redq_q_target_no_grad(self, obs_next_tensor, rews_tensor, done_tensor):
        # compute REDQ Q target
        num_mins_to_use = get_probabilistic_num_min(self.num_min)
        sample_idxs = np.random.choice(self.num_Q, num_mins_to_use, replace=False)
        with torch.no_grad():
            """Q target is min of a subset of Q values"""
            a_tilda_next, _, _, log_prob_a_tilda_next, _, _ = self.policy_net.forward(obs_next_tensor)
            q_prediction_next_list = []
            for sample_idx in sample_idxs:
                q_prediction_next = self.q_target_net_list[sample_idx](torch.cat([obs_next_tensor, a_tilda_next], 1))
                q_prediction_next_list.append(q_prediction_next)
            q_prediction_next_cat = torch.cat(q_prediction_next_list, 1)

            min_q, min_indices = torch.min(q_prediction_next_cat, dim=1, keepdim=True)
            if self.boundq:  # bounding target-Q. TH20230726.
                # lower bound and uppder bound of Q in Robotics tasks.
                lbq = -1.0 / (1.0 - self.gamma)
                ubq = 0.0
                min_q = torch.clamp(min_q, min=lbq)
                min_q = torch.clamp(min_q, max=ubq)
                # # mean operator without entropy terms version. TH20231101
                # q_prediction_next_ave = q_prediction_next_cat.mean(dim=1).reshape(-1, 1)
                # q_prediction_next_ave = torch.clamp(q_prediction_next_ave, min=lbq)
                # q_prediction_next_ave = torch.clamp(q_prediction_next_ave, max=ubq)
                # next_q_with_log_prob = q_prediction_next_ave # don't include entropy term
            next_q_with_log_prob = min_q - self.alpha * log_prob_a_tilda_next

            y_q = rews_tensor + self.gamma * (1 - done_tensor) * next_q_with_log_prob
        return y_q

    def train(self, logger):
        # this function is called after each datapoint collected.
        # when we only have very limited data, we don't make updates
        num_update = 0 if self.__get_current_num_data() <= self.delay_update_steps else self.utd_ratio
        for i_update in range(num_update):
            obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor = self.sample_data(self.batch_size)

            """Q loss"""
            y_q = self.get_redq_q_target_no_grad(obs_next_tensor, rews_tensor, done_tensor)
            q_prediction_list = []
            for q_i in range(self.num_Q):
                q_prediction = self.q_net_list[q_i](torch.cat([obs_tensor, acts_tensor], 1))
                q_prediction_list.append(q_prediction)
            q_prediction_cat = torch.cat(q_prediction_list, dim=1)
            y_q = y_q.expand((-1, self.num_Q)) if y_q.shape[1] == 1 else y_q
            q_loss_all = self.mse_criterion(q_prediction_cat, y_q) * self.num_Q

            for q_i in range(self.num_Q):
                self.q_optimizer_list[q_i].zero_grad()
            q_loss_all.backward()

            """policy and alpha loss"""
            if ((i_update + 1) % self.policy_update_delay == 0) or i_update == num_update - 1:
                # get policy loss
                a_tilda, mean_a_tilda, log_std_a_tilda, log_prob_a_tilda, _, pretanh = self.policy_net.forward(obs_tensor)
                q_a_tilda_list = []
                for sample_idx in range(self.num_Q):
                    self.q_net_list[sample_idx].requires_grad_(False)
                    q_a_tilda = self.q_net_list[sample_idx](torch.cat([obs_tensor, a_tilda], 1))
                    q_a_tilda_list.append(q_a_tilda)
                q_a_tilda_cat = torch.cat(q_a_tilda_list, 1)
                ave_q = torch.mean(q_a_tilda_cat, dim=1, keepdim=True)
                policy_loss = (self.alpha * log_prob_a_tilda - ave_q).mean()
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                for sample_idx in range(self.num_Q):
                    self.q_net_list[sample_idx].requires_grad_(True)

                # get alpha loss
                if self.auto_alpha:
                    alpha_loss = -(self.log_alpha * (log_prob_a_tilda + self.target_entropy).detach()).mean()
                    self.alpha_optim.zero_grad()
                    alpha_loss.backward()
                    self.alpha_optim.step()
                    self.alpha = self.log_alpha.cpu().exp().item()
                else:
                    alpha_loss = Tensor([0])

            """update networks"""
            for q_i in range(self.num_Q):
                self.q_optimizer_list[q_i].step()

            if ((i_update + 1) % self.policy_update_delay == 0) or i_update == num_update - 1:
                self.policy_optimizer.step()

            # polyak averaged Q target networks
            for q_i in range(self.num_Q):
                soft_update_model1_with_model2(self.q_target_net_list[q_i], self.q_net_list[q_i], self.polyak)

            # by default only log for the last update out of <num_update> updates
            if i_update == num_update - 1:
                logger.store(LossPi=policy_loss.cpu().item(), LossQ1=q_loss_all.cpu().item() / self.num_Q,
                             LossAlpha=alpha_loss.cpu().item(), Q1Vals=q_prediction.detach().cpu().numpy(),
                             Alpha=self.alpha, LogPi=log_prob_a_tilda.detach().cpu().numpy(),
                             PreTanh=pretanh.abs().detach().cpu().numpy().reshape(-1))

        # if there is no update, log 0 to prevent logging problems
        if num_update == 0:
            logger.store(LossPi=0, LossQ1=0, LossAlpha=0, Q1Vals=np.zeros((1,1), dtype=np.float32),
                         Alpha=0, LogPi=np.zeros((1,1), dtype=np.float32), PreTanh=np.zeros((self.act_dim), dtype=np.float32))

    # periodically reset networks and their optimizers. TH20231101
    def reset(self):
        # set up networks
        self.policy_net = TanhGaussianPolicy(self.obs_dim + self.dgoal_dim, self.act_dim, self.hidden_sizes, action_limit=self.act_limit).to(self.device)
        self.q_net_list, self.q_target_net_list = [], []
        for q_i in range(self.num_Q):
            new_q_net = Mlp(self.obs_dim + self.dgoal_dim + self.act_dim, 1, self.hidden_sizes, target_drop_rate=self.target_drop_rate, layer_norm=self.layer_norm).to(self.device)
            self.q_net_list.append(new_q_net)
            new_q_target_net = Mlp(self.obs_dim + self.dgoal_dim + self.act_dim, 1, self.hidden_sizes, target_drop_rate=self.target_drop_rate, layer_norm=self.layer_norm).to(self.device)
            new_q_target_net.load_state_dict(new_q_net.state_dict())
            self.q_target_net_list.append(new_q_target_net)
        # set up optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.q_optimizer_list = []
        for q_i in range(self.num_Q):
            self.q_optimizer_list.append(optim.Adam(self.q_net_list[q_i].parameters(), lr=self.lr))
        # set up adaptive entropy
        if self.auto_alpha:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = optim.Adam([self.log_alpha], lr=self.lr)
            self.alpha = self.log_alpha.cpu().exp().item()
        else:
            self.alpha = self.alpha
            self.target_entropy, self.log_alpha, self.alpha_optim = None, None, None
