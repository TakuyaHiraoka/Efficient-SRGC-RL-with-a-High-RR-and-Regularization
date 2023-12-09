import gym
import numpy as np
import torch
import time
import sys
from redq.algos.redq_sac import REDQSACAgent
from redq.algos.core import mbpo_epoches, test_agent
from redq.utils.run_utils import setup_logger_kwargs
from redq.utils.bias_utils import log_bias_evaluation
from redq.utils.logx import EpochLogger

# goal condition envs (gym Robotics envs). TH 20230627.
goal_condition_envs=[
    "FetchReach-v1",
    "FetchPush-v1",
    "FetchSlide-v1",
    "FetchPickAndPlace-v1",
    "HandManipulatePenRotate-v0",
    "HandManipulateEggRotate-v0",
    "HandManipulatePenFull-v0",
    "HandManipulateEggFull-v0",
    "HandManipulateBlockFull-v0",
    "HandManipulateBlockRotateZ-v0",
    "HandManipulateBlockRotateXYZ-v0",
    "HandManipulateBlockRotateParallel-v0",
]

def redq_sac(env_name, seed=0, epochs='mbpo', steps_per_epoch=1000,
             max_ep_len=1000, n_evals_per_epoch=1,
             logger_kwargs=dict(), debug=False,
             # following are agent related hyperparameters
             hidden_sizes=(256, 256),
             replay_size=int(1e6), batch_size=256,
             lr=3e-4, gamma=0.99, polyak=0.995,
             alpha=0.2, auto_alpha=True, target_entropy='mbpo',
             start_steps=5000, delay_update_steps='auto',
             utd_ratio=20, num_Q=10, num_min=2,
             policy_update_delay=20,
             # following are bias evaluation related
             evaluate_bias= True,
             n_mc_eval=5000, #changed by TH 20230817
             n_mc_cutoff=350, reseed_each_epoch=True,

             # added by TH 20211108
             gpu_id = 0, target_drop_rate = 0.0, layer_norm = False,
             method = "redq",

             # added by TH 20230625
             reset_interval=-1,
             # added by TH 20230626
             hindsight_experience_replay=False,
             # added by TH 20230705
             additional_goals=0, #4,
             # added by Th20230726
             boundq=False,
             # added by TH20230728
             stretch=1,
             ):
    """
    :param env_name: name of the gym environment
    :param seed: random seed
    :param epochs: number of epochs to run
    :param steps_per_epoch: number of timestep (datapoints) for each epoch
    :param max_ep_len: max timestep until an episode terminates
    :param n_evals_per_epoch: number of evaluation runs for each epoch
    :param logger_kwargs: arguments for logger
    :param debug: whether to run in debug mode
    :param hidden_sizes: hidden layer sizes
    :param replay_size: replay buffer size
    :param batch_size: mini-batch size
    :param lr: learning rate for all networks
    :param gamma: discount factor
    :param polyak: hyperparameter for polyak averaged target networks
    :param alpha: SAC entropy hyperparameter
    :param auto_alpha: whether to use adaptive SAC
    :param target_entropy: used for adaptive SAC
    :param start_steps: the number of random data collected in the beginning of training
    :param delay_update_steps: after how many data collected should we start updates
    :param utd_ratio: the update-to-data ratio
    :param num_Q: number of Q networks in the Q ensemble
    :param num_min: number of sampled Q values to take minimal from
    :param policy_update_delay: how many updates until we update policy network
    """

    if method == "redq":
        layer_norm = True
        num_Q = 5
    elif method == "sac":
        num_Q = 2
        policy_update_delay = 1
        utd_ratio = 20
    else:
        raise NotImplementedError

    if debug: # use --debug for very quick debugging
        hidden_sizes = [2,2]
        batch_size = 2
        utd_ratio = 2
        num_Q = 3
        max_ep_len = 100
        start_steps = 100
        steps_per_epoch = 100

    if boundq:
        hindsight_experience_replay = True

    # use gpu if available
    device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")
    # set number of epoch
    if epochs == 'mbpo' or epochs < 0:
        # add 20211206
        mbpo_epoches['AntTruncatedObs-v2'] = 300
        mbpo_epoches['HumanoidTruncatedObs-v2'] = 300

        # assign default epochs if not contained in envs. TH 20230627
        if env_name in mbpo_epoches.keys():
            epochs = mbpo_epoches[env_name]
        else:
            # TODO move to more suitable part.
            # overwrite the follows for experiments  TH20230702
            epochs = 30
            steps_per_epoch = 10000 # 10k as 1 epoch
            n_evals_per_epoch = 100 # eval 100 times per epoch to make result scores more reliable.
    steps_per_epoch *= stretch # multiply stretch coef TH 20270728
    total_steps = steps_per_epoch * epochs + 1

    """set up logger"""
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    """set up environment and seeding"""
    env_fn = lambda: gym.make(args.env)
    env, test_env, bias_eval_env = env_fn(), env_fn(), env_fn()
    # seed torch and numpy
    torch.manual_seed(seed)
    np.random.seed(seed)

    # seed environment along with env action space so that everything is properly seeded for reproducibility
    def seed_all(epoch):
        seed_shift = epoch * 9999
        mod_value = 999999
        env_seed = (seed + seed_shift) % mod_value
        test_env_seed = (seed + 10000 + seed_shift) % mod_value
        bias_eval_env_seed = (seed + 20000 + seed_shift) % mod_value
        torch.manual_seed(env_seed)
        np.random.seed(env_seed)
        env.seed(env_seed)
        env.action_space.np_random.seed(env_seed)
        test_env.seed(test_env_seed)
        test_env.action_space.np_random.seed(test_env_seed)
        bias_eval_env.seed(bias_eval_env_seed)
        bias_eval_env.action_space.np_random.seed(bias_eval_env_seed)
    seed_all(epoch=0)

    """prepare to init agent"""
    is_goal_conditioned_env = env_name in goal_condition_envs
    if hindsight_experience_replay:
        assert is_goal_conditioned_env, "hindsight replay works only in goal conditioned envs."
        start_steps = start_steps * (additional_goals+1)
    if is_goal_conditioned_env:
        # get obs and action dimensions
        obs_dim = env.observation_space["observation"].shape[0]
        agoal_dim = env.observation_space["achieved_goal"].shape[0]
        dgoal_dim = env.observation_space["desired_goal"].shape[0]
        target_entropy = 'auto' # overwrite target entropy since MBPO settings do not cover goal-conditioned envs settings. TH
        reward_function = env.env.compute_reward
    else:
        # get obs and action dimensions
        obs_dim = env.observation_space.shape[0]
        agoal_dim = 0
        dgoal_dim = 0
        reward_function = None
    act_dim = env.action_space.shape[0]
    # if environment has a smaller max episode length, then use the environment's max episode length.
    max_ep_len = env._max_episode_steps if max_ep_len > env._max_episode_steps else max_ep_len
    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    # we need .item() to convert it from numpy float to python float
    act_limit = env.action_space.high[0].item()
    # keep track of run time
    start_time = time.time()
    # flush logger (optional)
    sys.stdout.flush()
    #################################################################################################

    """init agent and start training"""
    agent = REDQSACAgent(env_name, obs_dim, act_dim, act_limit, device,
                         hidden_sizes, replay_size, batch_size,
                         lr, gamma, polyak,
                         alpha, auto_alpha, target_entropy,
                         start_steps, delay_update_steps,
                         utd_ratio, num_Q, num_min,
                         policy_update_delay,
                         target_drop_rate=target_drop_rate, layer_norm=layer_norm, # added by TH 20211206 <- bug fix 20211207
                         hindsight_experience_replay=hindsight_experience_replay,
                         # set variables related to goal conditioned envs. added by TH20230627
                         is_goal_conditioned_env=is_goal_conditioned_env, agoal_dim=agoal_dim,
                         dgoal_dim=dgoal_dim, reward_function=reward_function,
                         max_ep_len=max_ep_len,
                         # added by TH 20230705
                         additional_goals=additional_goals,
                         # added by TH 20230726
                         boundq=boundq,
                         )

    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

    for t in range(total_steps):
        # get action from agent
        a = agent.get_exploration_action(o, env)
        # Step the env, get next observation, reward and done signal
        o2, r, d, _ = env.step(a)

        # Very important: before we let agent store this transition,
        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        ep_len += 1
        d = False if ep_len == max_ep_len else d

        # give new data to agent
        agent.store_data(o, a, r, o2, d)

        # let agent update
        if (t + 1) % stretch == 0: # TH20230728
            agent.train(logger)
        # set obs to next obs
        o = o2
        ep_ret += r

        if d or (ep_len == max_ep_len):
            # store episode return and length to logger
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            # reset environment
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

            # do hindsight experience replay. Th2030628
            if hindsight_experience_replay:
                agent.replay_buffer.on_episode_end()

        # End of epoch wrap-up
        if (t+1) % steps_per_epoch == 0:
            epoch = t // steps_per_epoch

            # Test the performance of the deterministic version of the agent.
            test_agent(agent, test_env, max_ep_len, logger,
                       n_eval=n_evals_per_epoch,
                       is_goal_conditioned_env=is_goal_conditioned_env) # add logging here
            if evaluate_bias:
                log_bias_evaluation(bias_eval_env, agent, logger, max_ep_len, alpha, gamma, n_mc_eval, n_mc_cutoff,
                                    # TH 20230817
                                    is_goal_conditioned_env=is_goal_conditioned_env,
                                    boundq=boundq,
                                    )

            # reseed should improve reproducibility (should make results the same whether bias evaluation is on or not)
            if reseed_each_epoch:
                seed_all(epoch)

            """logging"""
            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Time', time.time()-start_time)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('LossQ1', average_only=True)
            logger.log_tabular('LogPi', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('Alpha', with_min_and_max=True)
            logger.log_tabular('LossAlpha', average_only=True)
            logger.log_tabular('PreTanh', with_min_and_max=True)

            if evaluate_bias:
                logger.log_tabular("MCDisRet", with_min_and_max=True)
                logger.log_tabular("MCDisRetEnt", with_min_and_max=True)
                logger.log_tabular("QPred", with_min_and_max=True)
                logger.log_tabular("QBias", with_min_and_max=True)
                logger.log_tabular("QBiasAbs", with_min_and_max=True)
                logger.log_tabular("NormQBias", with_min_and_max=True)
                logger.log_tabular("QBiasSqr", with_min_and_max=True)
                logger.log_tabular("NormQBiasSqr", with_min_and_max=True)
            logger.dump_tabular()

            # flush logged information to disk
            sys.stdout.flush()

        # reset agent. TH 20230625 -> add "* strech".  TH20230728
        if ((t % (reset_interval * stretch) ) == 0) and (reset_interval >= 0):
            agent.reset()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-env', type=str, default='Hopper-v2')
    parser.add_argument('-seed', type=int, default=0)
    parser.add_argument('-epochs', type=int, default=-1) # -1 means use mbpo epochs
    parser.add_argument('-exp_name', type=str, default='redq_sac')
    parser.add_argument('-data_dir', type=str, default='./data/')
    parser.add_argument('-debug', action='store_true')
    # added by TH 20211108
    # use: -info, -gpu_id, -method, -target_drop_rate, -layer_norm
    parser.add_argument("-info", type=str, help="Information or name of the run")
    parser.add_argument("-gpu_id", type=int, default=0, help="GPU device ID to be used in GPU experiment, default is 1e6")
    parser.add_argument("-method", default="redq", choices=["redq", "sac"], help="method, default=redq")

    # added by TH20230626
    parser.add_argument("-her", action="store_true", help="use hindsight experience replay")
    # added by TH 20230714
    parser.add_argument("-boundq", action="store_true", help="bound target-Q at target calculation")
    # added by TH 20230728
    parser.add_argument("-stretch", type=int, default=1, help="coef for stretching experiment while keeping UTD and evaluation / reset interval")
    # added by TH20230705
    parser.add_argument("-reset_interval", type=int, default=-1, help="reset interval w.r.t. number of environment interaction. -1 is no reset. Default is -1")
    parser.add_argument("-additional_goals", type=int, default=4, help="Number of additional goals at data augmentation in HER. Default is 4")

    args = parser.parse_args()

    # modify the code here if you want to use a different naming scheme
    exp_name_full = args.exp_name + '_%s' % args.env

    # override log directory path. TH 20211108
    args.data_dir  = './runs/' + str(args.info) + '/'

    # specify experiment name, seed and data_dir.
    # for example, for seed 0, the progress.txt will be saved under data_dir/exp_name/exp_name_s0
    logger_kwargs = setup_logger_kwargs(exp_name_full, args.seed, args.data_dir)


    redq_sac(args.env, seed=args.seed, epochs=args.epochs,
             logger_kwargs=logger_kwargs, debug=args.debug,
             # added by TH 20211206
             gpu_id=args.gpu_id,
             method=args.method,
             hindsight_experience_replay=args.her,
             # added by TH20230705
             reset_interval=args.reset_interval,
             additional_goals=args.additional_goals,
             # added by TH20230726
             boundq=args.boundq,
             # added by Th20230728
             stretch=args.stretch,
             )

