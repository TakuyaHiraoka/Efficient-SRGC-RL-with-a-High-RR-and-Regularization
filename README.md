# What is this?

This is the source code to replicate the experiments provided in ``Efficient Sparse-Reward Goal-Conditioned Reinforcement Learning with a High Replay Ratio and Regularization.''

[Demo video](https://drive.google.com/file/d/1UHd7JVPCwFLNFhy1QcycQfwU_nll_yII/view?usp=drive_link)

# USAGE

REDQ: 
```
python main.py -info redq -env FetchPickAndPlace-v1 -seed 0 -gpu_id 0 -method redq -her -additional_goals 0 -stretch 5
```

REDQ+HER: 
```
python main.py -info redq-her -env FetchPickAndPlace-v1 -seed 0 -gpu_id 0 -method redq -her -additional_goals 1 -stretch 5 
```

REDQ+BQ: 
```
python main.py -info redq-bq -env FetchPickAndPlace-v1 -seed 0 -gpu_id 0 -method redq -her -additional_goals 0 -boundq -stretch 5
```

REDQ+HER+BQ: 
```
python main.py -info redq-her-bq -env FetchPickAndPlace-v1 -seed 0 -gpu_id 0 -method redq -her -additional_goals 1 -boundq -stretch 5
```

Reset(9): 
```
python main.py -info reset-9 -env FetchPickAndPlace-v1 -seed 0 -gpu_id 0 -method sac -her -additional_goals 0 -stretch 5 -reset_interval 30000
```

Reset(9)+HER: 
```
python main.py -info reset-9-her -env FetchPickAndPlace-v1 -seed 0 -gpu_id 0 -method sac -her -additional_goals 1 -stretch 5 -reset_interval 30000
```

Reset(9)+BQ: 
```
python main.py -info reset-9-bq -env FetchPickAndPlace-v1 -seed 0 -gpu_id 0 -method sac -her -additional_goals 0 -stretch 5 -reset_interval 30000 -boundq
```


Reset(9)+HER+BQ: 
```
python main.py -info reset-9-her-bq -env FetchPickAndPlace-v1 -seed 0 -gpu_id 0 -method sac -her -additional_goals 1 -stretch 5 -reset_interval 30000 -boundq
```

# Note
The main part of this source code is based on the code in [1]. 
Part of this source code (./customexperiencereplays) is based on the code in [2]. 

[1] https://github.com/watchernyu/REDQ

[2] https://github.com/ymd-h/cpprb
