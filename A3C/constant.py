k=4

slot_time=1.0

min_fps=5

max_fps=30

# training parameters

# learning rate
lr = 0.001

# discount factor for rewards
gamma = 0.9

# lambda parameter for GAE
gae_lambda = 1.00

# entropy term coefficient
entropy_coef = 0.01

# value loss coefficient
value_loss_coef = 0.5

max_grad_norm = 50.0

random_seed = 1

# numbers of training processes
num_processes = 4

# number of forward steps in A3C
num_steps = 10

# maximum length of an episode
max_episode_length = 1000000

# use an optimizer without shared momentum
no_shared = False
