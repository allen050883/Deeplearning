import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.utils import common
from tf_agents.networks import sequential
from tf_agents.specs import tensor_spec
from tf_agents.environments import tf_py_environment
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.environments import utils
from tf_agents.trajectories import trajectory
from tf_agents.metrics import tf_metrics
from tf_agents.drivers import dynamic_step_driver
from schedule_env import ScheduleEnv


# GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Setting environment
schedule_env = ScheduleEnv()
test_env = ScheduleEnv()
utils.validate_py_environment(schedule_env, episodes=5)

train_env = tf_py_environment.TFPyEnvironment(schedule_env)
test_env = tf_py_environment.TFPyEnvironment(test_env)

# Hyperparameter
replay_buffer_capacity = 1000
learning_rate = 0.001
collect_steps_per_iteration = 1
log_interval = 200  # @param {type:"integer"}
num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 1000
fc_layer_params = (128, 64)
action_tensor_spec = tensor_spec.from_spec(train_env.action_spec())
num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1


# Collect input data into buffer
def collect_step(environment, policy, buffer):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)
    buffer.add_batch(traj)


# For calculating reward for one turn
def compute_avg_return(environment, policy, num_episodes=10):
    # time_step: dict{float32 discount, array observation, float32 reward, int32 step_type}
    total_return = 0.0
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0

        # Doing the action if the not finish the episode
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


# Testing the agent
def test_agent(policy, env, num_episodes=5):
    for i in range(num_episodes):
        time_step = env.reset()
        print("num_episodes--->", i)
        while not time_step.is_last():
            print(time_step.observation)
            action_step = policy.action(time_step)
            time_step = env.step(action_step.action)


## Q network
# Define a helper function to create Dense layers configured with the right
# activation and kernel initializer.
# def dense_layer(num_units):
#     return tf.keras.layers.Dense(
#         num_units,
#         activation=tf.keras.activations.tanh,
#         kernel_initializer=tf.keras.initializers.GlorotNormal)

def dense_layer(num_units):
    return tf.keras.layers.Dense(
        num_units,
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.keras.initializers.HeNormal)


# QNetwork consists of a sequence of Dense layers followed by a dense layer
# with `num_actions` units to generate one q_value per available action as
# its output.
dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
q_values_layer = tf.keras.layers.Dense(
    num_actions,
    activation=None,
    kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.03, maxval=0.03),
    bias_initializer=tf.keras.initializers.Constant(-0.2))
convs = tf.keras.layers.Conv1D(filters=16, kernel_size=3,activation='relu')
q_net = sequential.Sequential([convs] + dense_layers + [q_values_layer])
# optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
optimizer = tf.keras.optimizers.RMSprop(
    learning_rate=learning_rate, rho=0.9, momentum=0.9, epsilon=1e-07, centered=True, name='RMSprop')
train_step_counter = tf.Variable(0)

# agent = dqn_agent.DqnAgent(
#     train_env.time_step_spec(),
#     train_env.action_spec(),
#     q_network=q_net,
#     optimizer=optimizer,
#     td_errors_loss_fn=common.element_wise_squared_loss,
#     train_step_counter=train_step_counter)

agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter,
    epsilon_greedy=0.1,
    reward_scale_factor=1.0,  # default
    gradient_clipping=1.0
)

agent.initialize()

random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                train_env.action_spec())

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_capacity)
dataset = replay_buffer.as_dataset(
    num_parallel_calls=10,
    sample_batch_size=64,
    num_steps=2).prefetch(3)

iterator = iter(dataset)

agent.train = common.function(agent.train)

# Reset the train step.
agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
# avg_return = compute_avg_return(train_env, agent.policy, 100)
returns = []

# Reset the environment.
time_step = train_env.reset()

# batch_size = 64
# for _ in range(batch_size):
#     collect_step(train_env, agent.policy, replay_buffer)

env_steps = tf_metrics.EnvironmentSteps()
observers = [replay_buffer.add_batch, env_steps]

driver = dynamic_step_driver.DynamicStepDriver(
    train_env, random_policy, observers, num_steps=1)
batch_size = 64
for _ in range(batch_size):
    driver.run(time_step)

for _ in range(60000):
    # Collect a few steps and save to the replay buffer.
    # time_step, _ = collect_driver.run(time_step)
    time_step, _ = driver.run()
    # collect_step(train_env, agent.collect_policy, replay_buffer)
    experience, unused_info = next(iterator)
    train_loss = agent.train(experience).loss

    step = agent.train_step_counter.numpy()

    if step % log_interval == 0:
        print('step = {0}: loss = {1}'.format(step, train_loss))

    if step % eval_interval == 0:
        # avg_return = compute_avg_return(test_env, agent.policy, num_eval_episodes)
        avg_return = compute_avg_return(test_env, agent.policy, num_eval_episodes)
        print('step = {0}: Average Return = {1}'.format(step, avg_return))
        returns.append(avg_return)

    if step % 5000 == 0:
        # test_agent(agent.policy, test_env)
        test_agent(agent.policy, test_env)

# test_agent(agent.policy, test_env)
test_agent(random_policy, test_env)
