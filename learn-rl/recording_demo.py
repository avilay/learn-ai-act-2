import logging

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics

logging.basicConfig(level=logging.DEBUG)

# Record the agent after every 250 episodes
training_period = 250

num_training_episodes = 10_000

env = gym.make("CartPole-v1", render_mode="rgb_array")
# env = RecordVideo(
#     env,
#     video_folder="/home/avilay/temp/cart-pole",
#     name_prefix="training",
#     episode_trigger=lambda x: x % training_period == 0
# )
env = RecordEpisodeStatistics(env)

for episode_num in range(num_training_episodes):
    # noinspection PyRedeclaration
    obs, info = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
env.close()

print(f"Episode time taken: {env.time_queue}")
print(f"Episode total rewards: {env.return_queue}")
print(f"Episode lengths: {env.length_queue}")
