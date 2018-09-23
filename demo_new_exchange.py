import gym
import gym_exchange
from collections import Counter, defaultdict
import time
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C


if __name__ == '__main__':
    # Create and wrap the environment
    env = gym.make('game-stock-exchange-v0')
    env = DummyVecEnv([lambda: env])

    model = A2C(MlpPolicy, env, ent_coef=0.1, verbose=1)
    model = A2C.load("a2c_gym_exchange", env=env)
    model.learning_rate = 1e-7

    # Train the agent
    model.learn(total_timesteps=10000)

    # Save the agent
    model.save("a2c_gym_exchange")
    # del model  # delete trained model to demonstrate loading

    # Load the trained agent
    # model = A2C.load("a2c_gym_exchange")

    # Enjoy trained agent
    obs = env.reset()
    actions = Counter()
    pnl = defaultdict(float)

    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        actions[action.item()] += 1
        pnl[action.item()] += rewards
        env.render()
        if dones:
            break

    time.sleep(10)
    print('actions : {}'.format(actions))
    print('pnl : {}'.format(pnl))