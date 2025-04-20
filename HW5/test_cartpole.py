import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import os
import argparse

class DQN(nn.Module):
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 128),  
            nn.ReLU(),
            nn.Linear(128, 128),  
            nn.ReLU(),
            nn.Linear(128, num_actions)  
        )

    def forward(self, x):
        return self.network(x)

def evaluate(args):
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 创建环境
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)

    # 创建模型
    num_actions = env.action_space.n
    model = DQN(num_actions).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # 存储评估结果
    total_rewards = []

    # 多回合评估
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        total_reward = 0

        while not done:
            # 转换状态为张量
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)

            # 选择动作
            with torch.no_grad():
                action = model(obs_tensor).argmax().item()

            # 执行动作
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        total_rewards.append(total_reward)
        print(f"Episode {ep + 1}: Total Reward = {total_reward}")

    # 计算统计信息
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)

    print("\n--- Test Results ---")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Reward Standard Deviation: {std_reward:.2f}")
    print(f"Min Reward: {np.min(total_rewards)}")
    print(f"Max Reward: {np.max(total_rewards)}")

    return total_rewards

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained .pt model")
    parser.add_argument("--episodes", type=int, default=30, help="Number of test episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for evaluation")
    args = parser.parse_args()

    evaluate(args)