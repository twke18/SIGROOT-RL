"""
Runner script for Mujoco RL training and testing (HalfCheetah, Humanoid, HumanoidStandup).

Usage:
    # Train with PPO
    python runner.py --config configurations/HalfCheetah_PPO.yaml
    python runner.py --config configurations/Humanoid_PPO.yaml
    python runner.py --config configurations/HumanoidStandup_PPO.yaml

    # Train with FPO
    python runner.py --config configurations/HalfCheetah_FPO.yaml
    python runner.py --config configurations/Humanoid_FPO.yaml

    # Test a saved checkpoint
    python runner.py --config configurations/HalfCheetah_PPO.yaml --mode test --checkpoint checkpoints/PPO_HalfCheetah_epoch0500.pt
"""
import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
import yaml

from envs.mujoco_env import MujocoEnv
from utils.rollout import RolloutBuffer
from utils.logger import Logger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def detect_algo(config_path: str) -> str:
    name = os.path.basename(config_path).upper()
    if "PPO" in name:
        return "PPO"
    if "FPO" in name:
        return "FPO"
    raise ValueError(f"Cannot detect algorithm from config path: {config_path}. Name must contain PPO or FPO.")


def build_policy(algo: str, obs_dim: int, act_dim: int, cfg: dict, device: torch.device):
    policy_cfg = cfg["policy"]
    hidden_dims = policy_cfg["hidden_dims"]
    activation = policy_cfg.get("activation", "tanh")

    if algo == "PPO":
        from policies.gaussian_policy import GaussianPolicy
        policy = GaussianPolicy(obs_dim, act_dim, hidden_dims, activation)
    else:
        from policies.flow_policy import FlowPolicy
        policy = FlowPolicy(
            obs_dim,
            act_dim,
            hidden_dims,
            activation,
            num_flow_samples=policy_cfg.get("num_flow_samples", 8),
            flow_steps=policy_cfg.get("flow_steps", 10),
        )
    return policy.to(device)


def build_algorithm(algo: str, policy, cfg: dict, device: torch.device):
    alg_cfg = cfg["algorithm"]
    common = dict(
        policy=policy,
        learning_rate=alg_cfg["learning_rate"],
        value_loss_coef=alg_cfg["value_loss_coef"],
        epochs_per_update=alg_cfg["epochs_per_update"],
        minibatch_size=alg_cfg["minibatch_size"],
        max_grad_norm=alg_cfg.get("max_grad_norm", 0.5),
        device=device,
    )
    if algo == "PPO":
        from algorithms.ppo import PPO
        return PPO(
            clip_epsilon=alg_cfg["clip_epsilon"],
            entropy_coef=alg_cfg.get("entropy_coef", 0.0),
            **common,
        )
    else:
        from algorithms.fpo import FPO
        return FPO(
            clip_eps=alg_cfg.get("clip_eps", 0.05),
            n_mc=alg_cfg.get("n_mc", 8),
            **common,
        )


def collect_video(policy, env_wrapper: MujocoEnv, device: torch.device, max_steps: int = 1000) -> list:
    """Roll out one episode with the current policy and collect RGB frames."""
    render_env = env_wrapper.make_render_env()
    obs, _ = render_env.reset()
    frames = []
    for _ in range(max_steps):
        frame = render_env.render()
        if frame is not None:
            frames.append(frame)
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action, _, _ = policy(obs_t)
        action_np = action.cpu().numpy()[0]
        obs, _, terminated, truncated, _ = render_env.step(action_np)
        if terminated or truncated:
            break
    render_env.close()
    return frames


def run_test(
    policy,
    env_wrapper: MujocoEnv,
    device: torch.device,
    num_episodes: int = 5,
    max_steps: int = 1000,
    success_threshold: float = 300.0,
):
    """
    Run num_episodes test episodes sequentially.
    Returns (avg_accumulated_reward, success_rate).

    success_threshold is environment-specific and read from the training config.
    """
    render_env = env_wrapper.make_render_env()
    rewards = []
    for _ in range(num_episodes):
        obs, _ = render_env.reset()
        ep_reward = 0.0
        for _ in range(max_steps):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action, _, _ = policy(obs_t)
            action_np = action.cpu().numpy()[0]
            obs, reward, terminated, truncated, _ = render_env.step(action_np)
            ep_reward += reward
            if terminated or truncated:
                break
        rewards.append(ep_reward)
    render_env.close()

    avg_reward = float(np.mean(rewards))
    success_rate = float(np.mean([r >= success_threshold for r in rewards]))
    return avg_reward, success_rate


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: dict, algo: str, device: torch.device, checkpoint_path: str | None = None):
    env_cfg = cfg["env"]
    train_cfg = cfg["training"]
    alg_cfg = cfg["algorithm"]

    env = MujocoEnv(env_cfg["name"], env_cfg["num_envs"], seed=env_cfg.get("seed", 42))
    policy = build_policy(algo, env.obs_dim, env.act_dim, cfg, device)
    algorithm = build_algorithm(algo, policy, cfg, device)
    env_label = env_cfg["name"].split("-")[0]
    logger = Logger(f"{algo}_{env_label}")

    os.makedirs("checkpoints", exist_ok=True)
    start_epoch = 0

    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location=device)
        policy.load_state_dict(ckpt["policy"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Resumed from {checkpoint_path} (epoch {start_epoch})")

    buffer = RolloutBuffer(
        rollout_steps=train_cfg["rollout_steps"],
        num_envs=env_cfg["num_envs"],
        obs_dim=env.obs_dim,
        act_dim=env.act_dim,
        gamma=alg_cfg["gamma"],
        gae_lambda=alg_cfg["gae_lambda"],
        device=device,
    )

    obs = env.reset()

    for epoch in range(start_epoch, train_cfg["total_epochs"]):
        # ----------------------------------------------------------------
        # Rollout collection phase
        # ----------------------------------------------------------------
        policy.eval()
        buffer.reset()
        if algo == "FPO":
            fpo_mc_taus, fpo_mc_epsilons, fpo_old_losses = [], [], []
        for _ in range(train_cfg["rollout_steps"]):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                action, log_prob, value = policy(obs_t)
                if algo == "FPO":
                    mc_taus, mc_epsilons, old_loss = policy.sample_mc_pairs(
                        obs_t, action, n_mc=algorithm.n_mc
                    )
                    fpo_mc_taus.append(mc_taus.cpu())
                    fpo_mc_epsilons.append(mc_epsilons.cpu())
                    fpo_old_losses.append(old_loss.cpu())

            action_np = action.cpu().numpy()
            next_obs, reward, done, _ = env.step(action_np)

            buffer.add(
                obs=obs,
                action=action_np,
                reward=reward,
                done=done,
                value=value.cpu().numpy(),
                log_prob=log_prob.cpu().numpy(),
            )
            obs = next_obs

        # Bootstrap value for the last observation
        with torch.no_grad():
            last_obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            last_values = policy.get_value(last_obs_t).cpu().numpy()
        buffer.compute_returns_and_advantages(last_values)

        avg_rollout_reward = buffer.avg_reward
        logger.log_train_rollout(avg_rollout_reward, epoch)
        print(f"[{algo}] Epoch {epoch:4d} | Rollout avg reward: {avg_rollout_reward:.3f}", flush=True)

        # ----------------------------------------------------------------
        # Learning phase
        # ----------------------------------------------------------------
        policy.train()
        obs_t, actions_t, log_probs_t, returns_t, advantages_t = buffer.get_tensors()
        if algo == "FPO":
            actor_loss, critic_loss = algorithm.update(
                obs_t, actions_t, log_probs_t, returns_t, advantages_t,
                mc_taus=torch.cat(fpo_mc_taus).to(device),
                mc_epsilons=torch.cat(fpo_mc_epsilons).to(device),
                old_fpo_losses=torch.cat(fpo_old_losses).to(device),
            )
        else:
            actor_loss, critic_loss = algorithm.update(
                obs_t, actions_t, log_probs_t, returns_t, advantages_t
            )
        logger.log_train_losses(actor_loss, critic_loss, epoch)
        print(f"[{algo}] Epoch {epoch:4d} | Actor loss: {actor_loss:.4f} | Critic loss: {critic_loss:.4f}", flush=True)

        # ----------------------------------------------------------------
        # Testing phase
        # ----------------------------------------------------------------
        if (epoch + 1) % train_cfg["test_interval"] == 0:
            policy.eval()
            avg_test_reward, success_rate = run_test(
                policy, env, device, num_episodes=train_cfg["test_episodes"],
                success_threshold=train_cfg.get("success_threshold", 300.0),
            )
            logger.log_test(avg_test_reward, success_rate, epoch)
            print(
                f"[{algo}] Epoch {epoch:4d} | TEST reward: {avg_test_reward:.2f} | success: {success_rate:.2%}",
                flush=True,
            )

        # ----------------------------------------------------------------
        # Video capture
        # ----------------------------------------------------------------
        if (epoch + 1) % train_cfg["video_interval"] == 0:
            policy.eval()
            frames = collect_video(policy, env, device)
            logger.save_video(frames, epoch + 1)

        # ----------------------------------------------------------------
        # Checkpoint
        # ----------------------------------------------------------------
        if (epoch + 1) % train_cfg.get("checkpoint_interval", 50) == 0:
            ckpt_path = f"checkpoints/{algo}_epoch{epoch + 1:04d}.pt"
            torch.save({"policy": policy.state_dict(), "epoch": epoch}, ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}", flush=True)

    env.close()
    logger.close()
    print(f"[{algo}] Training complete.")


# ---------------------------------------------------------------------------
# Test-only mode
# ---------------------------------------------------------------------------

def test_only(cfg: dict, algo: str, device: torch.device, checkpoint_path: str):
    env_cfg = cfg["env"]
    train_cfg = cfg["training"]

    env = MujocoEnv(env_cfg["name"], 1, seed=env_cfg.get("seed", 0))
    policy = build_policy(algo, env.obs_dim, env.act_dim, cfg, device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")

    env_label = env_cfg["name"].split("-")[0]
    logger = Logger(f"{algo}_{env_label}")
    avg_reward, success_rate = run_test(
        policy, env, device, num_episodes=train_cfg.get("test_episodes", 5),
        success_threshold=train_cfg.get("success_threshold", 300.0),
    )
    print(f"[{algo}] Test reward: {avg_reward:.2f} | success rate: {success_rate:.2%}")

    frames = collect_video(policy, env, device)
    logger.save_video(frames, epoch=0)

    env.close()
    logger.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Mujoco RL training/testing runner")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint .pt file to load")
    parser.add_argument("--device", default="auto", help="cuda / cpu / auto")
    args = parser.parse_args()

    cfg = load_config(args.config)
    algo = detect_algo(args.config)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    if args.mode == "train":
        train(cfg, algo, device, checkpoint_path=args.checkpoint)
    else:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for --mode test")
        test_only(cfg, algo, device, args.checkpoint)


if __name__ == "__main__":
    main()
