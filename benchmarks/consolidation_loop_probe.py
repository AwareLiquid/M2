# consolidation_loop_probe.py — 海马固化环探针（入口 C · 持续动力学学习）
# 问题：液态世界模型学完动力学 A 再学 B，离线重放（真实/生成）能否防遗忘 A？
# 三臂：no_replay / real_replay / genreplay（生成式重放：只存 A 的起点，轨迹由模型自己想象）

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import copy

from world_model_probe import LiquidBackbone
from mt_lnn.world_model import PredictiveStateHead
from mt_lnn.imagination import LatentImagination


def gen_osc(n_traj, T, k, dt=0.05, seed=0):
    rng = np.random.default_rng(seed)
    trajs = []
    for _ in range(n_traj):
        x0 = rng.uniform(-1.5, 1.5)
        v0 = rng.uniform(-1.0, 1.0)
        x, v = x0, v0
        states = []
        for _ in range(T):
            states.append([x, v])
            a = -k * x
            v = v + a * dt
            x = x + v * dt
        trajs.append(np.asarray(states, dtype=np.float32))
    return torch.from_numpy(np.stack(trajs))


def make_models(d_model, n_layers, device):
    backbone = LiquidBackbone(2, d_model, n_layers).to(device)
    head = PredictiveStateHead(d_model).to(device)
    return backbone, head


def train_phase(backbone, head, opt, data, steps, batch, log_every=200, tag=""):
    backbone.train()
    head.train()
    n = data.shape[0]
    for step in range(steps):
        idx = torch.randint(0, n, (batch,))
        s = data[idx]
        h = backbone(s)
        _, loss = head(h, compute_loss=True)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(backbone.parameters(), 1.0)
        opt.step()
        if step % log_every == 0:
            print(f"  [{tag}] step {step} loss {loss.item():.4f}", flush=True)


def train_phase2_arms(backbone, head, opt, data_b, data_a, arm, steps, batch, H,
                      decoder=None, starts=None, log_every=200):
    """Phase 2: 学 B，按 arm 决定是否混入 A 的重放（真实数据或模型生成）。"""
    backbone.train()
    head.train()
    imag = LatentImagination(head)
    nb = data_b.shape[0]
    half = batch // 2

    for step in range(steps):
        idx_b = torch.randint(0, nb, (half if arm != "no_replay" else batch,))
        s_b = data_b[idx_b]
        h_b = backbone(s_b)
        _, loss = head(h_b, compute_loss=True)

        if arm == "no_replay":
            pass
        elif arm == "real_replay":
            idx_a = torch.randint(0, data_a.shape[0], (half,))
            s_a = data_a[idx_a]
            h_a = backbone(s_a)
            _, loss_a = head(h_a, compute_loss=True)
            loss = loss + loss_a
        elif arm == "genreplay":
            # 只存 A 的起点 (starts)，轨迹由模型自己生成
            idx_s = torch.randint(0, starts.shape[0], (half,))
            s0 = starts[idx_s]                       # (half, S, 2)
            with torch.no_grad():
                h0 = backbone(s0)                    # (half, S, d)
                imagined = imag.imagine(h0[:, -1], H)
                z_gen = imagined.latents             # (half, H, P)
                s_gen = z_gen @ decoder.weight.T + decoder.bias  # 线性解码回状态
            h_gen = backbone(s_gen)
            _, loss_g = head(h_gen, compute_loss=True)
            loss = loss + loss_g
        else:
            raise ValueError(arm)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(backbone.parameters(), 1.0)
        opt.step()
        if step % log_every == 0:
            print(f"  [{arm}] step {step} loss {loss.item():.4f}", flush=True)


def fit_decoder(backbone, head, data, n_fit=64):
    """线性最小二乘解码器：z (online_proj 隐变量) -> 状态 s。"""
    backbone.eval()
    head.eval()
    with torch.no_grad():
        h = backbone(data[:n_fit])
        z = head.online_proj(h)                      # (n, T, P)
    s = data[:n_fit]                                 # (n, T, 2)
    z_flat = z.reshape(-1, z.shape[-1])
    s_flat = s.reshape(-1, s.shape[-1])
    W = torch.linalg.lstsq(z_flat, s_flat).solution.T  # (2, P)
    dec = nn.Linear(z.shape[-1], 2).to(data.device)
    with torch.no_grad():
        dec.weight.copy_(W)
        dec.bias.copy_(s_flat.mean(0) - z_flat.mean(0) @ W.T)
    return dec


def eval_tracking(backbone, head, data, eval_traj, eval_start, H):
    """对 A 的想象跟踪误差（遗忘度量）——同世界模型探针的判据。"""
    backbone.eval()
    head.eval()
    imag = LatentImagination(head)
    with torch.no_grad():
        s = data[:eval_traj]
        h = backbone(s)
        z_true = []
        for k in range(1, H + 1):
            z_true.append(F.normalize(head.online_proj(h[:, eval_start + k]), dim=-1))
        z_true = torch.stack(z_true, dim=1)          # (B, H, P)
        imagined = imag.imagine(h[:, eval_start], H)
        z_imag = imagined.latents
        z0 = F.normalize(head.online_proj(h[:, eval_start]), dim=-1)
        z_static = z0.unsqueeze(1).repeat(1, H, 1)
        e_imag = (z_imag - z_true).norm(dim=-1).mean(0)
        e_static = (z_static - z_true).norm(dim=-1).mean(0)
    return e_imag.cpu().numpy(), e_static.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--n_traj", type=int, default=128)
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--eval_traj", type=int, default=32)
    ap.add_argument("--eval_start", type=int, default=64)
    ap.add_argument("--H", type=int, default=16)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--steps1", type=int, default=800)    # Phase 1: 学 A
    ap.add_argument("--steps2", type=int, default=800)    # Phase 2: 学 B
    ap.add_argument("--n_starts", type=int, default=32)   # genreplay 只存 32 个 A 起点
    ap.add_argument("--start_len", type=int, default=16)  # 每个起点 16 步
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_a = gen_osc(args.n_traj, args.T, 4.0, seed=args.seed).to(device)  # 动力学 A
    data_b = gen_osc(args.n_traj, args.T, 9.0, seed=args.seed + 1).to(device)  # 动力学 B

    # ---- Phase 1: 学 A ----
    print("=== Phase 1: 学动力学 A (k=4) ===", flush=True)
    backbone, head = make_models(args.d_model, args.n_layers, device)
    opt = torch.optim.Adam(
        list(backbone.parameters()) + [p for p in head.parameters() if p.requires_grad],
        lr=args.lr)
    train_phase(backbone, head, opt, data_a, args.steps1, args.batch, tag="A")
    e_p1, _ = eval_tracking(backbone, head, data_a, args.eval_traj, args.eval_start, args.H)
    print(f"Phase1 后 A 跟踪误差 @H16: {e_p1[15]:.4f} (参考线)", flush=True)

    # ---- 拟合线性解码器（genreplay 用，A 数据拟合一次后冻结）----
    decoder = fit_decoder(backbone, head, data_a)
    starts = data_a[:args.n_starts, :args.start_len]  # 只存的"种子"

    # ---- Phase 2: 学 B，三臂对照 ----
    results = {}
    for arm in ["no_replay", "real_replay", "genreplay"]:
        print(f"=== Phase 2 [{arm}]: 学动力学 B (k=9) ===", flush=True)
        b2 = copy.deepcopy(backbone)
        h2 = copy.deepcopy(head)
        opt2 = torch.optim.Adam(
            list(b2.parameters()) + [p for p in h2.parameters() if p.requires_grad],
            lr=args.lr)
        train_phase2_arms(b2, h2, opt2, data_b, data_a, arm,
                          args.steps2, args.batch, args.H, decoder, starts)
        e_imag, e_static = eval_tracking(b2, h2, data_a, args.eval_traj, args.eval_start, args.H)
        results[arm] = (e_imag, e_static)
        print(f"[{arm}] A 遗忘误差 @H16: {e_imag[15]:.4f} (静止基线 {e_static[15]:.4f})", flush=True)

    # ---- 判定 ----
    e_no = results["no_replay"][0][15]
    e_real = results["real_replay"][0][15]
    e_gen = results["genreplay"][0][15]
    print("=" * 60, flush=True)
    print(f"no_replay   A@H16 err: {e_no:.4f}", flush=True)
    print(f"real_replay A@H16 err: {e_real:.4f}  ({e_real/e_no:.2f}x no_replay)", flush=True)
    print(f"genreplay   A@H16 err: {e_gen:.4f}  ({e_gen/e_no:.2f}x no_replay)", flush=True)
    print(f"genreplay vs real_replay: +{e_gen - e_real:+.4f}", flush=True)
    verdict = (e_real < 0.7 * e_no) and (e_gen < 0.7 * e_no)
    print(f"固化环判定: {'SUPPORTED' if verdict else 'REJECTED'} "
          f"(重放臂均需 < 0.7x no_replay)", flush=True)


if __name__ == "__main__":
    main()
