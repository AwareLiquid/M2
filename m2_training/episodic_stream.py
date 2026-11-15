"""情节流格式 (episodic stream) — docs/DATA_FORMS.md §2 的统一输入形态。

一个"情节" = 一段有起点、有因果、可重放的经历, 供世界模型线 (JEPA) 与
类脑持续学习线消费。语言主线语料不动 (反模式 2: 勿给语言线造脑型数据)。

三条纪律的代码化 (DATA_FORMS §2):
1. 流式而非语料式: (obs, act, t) 三元组, 时间戳一等公民 — 步级时间戳
   必须严格存在且单调不减 (情节性: 先经历 A 后经历 B, 禁 shuffle)。
2. 多时标层级: 细粒度步 (短 τ) + 事件段 {t0,t1,label} (中 τ) + 情节级
   meta (长 τ / 睡眠巩固单位)。
3. 情节身份: SHA-256 内容指纹 (沿用 text_data.Corpus 的身份检查纪律) —
   identity 由规范化 JSON 决定, 任何字节级改动 (含单步时间戳) 都改变
   身份; 反序列化时强校验, 防静默篡改。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


FORMAT_VERSION = 1


def _identity(payload: dict) -> str:
    """规范化 JSON 的 SHA-256 — 与 text_data.Corpus.open 同一纪律。"""
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Step:
    """细粒度步 (短 τ): (obs, act, t) 三元组, 时间戳必填。"""

    t: float
    obs: list
    act: list | None = None
    reward: float | None = None


@dataclass(frozen=True)
class EventSegment:
    """事件段 (中 τ): 工况/任务级区间 [t0, t1]。"""

    t0: float
    t1: float
    label: str


@dataclass(frozen=True)
class Episode:
    """情节级元信息 (长 τ / 睡眠巩固单位) + 多时标流 + 内容身份。"""

    id: str
    time_base: str
    steps: tuple[Step, ...]
    events: tuple[EventSegment, ...] = ()
    meta: dict = field(default_factory=dict)
    actions: list | None = None  # 世界模型线必填; 纯感知线可空
    reward: float | None = None
    identity: str = ""

    def __post_init__(self):
        if not self.identity:
            object.__setattr__(self, "identity", _identity(self.payload()))

    def payload(self) -> dict:
        """参与身份计算的规范化载荷 (不含 identity 自身)。"""
        return {
            "format_version": FORMAT_VERSION,
            "id": self.id,
            "time_base": self.time_base,
            "steps": [asdict(s) for s in self.steps],
            "events": [asdict(e) for e in self.events],
            "meta": self.meta,
            "actions": self.actions,
            "reward": self.reward,
        }


def to_episode_stream(
    id: str,
    steps: list[dict],
    *,
    time_base: str = "unix_seconds",
    events: list[dict] | None = None,
    meta: dict | None = None,
    actions: list | None = None,
    reward: float | None = None,
) -> Episode:
    """原始记录列表 → 校验后的情节 (DATA_FORMS §2 纪律 1/2 的守门入口)。

    steps 记录格式: {t, obs[, act][, reward]}。拒绝: 空 steps、缺 t/obs、
    时间戳非单调不减 (情节性被 shuffle 破坏 = 反模式 3)、事件段越界。
    """
    if not id:
        raise ValueError("Episode id must be non-empty")
    if not steps:
        raise ValueError("Episode must contain at least one step")
    parsed: list[Step] = []
    last_t = None
    for i, rec in enumerate(steps):
        if "t" not in rec:
            raise ValueError(f"step {i}: missing timestamp t (时间戳一等公民)")
        if "obs" not in rec:
            raise ValueError(f"step {i}: missing obs")
        t = float(rec["t"])
        if last_t is not None and t < last_t:
            raise ValueError(
                f"step {i}: timestamp {t} < previous {last_t} — 流被 shuffle "
                f"(反模式 3); 情节性要求单调不减")
        last_t = t
        parsed.append(Step(t=t, obs=list(rec["obs"]),
                           act=None if rec.get("act") is None else list(rec["act"]),
                           reward=None if rec.get("reward") is None else float(rec["reward"])))
    segs: list[EventSegment] = []
    for i, ev in enumerate(events or []):
        t0, t1 = float(ev["t0"]), float(ev["t1"])
        if t1 < t0:
            raise ValueError(f"event {i}: t1 {t1} < t0 {t0}")
        segs.append(EventSegment(t0=t0, t1=t1, label=str(ev["label"])))
    return Episode(id=id, time_base=time_base, steps=tuple(parsed),
                   events=tuple(segs), meta=dict(meta or {}),
                   actions=actions, reward=reward)


def dump_episode(episode: Episode) -> str:
    """序列化为带身份的 JSON 文本 (落盘格式)。"""
    payload = episode.payload()
    payload["identity"] = episode.identity
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def load_episode(text: str) -> Episode:
    """反序列化并强校验 SHA-256 身份 (沿用 Corpus.open 的 checksum 纪律)。

    任何字节级内容改动 (时间戳/观测/动作/元信息) 都会因指纹不匹配被拒绝,
    防止情节被静默篡改后仍以同一身份进入重放/巩固管线。
    """
    payload = json.loads(text)
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"Unsupported episode format_version: {payload.get('format_version')}")
    claimed = payload.pop("identity", "")
    if not claimed:
        raise ValueError("Episode identity missing — 拒绝无身份情节")
    episode = to_episode_stream(
        payload["id"], payload["steps"], time_base=payload["time_base"],
        events=payload.get("events"), meta=payload.get("meta"),
        actions=payload.get("actions"), reward=payload.get("reward"))
    if episode.identity != claimed:
        raise ValueError(
            f"Episode checksum mismatch: claimed {claimed[:12]}…, "
            f"computed {episode.identity[:12]}…")
    return episode
