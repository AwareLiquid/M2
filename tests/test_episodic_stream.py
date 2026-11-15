"""episodic_stream 测试 — DATA_FORMS §2 纪律的代码化验证。"""

import json

import pytest

from m2_training.episodic_stream import (
    dump_episode,
    load_episode,
    to_episode_stream,
)


def _records(n=4, t0=100.0, dt=0.5):
    return [{"t": t0 + i * dt, "obs": [float(i), 1.0], "act": [i % 3],
             "reward": 0.1 * i} for i in range(n)]


def test_to_episode_stream_happy_path_and_identity():
    ep = to_episode_stream("ep-1", _records(), meta={"task": "nchain-0"},
                           events=[{"t0": 100.0, "t1": 101.5, "label": "phase-A"}],
                           actions=[[0]], reward=1.0)
    assert ep.id == "ep-1"
    assert len(ep.steps) == 4
    assert ep.identity and len(ep.identity) == 64
    # 身份确定性: 同内容同身份
    again = to_episode_stream("ep-1", _records(), meta={"task": "nchain-0"},
                              events=[{"t0": 100.0, "t1": 101.5, "label": "phase-A"}],
                              actions=[[0]], reward=1.0)
    assert again.identity == ep.identity


def test_identity_is_content_sensitive():
    base = to_episode_stream("ep-1", _records())
    tampered = _records()
    tampered[2]["t"] += 0.001  # 单步时间戳改动必须改变身份
    assert to_episode_stream("ep-1", tampered).identity != base.identity
    tampered_obs = _records()
    tampered_obs[0]["obs"] = [9.9, 1.0]
    assert to_episode_stream("ep-1", tampered_obs).identity != base.identity


def test_rejects_missing_timestamp_or_obs():
    bad_t = [{"obs": [1.0]}]
    with pytest.raises(ValueError, match="timestamp"):
        to_episode_stream("ep", bad_t)
    bad_obs = [{"t": 1.0}]
    with pytest.raises(ValueError, match="obs"):
        to_episode_stream("ep", bad_obs)


def test_rejects_non_monotonic_timestamps():
    shuffled = [{"t": 101.0, "obs": [0.0]}, {"t": 100.0, "obs": [1.0]}]
    with pytest.raises(ValueError, match="shuffle"):
        to_episode_stream("ep", shuffled)


def test_rejects_empty_steps_and_bad_events():
    with pytest.raises(ValueError, match="at least one step"):
        to_episode_stream("ep", [])
    with pytest.raises(ValueError, match="t1"):
        to_episode_stream("ep", _records(2),
                          events=[{"t0": 5.0, "t1": 4.0, "label": "x"}])


def test_json_round_trip_preserves_identity():
    ep = to_episode_stream("ep-9", _records(3), meta={"seed": 0})
    text = dump_episode(ep)
    loaded = load_episode(text)
    assert loaded.identity == ep.identity
    assert loaded.steps == ep.steps
    assert loaded.meta == ep.meta


def test_load_rejects_tampered_payload():
    ep = to_episode_stream("ep-9", _records(3))
    payload = json.loads(dump_episode(ep))
    payload["steps"][1]["obs"] = [7.7, 7.7]  # 落盘后静默篡改
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_episode(json.dumps(payload))


def test_load_rejects_missing_identity_and_bad_version():
    ep = to_episode_stream("ep-9", _records(2))
    payload = json.loads(dump_episode(ep))
    payload.pop("identity")
    with pytest.raises(ValueError, match="identity missing"):
        load_episode(json.dumps(payload))
    payload = json.loads(dump_episode(ep))
    payload["format_version"] = 99
    with pytest.raises(ValueError, match="format_version"):
        load_episode(json.dumps(payload))
