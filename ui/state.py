"""全局会话状态与轻量持久化。

所有页面共享 ``st.session_state``；生成历史与对话存档另外落盘到
``.demo_state/*.json``，浏览器刷新后仍能恢复。模型凭证由 ``.env`` 提供，
这里只保存会话内的临时覆盖值，不落盘。全程不引入数据库：写入失败时
降级为纯内存会话。
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

STATE_DIR = Path(__file__).resolve().parent.parent / ".demo_state"
_HISTORY_FILE = STATE_DIR / "history.json"
_CHAT_FILE = STATE_DIR / "chat.json"

# 生成历史与对话存档两个桶，读写语义保持一致
HISTORY_BUCKETS = {
    "experiment_history": "exp_history",
    "analysis_history": "an_history",
}


def _read_json(path: Path, default: Any) -> Any:
    """读取 JSON；文件缺失或损坏时返回默认值。"""
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return default


def _write_json(path: Path, payload: Any) -> None:
    """原子性尽量简单：直接覆盖写，失败不抛异常。"""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
    except OSError:
        pass


# --- 初始化 ---


def init_state() -> None:
    """补齐所有页面依赖的 session_state 键（幂等）。"""
    defaults: Dict[str, Any] = {
        # 实验方案页
        "exp_input": "",
        "exp_result": None,
        "exp_pending": False,
        "exp_history": [],
        # 数据分析页
        "an_result": None,
        "an_preview": None,
        "an_pending": False,
        "an_csv_path": None,
        "an_frame": None,
        "an_merged_key": None,
        "an_filenames": [],
        "an_summaries": [],
        "an_history": [],
        # 问答页
        "chat_messages": [],
        "chat_threads": [],
        "chat_pending": "",
        "chat_active_thread": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


# --- 生成历史 ---


def _bucket_key(kind: str) -> str:
    return HISTORY_BUCKETS.get(kind, "an_history")


def load_history_list(kind: str = "analysis_history") -> List[Dict[str, Any]]:
    """返回历史记录列表（时间倒序）。首次调用时从磁盘恢复。"""
    key = _bucket_key(kind)
    if key not in st.session_state:
        st.session_state[key] = _read_json(_HISTORY_FILE, {}).get(kind, [])
    return list(st.session_state[key])


def _persist_history() -> None:
    payload = {
        kind: st.session_state.get(bucket, [])
        for kind, bucket in HISTORY_BUCKETS.items()
    }
    _write_json(_HISTORY_FILE, payload)


def save_history(record: Dict[str, Any], kind: str = "analysis_history") -> str:
    """新增一条历史记录并返回记录 ID。"""
    load_history_list(kind)
    record_id = uuid.uuid4().hex[:12]
    item = {"id": record_id}
    item.update(record)
    item.setdefault("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    bucket = st.session_state[_bucket_key(kind)]
    st.session_state[_bucket_key(kind)] = [item] + list(bucket)
    _persist_history()
    return record_id


def delete_history(record_id: str, kind: str = "analysis_history") -> bool:
    return delete_histories([record_id], kind) > 0


def delete_histories(record_ids: List[str], kind: str = "analysis_history") -> int:
    load_history_list(kind)
    targets = set(record_ids)
    bucket = st.session_state[_bucket_key(kind)]
    kept = [item for item in bucket if item.get("id") not in targets]
    removed = len(bucket) - len(kept)
    st.session_state[_bucket_key(kind)] = kept
    _persist_history()
    return removed


def load_history_detail(record_id: str, kind: str = "analysis_history") -> Dict[str, Any] | None:
    for item in load_history_list(kind):
        if item.get("id") == record_id:
            return item
    return None


# --- 对话存档 ---


def load_chat_state() -> Dict[str, Any]:
    """恢复对话：当前会话 + 已归档线程。"""
    if not st.session_state.get("_chat_loaded"):
        payload = _read_json(_CHAT_FILE, {})
        st.session_state.chat_messages = payload.get("current", [])
        st.session_state.chat_threads = payload.get("threads", [])
        st.session_state["_chat_loaded"] = True
    return {
        "current": st.session_state.chat_messages,
        "threads": st.session_state.chat_threads,
    }


def save_chat_state() -> None:
    """把当前会话与存档写盘。"""
    _write_json(
        _CHAT_FILE,
        {
            "current": st.session_state.get("chat_messages", []),
            "threads": st.session_state.get("chat_threads", []),
        },
    )


def archive_chat(title: str = "") -> str | None:
    """把当前会话归档为一个线程，返回线程 ID（空会话返回 None）。"""
    messages = st.session_state.get("chat_messages") or []
    if not messages:
        return None
    thread_id = uuid.uuid4().hex[:12]
    first_user = next(
        (item["content"] for item in messages if item.get("role") == "user"), "未命名对话"
    )
    thread = {
        "id": thread_id,
        "title": (title or first_user)[:28],
        "updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "messages": list(messages),
    }
    st.session_state.chat_threads = [thread] + list(st.session_state.get("chat_threads", []))
    save_chat_state()
    return thread_id


def load_thread(thread_id: str) -> bool:
    """把某个存档载入当前会话。"""
    for thread in st.session_state.get("chat_threads", []):
        if thread.get("id") == thread_id:
            st.session_state.chat_messages = list(thread.get("messages", []))
            st.session_state.chat_active_thread = thread_id
            save_chat_state()
            return True
    return False


def delete_thread(thread_id: str) -> int:
    threads = st.session_state.get("chat_threads", [])
    kept = [item for item in threads if item.get("id") != thread_id]
    removed = len(threads) - len(kept)
    st.session_state.chat_threads = kept
    if st.session_state.get("chat_active_thread") == thread_id:
        st.session_state.chat_active_thread = None
    save_chat_state()
    return removed


def clear_chat() -> None:
    st.session_state.chat_messages = []
    st.session_state.chat_active_thread = None
    save_chat_state()
