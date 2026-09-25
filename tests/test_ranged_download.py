"""成品断点下载（HTTP Range）测试：会话下载与凭票下载的单范围流式输出。

覆盖三种单范围形式（闭区间/开放结尾/后缀）的 206 与规范化、无 Range 的
完整下载、多范围与越界的 416（Content-Range: bytes */总长度，不读取文件
内容），以及凭票下载“仅对有效范围原子扣减一次”的计数语义。
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from fastapi.testclient import TestClient

from app import db, storage


def _completed_session(client: TestClient, data: bytes, name: str = "data.bin") -> str:
    """上传一份完整文件并完成，返回 session_id。"""
    resp = client.post(
        "/api/v1/sessions",
        json={
            "filename": name,
            "total_size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        },
    )
    assert resp.status_code == 201, resp.text
    sid = resp.json()["session_id"]
    if data:
        r = client.put(
            f"/api/v1/sessions/{sid}/chunks?start=0&end={len(data)}", content=data
        )
        assert r.status_code == 200, r.text
    done = client.post(f"/api/v1/sessions/{sid}/complete")
    assert done.status_code == 200, done.text
    return sid


def _issue(client: TestClient, sid: str, *, minutes: int = 60, max_downloads: int = 1):
    r = client.post(
        f"/api/v1/sessions/{sid}/tickets",
        json={"expires_in_minutes": minutes, "max_downloads": max_downloads},
    )
    assert r.status_code == 201, r.text
    return r.json()


def _remaining(ticket_id: str) -> int:
    row = db.get_conn().execute(
        "SELECT remaining_downloads FROM download_tickets WHERE id = ?",
        (ticket_id,),
    ).fetchone()
    return row["remaining_downloads"]


DATA = bytes(range(256)) * 40  # 10240 字节，内容按偏移可校验


# ---------------------------------------------------------------------------
# 会话下载：单范围 206 与规范化
# ---------------------------------------------------------------------------


def test_session_download_closed_range(client: TestClient) -> None:
    sid = _completed_session(client, DATA)
    r = client.get(
        f"/api/v1/sessions/{sid}/download", headers={"Range": "bytes=10-19"}
    )
    assert r.status_code == 206
    assert r.content == DATA[10:20]
    assert r.headers["content-range"] == f"bytes 10-19/{len(DATA)}"
    assert r.headers["content-length"] == "10"
    assert r.headers["accept-ranges"] == "bytes"
    assert "attachment" in r.headers["content-disposition"]


def test_session_download_open_ended_range(client: TestClient) -> None:
    sid = _completed_session(client, DATA)
    r = client.get(
        f"/api/v1/sessions/{sid}/download", headers={"Range": "bytes=100-"}
    )
    assert r.status_code == 206
    assert r.content == DATA[100:]
    assert r.headers["content-range"] == f"bytes 100-{len(DATA) - 1}/{len(DATA)}"
    assert r.headers["content-length"] == str(len(DATA) - 100)


def test_session_download_suffix_range(client: TestClient) -> None:
    sid = _completed_session(client, DATA)
    r = client.get(
        f"/api/v1/sessions/{sid}/download", headers={"Range": "bytes=-64"}
    )
    assert r.status_code == 206
    assert r.content == DATA[-64:]
    assert r.headers["content-range"] == (
        f"bytes {len(DATA) - 64}-{len(DATA) - 1}/{len(DATA)}"
    )
    assert r.headers["content-length"] == "64"


def test_session_download_suffix_larger_than_file_returns_whole(
    client: TestClient,
) -> None:
    sid = _completed_session(client, DATA)
    r = client.get(
        f"/api/v1/sessions/{sid}/download",
        headers={"Range": f"bytes=-{len(DATA) + 5000}"},
    )
    assert r.status_code == 206
    assert r.content == DATA
    assert r.headers["content-range"] == f"bytes 0-{len(DATA) - 1}/{len(DATA)}"
    assert r.headers["content-length"] == str(len(DATA))


def test_session_download_end_beyond_file_is_clamped(client: TestClient) -> None:
    sid = _completed_session(client, DATA)
    r = client.get(
        f"/api/v1/sessions/{sid}/download",
        headers={"Range": f"bytes=0-{len(DATA) + 5000}"},
    )
    assert r.status_code == 206
    assert r.content == DATA
    assert r.headers["content-range"] == f"bytes 0-{len(DATA) - 1}/{len(DATA)}"


def test_session_download_single_and_last_byte(client: TestClient) -> None:
    sid = _completed_session(client, DATA)
    r = client.get(
        f"/api/v1/sessions/{sid}/download", headers={"Range": "bytes=0-0"}
    )
    assert r.status_code == 206
    assert r.content == DATA[0:1]
    assert r.headers["content-range"] == f"bytes 0-0/{len(DATA)}"

    last = len(DATA) - 1
    r = client.get(
        f"/api/v1/sessions/{sid}/download", headers={"Range": f"bytes={last}-"}
    )
    assert r.status_code == 206
    assert r.content == DATA[last:]
    assert r.headers["content-range"] == f"bytes {last}-{last}/{len(DATA)}"


def test_session_download_without_range_returns_full_file(client: TestClient) -> None:
    sid = _completed_session(client, DATA)
    r = client.get(f"/api/v1/sessions/{sid}/download")
    assert r.status_code == 200
    assert r.content == DATA
    assert r.headers["content-length"] == str(len(DATA))
    assert r.headers["accept-ranges"] == "bytes"
    assert "content-range" not in r.headers


def test_session_download_unrecognized_range_header_falls_back_to_full(
    client: TestClient,
) -> None:
    sid = _completed_session(client, DATA)
    # 非 bytes 单位与语法不成立的头按 RFC 7233 忽略，返回完整文件。
    for header in ("items=0-9", "bytes=abc", "bytes=-", "bytes=", "bytes=1-2-3"):
        r = client.get(
            f"/api/v1/sessions/{sid}/download", headers={"Range": header}
        )
        assert r.status_code == 200, header
        assert r.content == DATA


# ---------------------------------------------------------------------------
# 会话下载：多范围与越界 416（Content-Range: bytes */总长度，不读取文件内容）
# ---------------------------------------------------------------------------


def test_session_download_multiple_ranges_416(client: TestClient) -> None:
    sid = _completed_session(client, DATA)
    r = client.get(
        f"/api/v1/sessions/{sid}/download", headers={"Range": "bytes=0-9,100-199"}
    )
    assert r.status_code == 416
    assert r.headers["content-range"] == f"bytes */{len(DATA)}"
    assert r.json()["detail"]["error"] == "range not satisfiable"


def test_session_download_out_of_bounds_416(client: TestClient) -> None:
    sid = _completed_session(client, DATA)
    for header in (
        f"bytes={len(DATA)}-",  # 起点等于长度
        f"bytes={len(DATA)}-{len(DATA) + 99}",  # 整段越界
        f"bytes={len(DATA) + 1}-",  # 起点越过长度
    ):
        r = client.get(
            f"/api/v1/sessions/{sid}/download", headers={"Range": header}
        )
        assert r.status_code == 416, header
        assert r.headers["content-range"] == f"bytes */{len(DATA)}"


def test_session_download_inverted_and_zero_suffix_416(client: TestClient) -> None:
    sid = _completed_session(client, DATA)
    for header in ("bytes=20-10", "bytes=-0"):
        r = client.get(
            f"/api/v1/sessions/{sid}/download", headers={"Range": header}
        )
        assert r.status_code == 416, header
        assert r.headers["content-range"] == f"bytes */{len(DATA)}"


def test_416_does_not_read_file_content(client: TestClient) -> None:
    """成品被设为不可读时越界请求仍返回 416：判定只凭长度，不打开文件。"""
    sid = _completed_session(client, DATA)
    artifact = storage.artifact_path(sid)
    os.chmod(artifact, 0)
    try:
        r = client.get(
            f"/api/v1/sessions/{sid}/download",
            headers={"Range": f"bytes={len(DATA)}-"},
        )
        assert r.status_code == 416
        assert r.headers["content-range"] == f"bytes */{len(DATA)}"
        # 多范围同样只凭长度判定。
        r = client.get(
            f"/api/v1/sessions/{sid}/download", headers={"Range": "bytes=0-1,2-3"}
        )
        assert r.status_code == 416
    finally:
        os.chmod(artifact, 0o644)


def test_empty_artifact_full_download_and_range_416(client: TestClient) -> None:
    sid = _completed_session(client, b"", name="empty.txt")
    r = client.get(f"/api/v1/sessions/{sid}/download")
    assert r.status_code == 200
    assert r.content == b""
    assert r.headers["content-length"] == "0"
    assert r.headers["accept-ranges"] == "bytes"
    # 空成品没有任何可交付字节：任何成立的范围都 416，总长为 0。
    for header in ("bytes=0-0", "bytes=0-", "bytes=-1"):
        r = client.get(
            f"/api/v1/sessions/{sid}/download", headers={"Range": header}
        )
        assert r.status_code == 416, header
        assert r.headers["content-range"] == "bytes */0"


def test_session_download_range_repeated_requests(client: TestClient) -> None:
    """会话下载不计数：同一范围可反复拉取，配合 Range 实现断点续传。"""
    sid = _completed_session(client, DATA)
    received = b""
    step = 4096
    for start in range(0, len(DATA), step):
        end = min(start + step, len(DATA)) - 1
        r = client.get(
            f"/api/v1/sessions/{sid}/download",
            headers={"Range": f"bytes={start}-{end}"},
        )
        assert r.status_code == 206
        received += r.content
    assert received == DATA


# ---------------------------------------------------------------------------
# 凭票下载：仅对有效范围原子扣减一次
# ---------------------------------------------------------------------------


def test_ticket_valid_range_decrements_once(client: TestClient) -> None:
    sid = _completed_session(client, DATA)
    body = _issue(client, sid, max_downloads=2)
    url = f"/api/v1/tickets/{body['ticket']}/download"

    r = client.get(url, headers={"Range": "bytes=0-99"})
    assert r.status_code == 206
    assert r.content == DATA[:100]
    assert r.headers["content-range"] == f"bytes 0-99/{len(DATA)}"
    assert r.headers["accept-ranges"] == "bytes"
    assert _remaining(body["ticket_id"]) == 1

    # 后缀与开放结尾同样各计一次。
    body2 = _issue(client, sid, max_downloads=2)
    url2 = f"/api/v1/tickets/{body2['ticket']}/download"
    r = client.get(url2, headers={"Range": "bytes=-10"})
    assert r.status_code == 206
    assert r.content == DATA[-10:]
    r = client.get(url2, headers={"Range": f"bytes={len(DATA) - 5}-"})
    assert r.status_code == 206
    assert r.content == DATA[-5:]
    assert _remaining(body2["ticket_id"]) == 0
    # 次数用尽后 Range 请求同样 429。
    assert client.get(url2, headers={"Range": "bytes=0-0"}).status_code == 429


def test_ticket_invalid_range_never_decrements(client: TestClient) -> None:
    sid = _completed_session(client, DATA)
    body = _issue(client, sid, max_downloads=2)
    url = f"/api/v1/tickets/{body['ticket']}/download"

    # 多范围与越界：416 且不扣次，可反复尝试。
    for header in (
        "bytes=0-9,100-199",
        f"bytes={len(DATA)}-",
        "bytes=20-10",
        "bytes=-0",
    ):
        r = client.get(url, headers={"Range": header})
        assert r.status_code == 416, header
        assert r.headers["content-range"] == f"bytes */{len(DATA)}"
        assert _remaining(body["ticket_id"]) == 2

    # 无效范围不消耗次数：随后两次有效下载仍可用完配额。
    r = client.get(url, headers={"Range": "bytes=0-9"})
    assert r.status_code == 206
    r = client.get(url)
    assert r.status_code == 200
    assert r.content == DATA
    assert _remaining(body["ticket_id"]) == 0
    assert client.get(url).status_code == 429


def test_ticket_quarantined_range_rejected_without_decrement(
    client: TestClient,
) -> None:
    sid = _completed_session(client, DATA)
    body = _issue(client, sid, max_downloads=2)
    url = f"/api/v1/tickets/{body['ticket']}/download"

    # 破坏成品并巡检，使其被隔离。
    artifact = Path(storage.artifact_path(sid))
    raw = bytearray(artifact.read_bytes())
    raw[0] ^= 0xFF
    artifact.write_bytes(bytes(raw))
    check = client.post(f"/api/v1/sessions/{sid}/integrity-check")
    assert check.json()["quarantined"] is True

    # 隔离后带 Range 的凭票下载仍 423，且不扣次。
    r = client.get(url, headers={"Range": "bytes=0-9"})
    assert r.status_code == 423
    assert r.json()["detail"]["error"] == "session quarantined"
    assert _remaining(body["ticket_id"]) == 2


def test_ticket_download_without_range_still_full_and_counted(
    client: TestClient,
) -> None:
    sid = _completed_session(client, DATA)
    body = _issue(client, sid, max_downloads=1)
    r = client.get(f"/api/v1/tickets/{body['ticket']}/download")
    assert r.status_code == 200
    assert r.content == DATA
    assert r.headers["accept-ranges"] == "bytes"
    assert _remaining(body["ticket_id"]) == 0
