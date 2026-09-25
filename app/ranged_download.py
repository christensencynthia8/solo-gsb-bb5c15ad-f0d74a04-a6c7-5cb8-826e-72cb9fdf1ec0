"""成品文件断点下载（HTTP Range）支持。

会话下载（GET /sessions/{id}/download）与凭票下载
（GET /tickets/{ticket}/download）共用的成品字节流式输出模块：

  * 无 Range 头：流式输出完整成品（200）。
  * 单个 bytes 范围——闭区间 ``bytes=a-b``、开放结尾 ``bytes=a-``、
    后缀 ``bytes=-n`` 三种形式：按成品实际长度规范化后只流式输出对应
    字节，返回 206、Content-Range、Content-Length 与 Accept-Ranges。
  * 多范围（逗号分隔）或越界/不可满足范围：返回 416 与
    ``Content-Range: bytes */总长度``；判定只凭 stat 得到的长度，
    全程不读取文件内容。
  * 无法识别的 Range 头（非 bytes 单位、语法不成立）：按 RFC 7233
    “不理解的 range unit 必须忽略”处理，退回完整文件 200。

凭票下载在原子扣减次数之前调用 :func:`parse_range_header` 校验范围：
范围无效抛 416，扣减事务随之回滚，绝不扣次。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from urllib.parse import quote

from fastapi import HTTPException
from fastapi.responses import Response, StreamingResponse

# 与拼装/巡检一致的 1 MiB 流式块：再大的成品也不会被整体读进内存。
_BLOCK = 1024 * 1024


@dataclass(frozen=True)
class ByteRange:
    """按成品实际长度规范化后的闭区间 [start, end]（两端含）。"""

    start: int
    end: int
    size: int  # 成品实际长度

    @property
    def length(self) -> int:
        return self.end - self.start + 1


def _not_satisfiable(size: int) -> HTTPException:
    """416：携带 ``Content-Range: bytes */总长度``，不读取文件内容。"""
    return HTTPException(
        status_code=416,
        detail={"error": "range not satisfiable"},
        headers={"Content-Range": f"bytes */{size}"},
    )


def parse_range_header(value: str | None, size: int) -> ByteRange | None:
    """解析单个 Range 头并按成品实际长度 ``size`` 规范化。

    返回 ``None`` 表示按完整文件处理：无 Range 头，或头无法识别
    （非 bytes 单位、语法不成立——按 RFC 7233 忽略）。
    多范围与越界/不可满足范围抛 416（``Content-Range: bytes */size``）；
    ``size`` 由调用方 stat 得到，本函数不接触文件内容。
    """
    if value is None:
        return None
    unit, sep, spec = value.strip().partition("=")
    if not sep or unit.strip().lower() != "bytes":
        return None  # 不理解的 range unit：忽略
    spec = spec.strip()
    if "," in spec:
        # 本服务只支持单范围：多范围明确拒绝（416），不退回完整文件，
        # 免得客户端把整文件响应误当成所请求的某一段。
        raise _not_satisfiable(size)
    first, dash, last = spec.partition("-")
    first, last = first.strip(), last.strip()
    if (
        not dash
        or (not first and not last)  # "bytes=-" 两端皆空
        or "-" in last  # "bytes=1-2-3" 之类多段
        or (first and not first.isdigit())
        or (last and not last.isdigit())
    ):
        return None  # 语法不成立：忽略，按完整文件处理

    if size <= 0:
        # 空成品没有任何可交付字节：任何成立的范围都不可满足。
        raise _not_satisfiable(size)
    if not first:
        # 后缀形式 bytes=-n：最后 n 字节；n >= size 时为整个成品。
        n = int(last)
        if n <= 0:
            raise _not_satisfiable(size)
        return ByteRange(max(size - n, 0), size - 1, size)
    start = int(first)
    if start >= size:
        raise _not_satisfiable(size)
    if not last:
        end = size - 1  # 开放结尾 bytes=a-：到成品末尾
    else:
        end = int(last)
        if end < start:
            raise _not_satisfiable(size)
        end = min(end, size - 1)  # 闭区间终点越过末尾时按末尾截断
    return ByteRange(start, end, size)


def _iter_bytes(path: Path, start: int, length: int) -> Iterator[bytes]:
    """从 ``start`` 起流式产出 ``length`` 字节（1 MiB 块）。

    同步迭代器由 StreamingResponse 放到线程池消费，不阻塞事件循环。
    成品在传输中途被并发删除/截断时提前结束，绝不编造字节。
    """
    with path.open("rb") as fh:
        fh.seek(start)
        remaining = length
        while remaining > 0:
            block = fh.read(min(_BLOCK, remaining))
            if not block:
                break
            yield block
            remaining -= len(block)


def _content_disposition(filename: str | None) -> str | None:
    """与 FileResponse 同形的 attachment 处置头（非 ASCII 走 RFC 5987）。"""
    if filename is None:
        return None
    quoted = quote(filename)
    if quoted != filename:
        return f"attachment; filename*=utf-8''{quoted}"
    return f'attachment; filename="{filename}"'


def respond(
    path: Path,
    size: int,
    byte_range: ByteRange | None,
    *,
    media_type: str = "application/octet-stream",
    filename: str | None = None,
) -> Response:
    """按已规范化的范围装配 200/206 流式响应（Accept-Ranges 恒为 bytes）。

    ``byte_range`` 为 None 时输出完整成品（200）；否则只流式输出
    闭区间内的字节（206，带 Content-Range 与精确 Content-Length）。
    """
    headers = {"Accept-Ranges": "bytes"}
    disposition = _content_disposition(filename)
    if disposition is not None:
        headers["Content-Disposition"] = disposition
    if byte_range is None:
        start, length, status_code = 0, size, 200
    else:
        start, length, status_code = byte_range.start, byte_range.length, 206
        headers["Content-Range"] = f"bytes {byte_range.start}-{byte_range.end}/{size}"
    headers["Content-Length"] = str(length)
    return StreamingResponse(
        _iter_bytes(path, start, length),
        status_code=status_code,
        headers=headers,
        media_type=media_type,
    )


def respond_to_header(
    path: Path,
    range_header: str | None,
    *,
    media_type: str = "application/octet-stream",
    filename: str | None = None,
) -> Response:
    """一站式：stat 成品实际长度 → 解析 Range → 装配 200/206（或抛 416）。

    供会话下载使用；凭票下载需要在扣减事务内先校验范围，应改用
    :func:`parse_range_header` + :func:`respond` 分步调用。
    """
    size = path.stat().st_size
    return respond(
        path,
        size,
        parse_range_header(range_header, size),
        media_type=media_type,
        filename=filename,
    )
