"""
Agent 执行过程监控模块

负责把工具调用、子智能体调用、任务结果和会话目录等事件统一包装后推送给前端
在 Web 服务中优先通过 WebSocket 定向推送；在脚本调试场景中保留控制台输出

事件同时会写入按 thread_id 组织的环形缓冲，供页面刷新或断线重连后回放，
避免「后端跑完了但页面一直停在执行中」这类状态不同步问题。
"""

import asyncio
import builtins
import datetime
import json
import os
import threading
from pathlib import Path
from typing import Any, Optional

from fastapi import WebSocket

from app.api.context import get_thread_context

# 单个 thread 保留的最近事件条数：覆盖一次深度研搜的主要执行轨迹
MAX_BUFFERED_EVENTS = 200
# 最多保留多少个 thread 的事件缓冲，防止长时间运行的服务内存无上限增长
MAX_BUFFERED_THREADS = 50

# =============================================================================
# 评测采集开关（生成层评测，见 eval.md 第 10 节）
#
# 默认关闭：线上运行时本模块只做「构造事件 + 推送」，不产生任何磁盘写入。
# 打开（EVAL_CAPTURE=1）后，工具可以把**返回值**登记进来，评测 runner 据此判定：
#   · 引用准确率 —— 报告里的【引用B3-2】是否真实存在、指向哪份文档哪一节
#   · Faithfulness —— 报告论断是否能被引用的 chunk 原文支撑
#
# 为什么落盘而不是只放内存：一次跑批要跨进程核对（runner 与服务分进程），
# 落盘到 results/<run_id>/ 目录最简单；关闭时连目录都不创建。
# =============================================================================
_EVAL_CAPTURE = os.getenv("EVAL_CAPTURE", "0") == "1"
_EVAL_TRACE_DIR = os.getenv("EVAL_TRACE_DIR", "")

# =============================================================================
# 单任务工具调用预算（执行层硬刹车）
#
# 实测教训（2026-10-15 全量跑批）：题库的 `max_tool_calls` 只是评测侧**事后记账**的
# 阈值，执行链根本不知道它存在。一次 kb-02 发了 37 次知识库检索 + 12 次网络检索
# （题设上限 12），一道 db-05 卡到 900s 超时——模型「换词再搜」没有上界，反思轮数与
# token 预算都拦不住这种单轮内的打转。
#
# 因此这里做真正的熔断：超过预算后**检索类工具不再执行**，而是把「请基于已有材料收尾」
# 的指令返回给模型。上限取 30——高于题库最宽的 24（kb-13 / mix-06），不会误伤正常难题；
# 置 0 或负数表示关闭该闸门。
# =============================================================================
TOOL_CALL_BUDGET = int(os.getenv("TOOL_CALL_BUDGET", "30"))


def eval_capture_enabled() -> bool:
    """评测采集是否开启（供工具层判断是否登记返回值）"""
    return _EVAL_CAPTURE


def check_call_budget(tool_name: str) -> Optional[str]:
    """
    检索类工具的预算闸门。

    超预算时返回一段「立刻收尾」的指令文本——调用方把它原样 return 给模型，
    **不再执行本次检索**；未超预算、或不在任务链路里（脚本直跑、无 thread_id）、
    或闸门已关闭（`TOOL_CALL_BUDGET<=0`）时返回 None。

    计的是**检索类调用次数**（知识库检索 / 网络搜索 / SQL 查询），不是所有工具调用：
    预算的语义是「别再搜了，用已有材料收尾」，把生成报告也计入会导致出不了货。

    :param tool_name: 调用方工具名，仅用于日志
    """
    if TOOL_CALL_BUDGET <= 0:
        return None
    thread_id = get_thread_context()
    if not thread_id:
        return None
    used = monitor.retrieval_calls_used(thread_id)
    if used <= TOOL_CALL_BUDGET:
        return None
    print(f"[Monitor] 预算熔断：{tool_name} 被拦（检索类已调用 {used} 次，上限 {TOOL_CALL_BUDGET}）")
    return (
        f"检索预算已用尽（本次是第 {used} 次取信息调用，上限 {TOOL_CALL_BUDGET} 次），"
        "本次检索未执行。请立即基于已经获取到的材料输出结论："
        "不要再生成新的检索或换词重试；材料没有覆盖到的部分明确写「未获取到」，不要编造。"
    )


def register_tool_result(thread_id: Optional[str], tool_name: str, payload: dict) -> None:
    """
    登记工具返回值（仅 EVAL_CAPTURE=1 时真正写盘）。

    :param thread_id: 会话 ID；为空说明不在任务链路里，直接忽略
    :param tool_name: 工具名
    :param payload: 要保存的结构化内容（须可 JSON 序列化）
    """
    if not _EVAL_CAPTURE or not thread_id or not _EVAL_TRACE_DIR:
        return
    try:
        target = Path(_EVAL_TRACE_DIR) / thread_id
        target.mkdir(parents=True, exist_ok=True)
        # 文件名带序号：同一 thread 内多次调用同一工具不会互相覆盖
        seq = len(list(target.glob(f"{_safe_name(tool_name)}.*.json"))) + 1
        path = target / f"{_safe_name(tool_name)}.{seq:03d}.json"
        path.write_text(
            json.dumps(payload, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
    except Exception as e:
        # 采集失败绝不能影响主链路
        print(f"[Monitor] EVAL_CAPTURE 登记失败（{tool_name}）：{e}")


def _safe_name(name: str) -> str:
    """把工具名里的中文与符号换成文件名安全字符"""
    cleaned = "".join("_" if ch in '\\/:*?"<>|' or ord(ch) > 127 else ch for ch in name)
    return cleaned.strip("_")[:60] or "tool"


class ToolMonitor:
    """
    工具和助手调用的统一监控入口

    业务工具只需要导入全局 monitor，并调用 report_tool/report_assistant 等方法
    具体是通过 WebSocket 推送，还是输出到脚本运行时，由本类内部统一处理
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ToolMonitor, cls).__new__(cls)
            cls._instance.websocket_manager = None
            # thread_id -> 事件列表（有序），用于新连接回放
            cls._instance._event_buffers: dict[str, list[dict[str, Any]]] = {}
            # thread_id -> 自增序号，前端据此去重回放事件
            cls._instance._event_seq: dict[str, int] = {}
            # 事件可能从线程池工具里发出，缓冲读写加锁保证安全
            cls._instance._buffer_lock = threading.Lock()
            # thread_id -> 本任务已发生的工具调用次数（前端展示序号）
            cls._instance._tool_calls: dict[str, int] = {}
            # thread_id -> 其中**检索类**调用的次数（预算熔断只按这个判）
            cls._instance._retrieval_calls: dict[str, int] = {}
        return cls._instance

    def set_websocket_manager(self, manager: "ConnectionManager") -> None:
        """绑定 FastAPI WebSocket 连接管理器"""
        self.websocket_manager = manager

    def begin_task(self) -> None:
        """
        标记当前 thread 开始新任务

        清空上一轮任务的事件缓冲，避免新页面连上来时回放到历史任务的轨迹。
        由 run_deep_agent 在写入任何事件之前调用。
        """
        thread_id = get_thread_context()
        if not thread_id:
            return
        with self._buffer_lock:
            self._event_buffers.pop(thread_id, None)
            self._event_seq.pop(thread_id, None)
            # 同一会话连续追问时预算重新计：每题一个新任务，不该被上一题用光
            self._tool_calls.pop(thread_id, None)
            self._retrieval_calls.pop(thread_id, None)

    def snapshot_events(self, thread_id: str) -> list[dict[str, Any]]:
        """
        获取某个 thread 已产生的事件快照

        WebSocket 建连（含页面刷新、断线重连）时回放，让前端补齐错过的事件。
        """
        with self._buffer_lock:
            return list(self._event_buffers.get(thread_id, []))

    def _next_seq(self, thread_id: str) -> int:
        """分配当前 thread 的下一个事件序号"""
        with self._buffer_lock:
            seq = self._event_seq.get(thread_id, 0) + 1
            self._event_seq[thread_id] = seq
            return seq

    def _remember(self, thread_id: str, payload: dict[str, Any]) -> None:
        """把事件写入缓冲区，并淘汰过老的 thread 记录"""
        with self._buffer_lock:
            buffer = self._event_buffers.setdefault(thread_id, [])
            buffer.append(payload)
            if len(buffer) > MAX_BUFFERED_EVENTS:
                del buffer[: len(buffer) - MAX_BUFFERED_EVENTS]

            while len(self._event_buffers) > MAX_BUFFERED_THREADS:
                oldest = next(iter(self._event_buffers))
                if oldest == thread_id:
                    break
                self._event_buffers.pop(oldest, None)
                self._event_seq.pop(oldest, None)
                self._tool_calls.pop(oldest, None)
                self._retrieval_calls.pop(oldest, None)

    def _count_tool_call(self, thread_id: Optional[str], retrieval: bool) -> tuple[int, int]:
        """
        累加本任务的调用计数，返回（总次数, 检索类次数）。

        预算只看检索类：熔断的语义是「别再搜了」，不是「什么都别做」。生成 Markdown、
        读附件这类调用如果也占配额，熔断会把收尾动作一起掐掉，反而产出空报告。
        """
        if not thread_id:
            return 0, 0
        with self._buffer_lock:
            self._tool_calls[thread_id] = self._tool_calls.get(thread_id, 0) + 1
            total = self._tool_calls[thread_id]
            if retrieval:
                self._retrieval_calls[thread_id] = self._retrieval_calls.get(thread_id, 0) + 1
            return total, self._retrieval_calls.get(thread_id, 0)

    def tool_calls_used(self, thread_id: Optional[str] = None) -> int:
        """本任务已发生的工具调用总次数；未传 thread_id 时取当前上下文"""
        thread_id = thread_id or get_thread_context()
        if not thread_id:
            return 0
        with self._buffer_lock:
            return self._tool_calls.get(thread_id, 0)

    def retrieval_calls_used(self, thread_id: Optional[str] = None) -> int:
        """本任务已发生的**检索类**调用次数（预算熔断的判断依据）"""
        thread_id = thread_id or get_thread_context()
        if not thread_id:
            return 0
        with self._buffer_lock:
            return self._retrieval_calls.get(thread_id, 0)

    def _emit(
        self,
        event_type: str,
        message: str,
        data: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        构造统一监控事件，写入缓冲并推送给当前 thread_id 的所有前端连接

        :param event_type: 事件类型，例如 tool_start、assistant_call
        :param message: 面向前端展示的事件说明
        :param data: 附加结构化数据
        """
        payload = {
            "type": "monitor_event",
            "event": event_type,
            "message": message,
            "data": data or {},
            "timestamp": datetime.datetime.now().isoformat(),
        }

        try:
            thread_id = get_thread_context()
        except Exception:
            thread_id = None

        # 即使当前没有前端连接也要先入缓冲：任务跑完页面才刷新/重连时仍能看到轨迹
        if thread_id:
            payload["seq"] = self._next_seq(thread_id)
            self._remember(thread_id, payload)

            if self.websocket_manager:
                try:
                    manager_loop = self.websocket_manager.loop
                    if manager_loop:
                        self._send_to_websocket(payload, thread_id, manager_loop)
                except Exception as e:
                    print(f"[Monitor] WebSocket send failed: {e}")

        # DeepAgents 脚本调试时，如果运行时暴露了 stream_writer，也同步写入流式输出
        if hasattr(builtins, "runtime") and hasattr(builtins.runtime, "stream_writer"):
            try:
                builtins.runtime.stream_writer(payload)
            except Exception:
                pass

        # 控制台保底输出，便于无前端场景下观察执行过程
        print(f"\n[Monitor:{event_type}] {message}")

    def _send_to_websocket(
        self,
        payload: dict[str, Any],
        thread_id: str,
        manager_loop: asyncio.AbstractEventLoop,
    ) -> None:
        """
        将监控事件投递到 WebSocket 所在事件循环

        FastAPI 的 WebSocket 必须在创建它的事件循环中发送消息
        如果当前代码已经在同一个循环里，直接 create_task；否则使用线程安全投递
        """
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        coroutine = self.websocket_manager.send_to_thread(payload, thread_id)
        if current_loop and current_loop == manager_loop:
            current_loop.create_task(coroutine)
        else:
            asyncio.run_coroutine_threadsafe(coroutine, manager_loop)

    def report_tool(
        self,
        tool_name: str,
        args: Optional[dict[str, Any]] = None,
        retrieval: bool = False,
    ) -> None:
        """
        报告开始执行某个工具，并累加本任务的调用计数（预算熔断据此判断）

        :param retrieval: 本次是否属于**取信息**的调用（知识库检索 / 网络搜索 / SQL 查询）。
                          只有这类调用计入检索预算；生成报告、读附件不占配额，
                          否则熔断会把「把已有材料写成报告」也一起掐掉，反而产出空结果。
        """
        call_index, retrieval_used = self._count_tool_call(get_thread_context(), retrieval)
        self._emit(
            "tool_start",
            f"开始执行工具: {tool_name}",
            {
                "tool_name": tool_name,
                "args": args,
                "call_index": call_index,
                "retrieval_used": retrieval_used if retrieval else None,
            },
        )

    def report_tool_error(
        self,
        tool_name: str,
        reason: str,
        detail: str = "",
        context: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        报告工具执行失败

        与 error 事件的区别：error 表示整轮任务中断，tool_error 只表示某次工具调用
        失败但任务仍在继续。分开上报，用户才能区分「模型没去搜」和「搜了但网络断了」。
        """
        self._emit(
            "tool_error",
            f"工具执行失败: {tool_name}（{reason}）",
            {
                "tool_name": tool_name,
                "reason": reason,
                "detail": detail,
                **(context or {}),
            },
        )

    def report_assistant(
        self,
        assistant_name: str,
        args: Optional[dict[str, Any]] = None,
    ) -> None:
        """报告正在调用某个子智能体"""
        self._emit(
            "assistant_call",
            f"正在调用助手: {assistant_name}",
            {"assistant_name": assistant_name, "args": args},
        )

    def report_task_result(self, result: str, partial: bool = False) -> None:
        """
        报告任务最终结果

        partial=True 表示任务中途异常终止、这是已经产出的部分内容，
        前端据此提示「结果不完整」，避免用户把半成品当成完整交付。
        """
        self._emit(
            "task_result",
            "任务执行完成（部分结果）" if partial else "任务执行完成",
            {"result": result, "partial": partial},
        )

    def report_task_cancelled(self) -> None:
        """报告任务已被用户取消"""
        self._emit("task_cancelled", "任务已取消")

    def report_error(
        self,
        reason: str,
        detail: str = "",
        has_partial_result: bool = False,
    ) -> None:
        """
        报告任务级异常（整轮执行中断）

        与 tool_error 区分：这是「本次任务失败」，前端会复位运行态并提示错误。
        has_partial_result 为真时前端会提示「已保留部分结果」。
        """
        message = f"任务执行中断：{reason}"
        if detail:
            message = f"{message}（{detail}）"
        self._emit(
            "error",
            message,
            {
                "reason": reason,
                "detail": detail,
                "has_partial_result": has_partial_result,
            },
        )

    def report_round_result(
        self,
        round_num: int,
        result: str,
        stop_reason: str = "",
    ) -> None:
        """
        报告反思循环中某一轮的执行结果

        中间轮使用 round_result 而非 task_result，避免前端把中间结果误判为任务结束。
        """
        self._emit(
            "round_result",
            f"第 {round_num} 轮执行完成" + (f"（{stop_reason}）" if stop_reason else ""),
            {"round": round_num, "result": result, "stop_reason": stop_reason},
        )

    def report_reflection_evaluation(self, round_num: int, evaluation: Any) -> None:
        """报告反思评估结论（是否充分、缺口维度、判断依据）"""
        sufficient = getattr(evaluation, "sufficient", False)
        missing = getattr(evaluation, "missing_dimensions", []) or []
        reasoning = getattr(evaluation, "reasoning", "")

        if sufficient:
            message = f"第 {round_num} 轮反思：信息已充分，准备输出最终结果"
        else:
            message = f"第 {round_num} 轮反思：发现 {len(missing)} 个信息缺口，准备补充检索"

        self._emit(
            "reflection_evaluation",
            message,
            {
                "round": round_num,
                "sufficient": sufficient,
                "missing_dimensions": missing,
                "reasoning": reasoning,
            },
        )

    def report_reflection_supplement(
        self,
        next_round: int,
        queries: list[str],
    ) -> None:
        """报告即将进入补搜轮，并列出补充查询方向"""
        self._emit(
            "reflection_supplement",
            f"启动第 {next_round} 轮补充检索，共 {len(queries)} 个查询方向",
            {"round": next_round, "follow_up_queries": queries},
        )

    def report_reflection_stopped(self, reason: str, detail: str = "") -> None:
        """报告反思循环被终止条件提前结束（非充分性终止）"""
        self._emit(
            "reflection_stopped",
            f"反思循环提前结束：{reason}",
            {"reason": reason, "detail": detail},
        )

    def report_session_dir(self, path: str, query: str = "") -> None:
        """
        报告当前任务工作目录

        query 一并发给前端，页面刷新或跨标签页看到任务时才能还原「谁问的什么问题」。
        """
        self._emit(
            "session_created",
            f"工作目录已创建: {path}",
            {"path": path, "query": query},
        )

    def report_token_usage(self, usage: dict) -> None:
        """
        上报一次模型调用的 token 用量。

        用途：生成层评测的「单任务 API 成本」指标（eval.md §10.2）需要按
        DeepSeek 峰谷价把 token 折算成钱，而这部分数据只存在于模型返回的
        usage_metadata 里，不上报就无法统计。前端不消费此事件。
        """
        self._emit("token_usage", f"token 用量: {usage}", usage)


monitor = ToolMonitor()


class ConnectionManager:
    """
    WebSocket 连接管理器

    active_connections 使用 thread_id 作为 key、连接集合作为 value：
    同一个浏览器会话可能有多个页面（多个标签页共享同一个 thread_id），
    事件必须广播给全部连接，否则只有最后建连的页面能看到执行轨迹。
    """

    def __init__(self) -> None:
        self.active_connections: dict[str, set[WebSocket]] = {}
        # WebSocket 发送必须回到创建连接的事件循环，因此启动时需要显式绑定 loop
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """绑定 FastAPI 主事件循环，并同步注册到 monitor"""
        self.loop = loop
        monitor.set_websocket_manager(self)
        print(f"[Monitor] ConnectionManager manually bound to loop: {id(self.loop)}")

    async def connect(self, websocket: WebSocket, thread_id: str) -> None:
        """
        接受 WebSocket 连接，按 thread_id 登记，并回放该 thread 已产生的事件

        回放解决两类状态不同步：页面在任务执行中刷新，以及断线重连后错过中途事件。
        """
        await websocket.accept()
        connections = self.active_connections.setdefault(thread_id, set())
        connections.add(websocket)
        print(f"Client connected: {thread_id} (active={len(connections)})")

        # 先补历史再接收新事件，前端可见完整执行轨迹
        for payload in monitor.snapshot_events(thread_id):
            await websocket.send_json(payload)

    def disconnect(self, websocket: WebSocket, thread_id: str) -> None:
        """移除已断开的 WebSocket 连接，最后一个连接断开时清理该 thread 记录"""
        connections = self.active_connections.get(thread_id)
        if not connections:
            return

        connections.discard(websocket)
        if not connections:
            del self.active_connections[thread_id]
        print(f"Client disconnected: {thread_id} (active={len(connections)})")

    async def send_to_thread(self, message: dict[str, Any], thread_id: str) -> None:
        """
        向指定 thread_id 下的所有前端连接广播 JSON 消息

        单个页面断网时发送会抛异常，这里就地剔除失效连接，避免拖垮其他页面。
        """
        connections = self.active_connections.get(thread_id)
        if not connections:
            return

        stale: list[WebSocket] = []
        for websocket in list(connections):
            try:
                await websocket.send_json(message)
            except Exception:
                stale.append(websocket)

        for websocket in stale:
            connections.discard(websocket)
        if not connections:
            self.active_connections.pop(thread_id, None)


manager = ConnectionManager()
