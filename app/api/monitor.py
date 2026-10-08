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
import threading
from typing import Any, Optional

from fastapi import WebSocket

from app.api.context import get_thread_context

# 单个 thread 保留的最近事件条数：覆盖一次深度研搜的主要执行轨迹
MAX_BUFFERED_EVENTS = 200
# 最多保留多少个 thread 的事件缓冲，防止长时间运行的服务内存无上限增长
MAX_BUFFERED_THREADS = 50


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
    ) -> None:
        """报告开始执行某个工具"""
        self._emit(
            "tool_start",
            f"开始执行工具: {tool_name}",
            {"tool_name": tool_name, "args": args},
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

    async def send_personal_message(self, message: str, websocket: WebSocket) -> None:
        """向指定 WebSocket 发送纯文本消息"""
        await websocket.send_text(message)

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
