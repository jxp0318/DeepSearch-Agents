import {
  ApiOutlined,
  ArrowDownOutlined,
  BranchesOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  CloudServerOutlined,
  DatabaseOutlined,
  FileSearchOutlined,
  ToolOutlined
} from "@ant-design/icons";
import { Alert, App as AntApp, Button } from "antd";
import { useEffect, useRef, useState } from "react";
import { ChatComposer } from "./components/ChatComposer";
import { ConversationThread } from "./components/ConversationThread";
import type { ChatTurn } from "./components/ConversationThread";
import { API_BASE_URL, WS_BASE_URL } from "./lib/config";
import { useDeepAgentSession } from "./hooks/useDeepAgentSession";
import type { ConnectionState, UploadedItem } from "./types";

function connectionLabel(state: ConnectionState): string {
  const labels: Record<ConnectionState, string> = {
    connecting: "连接中",
    connected: "已连接",
    reconnecting: "重连中",
    closed: "已关闭"
  };
  return labels[state];
}

function createTurn(content: string): ChatTurn {
  return {
    id: crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}`,
    content,
    events: [],
    files: [],
    isRunning: true,
    result: "",
    timestamp: new Date().toISOString()
  };
}

// 判定“用户是否停留在底部”的容差：滚动位置距底部小于该值视为跟随中
const STICK_TO_BOTTOM_THRESHOLD = 80;

export default function App() {
  const { message } = AntApp.useApp();
  const [query, setQuery] = useState("");
  const [stagedItems, setStagedItems] = useState<UploadedItem[]>([]);
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [isFollowingLatest, setIsFollowingLatest] = useState(true);
  const streamRef = useRef<HTMLElement | null>(null);
  // 用 ref 镜像跟随状态，供滚动副作用同步读取，避免闭包读到旧值
  const followingRef = useRef(true);
  const session = useDeepAgentSession();

  const setFollowing = (next: boolean) => {
    if (followingRef.current === next) {
      return;
    }
    followingRef.current = next;
    setIsFollowingLatest(next);
  };

  const scrollToLatest = (behavior: ScrollBehavior = "smooth") => {
    const streamNode = streamRef.current;
    if (!streamNode) {
      return;
    }
    streamNode.scrollTo({ top: streamNode.scrollHeight, behavior });
  };

  useEffect(() => {
    setTurns((previous) => {
      if (previous.length === 0) {
        // 页面刷新、跨标签页或重连回放时会先收到事件、但本地还没有对应轮次，
        // 这里按事件补建一条，避免「后端在跑、页面却一片空白」
        if (session.events.length === 0) {
          return previous;
        }
        return [
          {
            ...createTurn(session.currentQuery || "正在执行的任务"),
            isRunning: session.isRunning,
            result: session.result
          }
        ];
      }

      const latestTurn = previous[previous.length - 1];
      const nextLatestTurn = {
        ...latestTurn,
        events: session.events,
        files: session.files,
        isRunning: session.isRunning,
        result: session.result
      };

      return [...previous.slice(0, -1), nextLatestTurn];
    });
  }, [
    session.currentQuery,
    session.events,
    session.files,
    session.isRunning,
    session.result
  ]);

  // 监听流式区域滚动：用户主动往上翻超过阈值就停止自动跟随，回到底部附近再恢复
  useEffect(() => {
    const streamNode = streamRef.current;
    if (!streamNode) {
      return;
    }

    const handleScroll = () => {
      const distance =
        streamNode.scrollHeight - streamNode.scrollTop - streamNode.clientHeight;
      setFollowing(distance <= STICK_TO_BOTTOM_THRESHOLD);
    };

    streamNode.addEventListener("scroll", handleScroll, { passive: true });
    return () => streamNode.removeEventListener("scroll", handleScroll);
  }, []);

  // 仅在“跟随中”才滚到底部：用户正在回看历史时不再打断
  // behavior 用 auto：运行期间事件密集，smooth 会排成一串动画，产生被拽着下滑的粘滞感
  useEffect(() => {
    if (!followingRef.current) {
      return;
    }

    const frame = window.requestAnimationFrame(() => {
      scrollToLatest("auto");
    });
    return () => window.cancelAnimationFrame(frame);
  }, [turns]);

  async function handleSubmit() {
    const cleanQuery = query.trim();
    if (!cleanQuery) {
      message.warning("请输入研搜任务");
      return;
    }

    const nextTurn = createTurn(cleanQuery);
    setTurns((previous) => [...previous, nextTurn]);
    setQuery("");
    // 提交新任务时恢复自动跟随：用户此刻期望看到自己刚发出的问题
    setFollowing(true);
    window.requestAnimationFrame(() => scrollToLatest("smooth"));

    try {
      await session.submitTask(cleanQuery);
      message.success("任务已启动，执行过程会显示在对话中");
    } catch (error) {
      setTurns((previous) =>
        previous.map((turn) =>
          turn.id === nextTurn.id
            ? {
                ...turn,
                isRunning: false,
                result: error instanceof Error ? error.message : "任务启动失败"
              }
            : turn
        )
      );
      message.error(error instanceof Error ? error.message : "任务启动失败");
    }
  }

  async function handleCancel() {
    try {
      const response = await session.cancelCurrentTask();
      message.info(response.status === "cancelling" ? "取消请求已发送，正在等待当前调用结束" : "任务已取消");
    } catch (error) {
      message.error(error instanceof Error ? error.message : "取消任务失败");
    }
  }

  async function handleUpload(items: UploadedItem[]) {
    try {
      const response = await session.uploadFiles(items);
      setStagedItems([]);
      message.success(`已上传 ${response.files.length} 个文件`);
    } catch (error) {
      message.error(error instanceof Error ? error.message : "上传失败");
    }
  }

  function handleNewSession() {
    session.resetSession();
    setTurns([]);
    setQuery("");
    setStagedItems([]);
    setFollowing(true);
  }

  const online = session.connectionState === "connected";

  return (
    <div className="chat-app-shell min-h-dvh">
      <aside className="chat-sidebar" aria-label="会话信息">
        <div className="sidebar-brand">
          <span className="panel-kicker">DEEPSEARCH</span>
          <h1>深度研搜</h1>
          <p>对话式多智能体研究台</p>
        </div>

        <Button className="new-chat-button" block onClick={handleNewSession}>
          新建研搜
        </Button>

        <div className="sidebar-section">
          <span className="sidebar-label">THREAD</span>
          <strong className="thread-id" title={session.threadId}>
            {session.threadId.slice(0, 8)}
          </strong>
        </div>

        <div className="sidebar-status-list">
          <div className={`sidebar-status ${online ? "sidebar-status--online" : "sidebar-status--warn"}`}>
            <ApiOutlined aria-hidden />
            <span>WebSocket</span>
            <strong>{connectionLabel(session.connectionState)}</strong>
          </div>
          <div className="sidebar-status">
            <BranchesOutlined aria-hidden />
            <span>助手调度</span>
            <strong>{session.stats.assistantEvents}</strong>
          </div>
          <div className="sidebar-status">
            <ToolOutlined aria-hidden />
            <span>工具调用</span>
            <strong>{session.stats.toolEvents}</strong>
          </div>
          <div className={session.stats.errorEvents > 0 ? "sidebar-status sidebar-status--error" : "sidebar-status"}>
            <CloseCircleOutlined aria-hidden />
            <span>异常</span>
            <strong>{session.stats.errorEvents}</strong>
          </div>
        </div>

        <div className="sidebar-section">
          <span className="sidebar-label">AGENTS</span>
          <ul className="agent-mini-list">
            <li>
              <CloudServerOutlined aria-hidden />
              网络搜索助手
            </li>
            <li>
              <DatabaseOutlined aria-hidden />
              数据库查询助手
            </li>
            <li>
              <FileSearchOutlined aria-hidden />
              RAGFlow 助手
            </li>
          </ul>
        </div>

        <div className="sidebar-section sidebar-endpoints">
          <span className="sidebar-label">ENDPOINTS</span>
          <code>{API_BASE_URL}</code>
          <code>{WS_BASE_URL}</code>
        </div>
      </aside>

      <main className="chat-main">
        <header className="chat-topbar">
          <div>
            <span className="panel-kicker">CHAT WORKSPACE</span>
            <h2>深度研搜对话</h2>
          </div>
          <div className={`run-indicator ${session.isRunning ? "run-indicator--live" : ""}`}>
            {session.isRunning ? <BranchesOutlined aria-hidden /> : <CheckCircleOutlined aria-hidden />}
            {session.isRunning ? "研搜中" : "待命"}
          </div>
        </header>

        {session.lastError ? (
          <Alert
            className="chat-alert"
            message={session.lastError}
            showIcon
            type="error"
          />
        ) : null}

        <section className="chat-stream-panel" ref={streamRef}>
          <ConversationThread
            onUseExample={setQuery}
            turns={turns}
          />
        </section>

        {!isFollowingLatest && turns.length > 0 ? (
          <button
            className="scroll-to-latest"
            onClick={() => {
              setFollowing(true);
              scrollToLatest("smooth");
            }}
            type="button"
          >
            <ArrowDownOutlined aria-hidden />
            回到最新
          </button>
        ) : null}

        <ChatComposer
          isCancelling={session.isCancelling}
          isRunning={session.isRunning}
          isUploading={session.isUploading}
          onCancel={handleCancel}
          onNewSession={handleNewSession}
          onQueryChange={setQuery}
          onStagedItemsChange={setStagedItems}
          onSubmit={handleSubmit}
          onUpload={handleUpload}
          query={query}
          stagedItems={stagedItems}
          uploadedItems={session.uploadedItems}
        />
      </main>
    </div>
  );
}
