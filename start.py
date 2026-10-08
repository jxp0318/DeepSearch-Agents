"""Windows one-click launcher for DeepSearch Agents.

Double-click this file after completing the environment configuration in .env.
It uses the project's existing virtual environment and frontend dependencies;
it deliberately does not reinstall dependencies on every launch.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
ENV_FILE = PROJECT_ROOT / ".env"
ENV_EXAMPLE_FILE = PROJECT_ROOT / ".env.example"
BACKEND_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
VITE_ENTRY = FRONTEND_DIR / "node_modules" / "vite" / "bin" / "vite.js"
FRONTEND_URL = "http://localhost:5173"
BACKEND_PORT = "8001"


def pause(message: str = "按 Enter 关闭此启动窗口...") -> None:
    try:
        input(message)
    except EOFError:
        pass


def require_file(path: Path, hint: str) -> bool:
    if path.is_file():
        return True
    print(f"\n缺少 {path.relative_to(PROJECT_ROOT)}。{hint}")
    return False


def start_services() -> None:
    """启动 docker 内的本地服务（MySQL / embedding / Qdrant / Elasticsearch）"""
    docker = shutil.which("docker.exe") or shutil.which("docker")
    if not docker:
        print("未找到 Docker，跳过本地服务（MySQL / embedding / Qdrant / Elasticsearch）。")
        return

    info = subprocess.run(
        [docker, "info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
    )
    if info.returncode != 0:
        print("Docker Desktop 未启动，跳过本地服务。")
        return

    print("启动本地服务（MySQL / embedding / Qdrant / Elasticsearch）...")
    result = subprocess.run(
        [
            docker,
            "compose",
            "--env-file",
            str(ENV_FILE),
            "-f",
            str(PROJECT_ROOT / "docker" / "docker-compose.yaml"),
            "up",
            "-d",
        ],
        cwd=PROJECT_ROOT,
        check=False,
    )
    if result.returncode:
        print("本地服务启动失败，已继续启动前后端；数据库查询与知识库检索将按降级链运行。")


def start_service(command: list[str], working_directory: Path) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        command,
        cwd=working_directory,
        creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0),
    )


def frontend_is_ready() -> bool:
    try:
        with urllib.request.urlopen(FRONTEND_URL, timeout=1) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError):
        return False


def main() -> int:
    os.chdir(PROJECT_ROOT)
    print("=== DeepSearch Agents 一键启动 ===")

    if not ENV_FILE.exists():
        shutil.copyfile(ENV_EXAMPLE_FILE, ENV_FILE)
        print("\n已从 .env.example 创建 .env。请填写 API Key 后重新运行 start.py。")
        os.startfile(ENV_FILE)
        pause()
        return 1

    ready = require_file(BACKEND_PYTHON, "请先按 README 执行 uv sync。")
    ready = require_file(VITE_ENTRY, "请先在 frontend 目录执行 pnpm install。") and ready
    node = shutil.which("node.exe") or shutil.which("node")
    if not node:
        print("\n未找到 Node.js。请按 README 安装 Node.js 后重试。")
        ready = False
    if not ready:
        pause()
        return 1

    start_services()
    print("启动后端...")
    start_service(
        [
            str(BACKEND_PYTHON),
            "-m",
            "uvicorn",
            "app.api.server:app",
            "--host",
            "0.0.0.0",
            "--port",
            BACKEND_PORT,
            "--reload",
        ],
        PROJECT_ROOT,
    )
    print("启动前端...")
    start_service([node, str(VITE_ENTRY), "--host", "0.0.0.0"], FRONTEND_DIR)

    print("等待前端服务就绪...")
    for _ in range(30):
        time.sleep(1)
        if frontend_is_ready():
            webbrowser.open(FRONTEND_URL)
            print(f"\n启动完成，已打开 {FRONTEND_URL}")
            pause("按 Enter 关闭此启动窗口；前后端服务会继续运行...")
            return 0

    print("\n前端未在 30 秒内就绪，请查看新打开的前端终端窗口。")
    pause()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
