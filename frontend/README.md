# DeepSearch Agents Frontend

React + Vite + Tailwind CSS + Ant Design frontend for the DeepSearch Agents FastAPI backend.

## Run

```bash
pnpm install
pnpm dev
```

By default the app talks to `http://localhost:8001` and `ws://localhost:8001`.
Override with `.env.local`:

```bash
VITE_API_BASE_URL=http://localhost:8001
VITE_WS_BASE_URL=ws://localhost:8001
```

## Backend Contract

- `POST /api/task`
- `POST /api/upload`
- `GET /api/files`
- `GET /api/download`
- `WebSocket /ws/{thread_id}`
