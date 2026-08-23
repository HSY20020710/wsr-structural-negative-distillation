# API Contract v0.1

- `GET /health`
- `GET /api/v1/project/summary`
- `GET /api/v1/cases`
- `GET /api/v1/cases/{case_id}`
- `POST /api/v1/gate/check`
- `POST /api/v1/experiments/run`
- `GET /api/v1/experiments/{run_id}`

后续阶段增加：SSE/WebSocket event stream、artifact download、private dataset search、long-running job queue。
