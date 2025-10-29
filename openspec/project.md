# Project Context

## Purpose
TBD. Working title inferred from path: "AIoT HW3". Add a 1–2 sentence description of what this project aims to deliver (e.g., collect and process IoT sensor telemetry, run ML inference at the edge, visualize metrics, etc.).

## Tech Stack
- TBD. No application code detected yet (only OpenSpec scaffolding). Please list planned/runtime stacks here.
- Examples to choose from:
  - Python (FastAPI, Pydantic, Uvicorn) for APIs; Pandas/Numpy for data; MQTT via `paho-mqtt`
  - Node.js/TypeScript (Express/Fastify) for APIs; MQTT via `mqtt`; frontend with React/Vite
  - Edge/embedded: MicroPython/Arduino for device firmware; MQTT/HTTP uplink

## Project Conventions

### Code Style
- TBD based on chosen language.
- Recommended defaults:
  - Python: Black (format), Ruff (lint), isort (imports)
  - TypeScript/JavaScript: Prettier (format), ESLint (lint) with Airbnb/Base config
  - Commit messages: Conventional Commits (e.g., feat:, fix:, chore:)

### Architecture Patterns
- Spec-first development using OpenSpec. Capabilities live under `openspec/specs/`; proposed changes under `openspec/changes/`.
- Prefer simple, single-purpose modules; keep implementations under ~100 lines until proven otherwise.
- Clear boundaries per capability (e.g., `auth`, `telemetry`, `notifications`).

### Testing Strategy
- TBD. Choose based on stack.
- Suggested defaults:
  - Python: `pytest` with coverage; unit tests near code; integration tests for broker/API
  - TypeScript: `vitest` or `jest`; integration tests for endpoints and messaging
  - Define acceptance tests mapped to OpenSpec scenarios where feasible

### Git Workflow
- TBD. Recommended: GitHub Flow (main + short-lived feature branches) or trunk-based with small PRs.
- Use PR templates referencing relevant change IDs (e.g., `openspec/changes/add-...`).

## Domain Context
- TBD. If AIoT-related, note device constraints (power, connectivity), data rates, QoS needs, and security considerations (device auth, credentials handling).

## Important Constraints
- TBD. Capture performance targets, latency/SLA, data retention, privacy/security requirements, course deliverables, or deployment constraints (on-prem/cloud-edge).

## External Dependencies
- TBD. List brokers/services/APIs (e.g., MQTT broker address/QoS, database, cloud services), along with auth methods and rate limits.
