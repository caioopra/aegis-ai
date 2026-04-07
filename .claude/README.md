# AegisNode — Claude Code Agent Team

This directory configures a formal **agent team** for AegisNode so that
complex, multi-file work is routed to specialists instead of landing on
one generalist context. Everything here is checked into git; per-user
overrides live in `.claude/settings.local.json` (gitignored).

## Layout

```
.claude/
├── agents/            # specialized sub-agent definitions (8)
├── skills/            # user- and model-invocable skills (8)
├── hooks/             # shell scripts triggered by lifecycle events
├── settings.json      # project-level hooks + permissions (committed)
└── settings.local.json # per-user overrides (gitignored)
```

## Agent roster

Delegate via the `Agent` tool with `subagent_type="<name>"`. Each agent
owns a specific slice of the codebase and is read-only outside that
slice — cross-cutting changes should be split across multiple agents.

| Agent | Scope | Tools | Model |
|---|---|---|---|
| `llm-prompt-specialist` | `src/aegis/llm.py`, `src/aegis/providers/*` — prompts, JSON parsing, token budget, retries | Read/Grep/Glob/Edit/Write/Bash | sonnet |
| `agent-graph-architect` | `src/aegis/agent/*`, `scripts/run_agent.py` — LangGraph state, nodes, runner, safety nets | Read/Grep/Glob/Edit/Write/Bash | sonnet |
| `rag-retrieval-specialist` | `src/aegis/rag/*`, `data/guidelines/*`, `scripts/ingest_guidelines.py` | Read/Grep/Glob/Edit/Write/Bash | sonnet |
| `mcp-fhir-specialist` | `src/aegis/mcp_server.py`, `src/aegis/fhir.py`, `data/synthea/sample_*` | Read/Grep/Glob/Edit/Write/Bash | sonnet |
| `medical-clinical-expert` | Clinical content review — prompts, drug interactions, SNOMED, pt-BR terminology, guidelines. **Read-only reviewer.** | Read/Grep/Glob/WebSearch/WebFetch | opus |
| `security-privacy-reviewer` | LGPD, PII leakage, prompt injection, secrets, trust boundaries. **Read-only.** | Read/Grep/Glob/Bash | sonnet |
| `test-quality-engineer` | `tests/*`, conftest fixtures, singleton reset, mocks, unit/integration split | Read/Grep/Glob/Edit/Write/Bash | sonnet |
| `infra-deployment-engineer` | `pyproject.toml`, `src/aegis/config.py`, `.env*`, provider SDKs, Docker, CI/CD | Read/Grep/Glob/Edit/Write/Bash | sonnet |

Each agent's markdown file in `.claude/agents/` contains its
non-negotiable invariants, verification commands, and escalation rules
to other agents. Read them before delegating so you know what you're
asking for.

### When to use which specialist

- Prompt wording, JSON schema, temperature, retry loops → **llm-prompt-specialist**
- New LangGraph node, state field, retry threshold, safety net → **agent-graph-architect**
- Chunking, BM25, hybrid search, Portuguese NLP, guideline ingest → **rag-retrieval-specialist**
- MCP tool signature, FHIR resource handling, drug interactions → **mcp-fhir-specialist**
- "Is this clinical content correct?" for a Brazilian physician → **medical-clinical-expert**
- "Does this leak PII / violate LGPD / enable prompt injection?" → **security-privacy-reviewer**
- Test infrastructure, flaky tests, mock strategies, fixtures → **test-quality-engineer**
- uv, pyproject, config loader, Docker, AWS, Gemini switch → **infra-deployment-engineer**

### When to use *multiple* specialists in parallel

Multi-file changes often cross specialist boundaries. Spawn agents in
parallel (single message, multiple `Agent` tool calls) when:

- A prompt change also touches LangGraph state → llm-prompt-specialist
  **and** agent-graph-architect.
- A new MCP tool should be dynamically selected by the agent → mcp-fhir-specialist
  **and** agent-graph-architect.
- A guideline file is added and the retriever benchmark should reflect it →
  rag-retrieval-specialist **and** (after merge) medical-clinical-expert.
- Any commit with clinical content AND staged diffs → clinical-review
  **and** security-review in parallel via the `/commit` skill.

## Skills

User-invocable skills live in `.claude/skills/<name>/SKILL.md`. Invoke
with `/name` (e.g., `/run-tests`) or let Claude auto-invoke based on
the skill's `description`.

| Skill | What it does |
|---|---|
| `/run-tests` | Runs `uv run pytest -m "not integration" -q` and reports pass/fail |
| `/lint-and-format` | Runs `ruff check` + `ruff format` across src/tests/scripts |
| `/pre-commit-check` | Full gate: tests + lint + format; reports GO/NO-GO |
| `/security-review` | Delegates staged diff to `security-privacy-reviewer` |
| `/clinical-review` | Delegates staged diff to `medical-clinical-expert` |
| `/phase-status` | Reads `notes/PLAN.md` progress tracker and reports current phase |
| `/ingest-guidelines` | Re-ingests `data/guidelines/` into Qdrant (dense + BM25) |
| `/commit [message]` | Full commit workflow: pre-commit-check → targeted reviews → `git commit` |

## Commit workflow (`/commit`)

The `/commit` skill is the recommended path for non-trivial changes.
It runs:

1. **Pre-commit gate** — unit tests + lint + format must pass.
2. **Targeted reviews** in parallel, based on staged files:
   - Touches prompts, `mcp_server.py` drug dict, `nodes.py` safety nets,
     guidelines, or sample patients → `clinical-review`.
   - Touches `llm.py`, `config.py`, `.env*`, or adds outbound HTTP →
     `security-review`.
3. **Commit** only if every gate and review is clean.

A reviewer's **Blocker** / **Critical** finding halts the workflow.
Override with an explicit "commit anyway" instruction, and the skill
will add a `[review-override]` trailer to the commit message.

## Hooks

Project-level hooks are in `.claude/settings.json`:

- **`PostToolUse` on `Edit|Write`** → `bash .claude/hooks/ruff-format-on-edit.sh`
  - Auto-formats Python files inside `src/`, `tests/`, or `scripts/`
    immediately after Claude writes to them.
  - Non-blocking: exits 0 even on failure.
  - Requires `jq` and `uv`. Silently skips if either is missing.

No blocking hooks are installed by default. Use `/commit` for gating
— it's more explicit and easier to override when necessary.

## Adding to the team

To add a new specialist:

1. Create `.claude/agents/<name>.md` with the frontmatter fields:
   `name`, `description` (used for auto-delegation), `tools`,
   `model`.
2. The body should list: files owned, invariants, verification
   commands, and escalation rules to other agents.
3. Update this README's roster table.
4. Update `.claude/skills/commit/SKILL.md` if the new agent should
   participate in the commit workflow.

To add a new skill:

1. Create `.claude/skills/<name>/SKILL.md` with frontmatter fields:
   `name`, `description`, `allowed-tools`, optionally `argument-hint`.
2. Keep the body scoped to one verb — composable skills beat one
   mega-skill.
3. Add to this README's skills table.
