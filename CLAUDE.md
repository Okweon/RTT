# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## important-instruction-reminders
- Do what has been asked; nothing more, nothing less.
- NEVER create files unless they're absolutely necessary for achieving your goal.
- ALWAYS prefer editing an existing file to creating a new one.
- NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.
- Documentation is not optional - it's a core deliverable for every task. Update documentation after completion of discussion, planning, implementation, and testing as directed in the following sections. 

## Repository Directory Structure
```
CLAUDE.md                  # claude code policies and rules (Do not update CLAUDE.md unless specified)
README.md                  # Project overview, tech stack, and quick start (Do not update README.md unless specified)
docs/
├── 1-prd/                 # Product Requirements Document (Do not update unless specified)   
├── 2-current/             # Design/system documentation to reflect current implementation
├── 3-reference/           # External reference (manuals, etc)
└── tickets/               # collection of tickets     
		├── archive/       # collection of old tickets   
		├── TD000X/        # ticket TD000X folder  
		└── index.md       # index of all ticket numbers and titles. 
scripts/				   # Shell scripts for the entire repo
database/				   # database sql scripts 
├── init_db.sql            # database initialization sql script (**CRITICAL**: need update whenever schema in live database is being updated manually)
└── backup/                # database backup directory
ffi-api/				   # fastapi project root
├── .logs/                 # log files from fastapi server
├── .uv_cache/             # uv cache directory
├── scripts/               # scripts for the fastapi projects (including live testing)
├── ffi/               	   # python module ffi
│   ├── __init__.py
│   └── py.typed
├── pyproject.toml         # Python project configuration (uv managed)
├── .python-version        # Python version specification
└── main.py                # Main application entry point
ffi-web/				   # nextjs project root
├── .logs/                 # log files from nextjs server
└── app/               	   # nextjs app router directory
.env					   # environment variable definition files for this project. (Do not update unless specified)  
docker-compose.yml		   # database and other components
```

## TICKETS Workflow

**CRITICAL**: Ticket system is a main workflow that defines how this project should be managed. Each ticket defines a task request from the user. Each ticket is assigned to have its own directory under docs/tickets/.

Each ticket has sequence number of the format TD000X (in regex ^T[SJGN]\d{4}$, starting with T, developer code [SJGN], 4 digits 0 padded incremented by 1 integer number within the same developer code). 

Under each ticket directory like docs/tickets/TD000X/, the following documents should be managed. These documents should have FIXED naming. The number suggests the order of tasks. 

```
docs/tickets/TD000X
├── 1-definition.md
├── 2-plan.md
├── 3-spec.md
├── 4-implementation-status.md
├── 5-test-status.md
├── 6-issues.md
└── 7-final.md          
```

- `1-definition.md`: This definition file contains the task objective, related requirements, out-of-scope items (for clarification if any) and success criteria if any.

- `2-plan.md`: Based on `1-definition.md`, plan how to tackle given task. The plan should be concise. This plan should NOT contain time estimation. Need to add this ticket name to docs/tickets/index.md when this file is created. 

- `3-spec.md`: Based on previous documents `1-definition.md` and `2-plan.md`, define specifications. You need to clearly define input and output format, major error cases. The specification should be have enough details for code generation. 

- `4-implementation-status.md`: Based on `3-spec.md`, source code should be implemented. This document should contain the current implementation status. If any issue is identified during implementation, it should be registered at `6-issues.md`.  

- `5-test-status.md`: Based on `3-spec.md`, testing scripts should be implemented. This document should contain the current testing results. If any issue is identified during testing, it should be registered at `6-issues.md`. See the testing section for detailed testing insturctions. 

- `6-issues.md`: Keep all identified issues. 

- `7-final.md`: Final reports on this task. Summarize all events happened in this ticket. Address remaining issues identified. You may suggest next steps. Keep it simple and ultra consice.  

**IMPORTANT**: DO NOT USE TICKET NUMBER (e.g. TS0064) AS PART OF CLASS, VARIABLE, FUNCTION, MODULE, AND/OR SCRIPT NAME. ONLY USE IT IN COMMENTS OR DOCSTRING AS REFERENCE.
**CRITICAL**: While working on a ticket, if the user requests some changes to codebase, always append them to `1-definion.md` (or rewrite the sections if explicitly requested) first, then update `2-plan.md` and `3-spec.md` before applying the changes to the codebase. Always remember documentation first before implementation.   

### TICKETS Index

To maintain index of all tickets, each ticket name should be indexed in docs/tickets/index.md. The content of the docs/tickets/index.md should be EXACTLY the following:

```
TS0001 {Brief Name of the task generated baesd on 1-definition.md file under docs/tickets/TS0001/}
TJ0001 {Brief Name of the task generated baesd on 1-definition.md file under docs/tickets/TJ0001/}
TS0002 {Brief Name of the task generated baesd on 1-definition.md file under docs/tickets/TS0002/}
TS0003 {Brief Name of the task generated baesd on 1-definition.md file under docs/tickets/TS0003/}
TJ0002 {Brief Name of the task generated baesd on 1-definition.md file under docs/tickets/TJ0002/}
...
```
No other content should be included. The new ticket MUST BE ADDED when creating `2-plan.md` file for the first time for that ticket. New ticket numbers should be appended to the end in the order of creation. 

### TICKETS Workflow Archive for Older Tickets

Older tickets are moved to docs/tickets/archive/. If a ticket directory is not found under docs/tickets/, you should look for docs/tickets/archive/TD000X. 

## FFI Project Overview

- ffi-api: python fastapi for FFI Agentic AI system backend
- ffi-web: nextjs frontend for FFI Agentic AI system
- postgres database with pgvector support

### ffi-api specific information

- always use 'uv' as package manager for python. Use 'uv run python ...'.
- [Coding Style] prefer to use/implement async functions over synchronous version. 
- [Coding Style] keep each function short and modular. prefer functional programming style whenever possible with PyMonad etc but use traditional style if it fits better. 
- Follow the established Python package structure for ALL source code. Do NOT create any source code in the root directory. Creating source code outside this structure will break package imports and violate project consistency standards.

- To run python script using uv
```bash
cd ffi-api # always set working directory at ffi-api/
UV_CACHE_DIR=.uv_cache uv run python ...
```

#### ffi-api Logging

- Always setup logging to both file and console. Log files should be stored under ffi-api/.logs/ with timestamp in the file name. 
- The default log-level to file should be 'DEBUG'
- The default log-level to console should be 'INFO' 

#### ffi-api Testing

- make the script to use .env in the root
- create testing scripts under ffi-api/scripts/ only with naming convention of "test_". DO NOT PLACE THEM ANYWHERE ELSE.
- No unit testing or integration testing unless explicitly asked. Only live testing with live database. 

### ffi-web specific information

#### Tech Stack
- nextjs 
- typescript
- tailwind css
- shadcn/ui
- bun package manager
- tanstack table (headless UI for building tables & datagrids)

#### ffi-web nextjs architecture 

- App Router
- Server Action (over API)
- Server Side Rendering (SSR) (over data fetching from client side) 

#### ffi-web Live Testing using Chrome Devtools MCP
- **IMPORTANT**: see docker_compose.yml and .env for live database info and API keys.
- Both fastapi and nextjs servers are typically setup using auto-reloading and running at background (port 8000 and 3000, respectively)
- Code changes will be affected immediately.
- Use Chrome Devtools mcp server to access localhost:3000 for testing.
- Inspect log files from the servers.

### Postgres database specific information

```bash
> docker ps
CONTAINER ID   IMAGE                    COMMAND                  CREATED          STATUS                    PORTS                                         NAMES
c7b98195be04   dpage/pgadmin4:latest    "/entrypoint.sh"         20 seconds ago   Up 5 seconds              0.0.0.0:8082->80/tcp, [::]:8081->80/tcp       ffi_pgadmin
906bf6f39d15   pgvector/pgvector:pg17   "docker-entrypoint.s…"   20 seconds ago   Up 18 seconds (healthy)   0.0.0.0:5433->5432/tcp, [::]:5432->5432/tcp   ffi_postgres
```

- How to use psql:
```bash
docker exec -i ffi_postgres psql -U ffi_user -d ffi -c "\d users"
```

#### database reset (when requested)
- reset database by removing docker volume ffi_postgres_data (not ffi_pgadmin_data) and respawn database.

## LLM models (litellm)

openai/gpt-5-mini (default, temperature=1, no streaming)
anthropic/claude-sonnet-4-5-20250929 
anthropic/claude-haiku-4-5-20251001