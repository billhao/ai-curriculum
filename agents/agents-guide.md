# LLM Agents

A step-by-step guide to how LLMs become autonomous agents. Covers the research lineage, architectures, memory, planning, multi-agent systems, code agents, evaluation, training, failure modes, and practical patterns.

## Background

**Foundational paper**: [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) (Yao et al., Princeton/Google, Oct 2022, ICLR 2023)

**Research lineage** — LLM agents evolved from tool-use prompting to autonomous systems:

1. **WebGPT** (Nakano et al., OpenAI, Dec 2021, [arxiv 2112.09332](https://arxiv.org/abs/2112.09332)) — Fine-tuned GPT-3 to browse the web via special actions (search, click, scroll, quote). Trained with imitation learning on human demonstrations, then optimized with human feedback. First demonstration that LLMs could learn to interact with external systems through structured actions.

2. **ReAct** (Yao et al., Princeton/Google, Oct 2022, [arxiv 2210.03629](https://arxiv.org/abs/2210.03629)) — Introduced the Thought-Action-Observation loop for interleaving reasoning traces with tool actions via few-shot prompting. No fine-tuning needed. This pattern became the backbone of nearly every agent framework that followed. Published at ICLR 2023.

3. **Toolformer** (Schick et al., Meta, Feb 2023, [arxiv 2302.04761](https://arxiv.org/abs/2302.04761)) — Self-supervised approach to tool learning. The model taught itself when and how to insert API calls by filtering for calls that reduced perplexity on future tokens. No human demonstrations needed. Published at NeurIPS 2023.

4. **Reflexion** (Shinn et al., Mar 2023, [arxiv 2303.11366](https://arxiv.org/abs/2303.11366)) — Verbal reinforcement learning for agents. Instead of updating weights, the agent stores natural-language reflections on past failures in an episodic memory buffer, then uses them as context in subsequent attempts. Completed 130/134 AlfWorld tasks with ReAct + Reflexion. Published at NeurIPS 2023.

5. **AutoGPT** (Significant Gravitas, Mar 2023, [github](https://github.com/Significant-Gravitas/AutoGPT)) — Open-source project that chained GPT-4 calls in an autonomous loop with internet access, file I/O, and memory. Became the fastest-growing GitHub repo in history (100K+ stars in weeks). Demonstrated massive public appetite for autonomous agents, despite limited reliability.

6. **BabyAGI** (Nakajima, Apr 2023, [github](https://github.com/yoheinakajima/babyagi)) — Minimal task-management agent that maintains a task list, executes tasks, and creates new sub-tasks based on results. Introduced the plan-execute-replan pattern in a simple, hackable form.

7. **Voyager** (Wang et al., NVIDIA, May 2023, [arxiv 2305.16291](https://arxiv.org/abs/2305.16291)) — First LLM-powered embodied lifelong learning agent, operating in Minecraft. Three key modules: automatic curriculum, skill library, and iterative prompting for executable code. Unlocked diamond tools 15.3x faster than prior SOTA. Demonstrated that agents could build persistent skill libraries and continually learn.

8. **Tree of Thoughts** (Yao et al., Princeton, May 2023, [arxiv 2305.10601](https://arxiv.org/abs/2305.10601)) — Extended chain-of-thought to a tree structure where the model explores multiple reasoning paths, evaluates them, and backtracks. Enabled deliberate planning with BFS/DFS search. Published at NeurIPS 2023.

9. **Language Agent Tree Search (LATS)** (Zhou et al., Princeton, Oct 2023, [arxiv 2310.04406](https://arxiv.org/abs/2310.04406)) — Unified reasoning, acting, and planning by combining ReAct-style agents with Monte Carlo Tree Search. The LLM serves as the value function, policy, and world model simultaneously.

10. **SWE-agent** (Yang et al., Princeton, Apr 2024, [arxiv 2405.15793](https://arxiv.org/abs/2405.15793)) — Purpose-built agent interface (ACI) for software engineering. Custom commands for file navigation, editing, and testing in a sandboxed environment. Demonstrated that agent-computer interface design matters as much as the underlying LLM.

11. **Devin** (Cognition Labs, Mar 2024) — First commercial "AI software engineer." Operates autonomously in a sandboxed environment with shell, editor, and browser. Scored 13.86% on SWE-bench (vs. prior SOTA of 1.96%) at launch. Devin 2.0 (2025) added interactive planning, codebase search, and parallel agent instances.

12. **Claude Code** (Anthropic, Feb 2025) — Terminal-based agentic coding tool. Architecture: single-threaded master loop (`while(tool_call) → execute → feed results → repeat`). Three-phase execution: gather context, take action, verify results. Sub-agents for parallelism with strict depth limits. By mid-2025, became the dominant agentic coding tool among professional developers.

13. **OpenHands** (Wang et al., UIUC, Jul 2024, [arxiv 2407.16741](https://arxiv.org/abs/2407.16741)) — Open-source platform for AI software developers. CodeAct 2.1 achieved 53% on SWE-bench Verified. Modular SDK architecture with sandboxed execution, event-sourced state, and model-agnostic design. Published at ICLR 2025.

14. **OpenAI Codex** (OpenAI, May 2025) — Cloud-based coding agent powered by codex-1 (o3 optimized for SWE). Runs tasks in parallel sandboxed environments. GPT-5.3-Codex (Feb 2026) expanded to general knowledge work beyond coding.

15. **Cursor Agent** (Anysphere, Oct 2025) — Shipped Composer, a MoE-based agentic coding model. Fork of VS Code with deep IDE integration: file editing, terminal execution, browser interaction. Agent mode with plan→execute→verify workflow. By 2026, became the standard for agentic coding in IDE form.

16. **Model Context Protocol (MCP)** (Anthropic, Nov 2024) — Open standard for connecting agents to external tools and data. JSON-RPC 2.0 transport, inspired by LSP. Adopted by OpenAI (Mar 2025), Google DeepMind (Apr 2025). Donated to Linux Foundation's Agentic AI Foundation (Dec 2025). 10,000+ public MCP servers, 97M+ monthly SDK downloads by early 2026.

**The arc**: prompting-based agents (ReAct) → autonomous loops (AutoGPT, BabyAGI) → embodied agents with skill libraries (Voyager) → specialized code agents (SWE-agent, Devin, Claude Code) → production-grade frameworks with standardized tool protocols (MCP, Agent SDK) → RL-trained agents (Agent-R1, WebAgent-R1).

## Key Terms

**Agent**: A system where an LLM operates in a loop — perceiving its environment, reasoning about what to do, taking actions, and observing results — to accomplish a goal. The key distinction from a chatbot: an agent has *autonomy* over its action sequence. It decides what to do next, not the user.

**Environment**: The world the agent interacts with. For a coding agent, this is a filesystem + terminal + browser. For a web agent, it's a browser DOM. For Voyager, it's the Minecraft game state. The environment receives actions and returns observations.

**Observation**: What the agent perceives after taking an action. Terminal output, file contents, browser HTML, error messages, test results. Observations are appended to the agent's context as new information.

**Action space**: The set of actions available to the agent. For Claude Code: `read_file`, `edit_file`, `bash`, `search`, `web_fetch`. For a web agent: `click`, `type`, `scroll`, `navigate`. A well-designed action space is critical — too many actions confuse the model, too few limit capability.

**Trajectory**: The complete sequence of (observation, thought, action) tuples from task start to completion. A single "episode" of agent behavior. Training data for agent SFT/RL is stored as trajectories.

**Episode**: One complete attempt at a task, from initial prompt to final output or termination. In RL for agents, episodes are the unit of training — each episode produces a reward signal.

**Scaffold / Harness**: The non-LLM code that wraps the model and gives it agency. The while-loop, tool executor, context manager, error handler, and output parser. Claude Code is a scaffold around Claude. SWE-agent is a scaffold around GPT-4/Claude. The scaffold is what turns a language model into an agent.

**Orchestrator**: The component that manages agent execution — routing tasks, managing state, handling tool calls, enforcing limits (max steps, max tokens, timeouts). In multi-agent systems, the orchestrator coordinates between agents.

**Grounding**: Connecting the agent's reasoning to real-world data. An agent is "grounded" when its actions are based on actual observations (file contents, test output) rather than hallucinated assumptions.

## Agent Architectures

### Single-Agent Loop

The simplest and most common architecture. One LLM in a loop with tools.

```
┌──────────────────────────────────────────────────────────────────┐
│                     SINGLE-AGENT LOOP                            │
│                                                                  │
│   User Task                                                      │
│      │                                                           │
│      v                                                           │
│   ┌─────────┐     ┌──────────┐     ┌─────────────┐              │
│   │         │     │          │     │             │              │
│   │  Think  │────>│  Act     │────>│  Observe    │───┐          │
│   │         │     │ (tool    │     │  (tool      │   │          │
│   │         │     │  call)   │     │   result)   │   │          │
│   └─────────┘     └──────────┘     └─────────────┘   │          │
│       ^                                               │          │
│       └───────────── loop until done ────────────────┘          │
│                           │                                      │
│                           v                                      │
│                     Final Response                               │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

This is Claude Code's architecture. The pseudocode is deceptively simple:

```python
def agent_loop(task, tools, max_steps=50):
    messages = [system_prompt(tools), user_message(task)]

    for step in range(max_steps):
        response = llm(messages)
        messages.append(response)

        if response.has_tool_calls:
            for call in response.tool_calls:
                result = execute(call)
                messages.append(tool_result(call.id, result))
        else:
            return response.text  # done — no more tool calls

    return "Max steps reached"
```

The model controls everything: what tools to call, in what order, when to stop. The scaffold just executes what the model asks for and feeds results back.

**When it works**: Most tasks. Single-agent loops with a capable model (Claude Opus 4.6, GPT-5.4) solve the vast majority of practical problems. The simplicity makes them debuggable and reliable.

**When it breaks**: Tasks requiring very long context (100+ tool calls), tasks needing true parallelism, or tasks where a single model lacks the specialized knowledge for every sub-task.

### Multi-Agent Systems

Multiple LLM instances collaborating on a task, each potentially with different roles, prompts, or even different models.

```
┌───────────────────────────────────────────────────────────────────┐
│                    MULTI-AGENT PATTERNS                           │
│                                                                   │
│  A. Assembly Line          B. Debate/Critique                     │
│                                                                   │
│  ┌───────┐  ┌───────┐     ┌───────┐   ┌───────┐                 │
│  │Planner│─>│Executor│     │Agent A│<─>│Agent B│                 │
│  └───────┘  └───┬───┘     │(solve)│   │(critiqu│                │
│                  │         └───┬───┘   └───┬───┘                 │
│            ┌─────v─────┐       └─────┬─────┘                     │
│            │ Verifier  │             v                            │
│            └───────────┘       ┌───────────┐                     │
│                                │ Consensus │                     │
│  C. Division of Labor          └───────────┘                     │
│                                                                   │
│         ┌──────────┐                                             │
│         │Orchestrat│                                             │
│         └────┬─────┘                                             │
│      ┌───────┼────────┐                                          │
│      v       v        v                                          │
│  ┌──────┐┌──────┐┌──────┐                                       │
│  │Search││Code  ││Review│                                       │
│  │Agent ││Agent ││Agent │                                       │
│  └──────┘└──────┘└──────┘                                       │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

**Assembly line (MetaGPT)**: Agents in sequence, each performing a specialized phase. MetaGPT implements a software development pipeline: ProductManager → Architect → Engineer → QA. Each agent outputs structured artifacts that the next agent consumes.

**Debate/critique**: Two or more agents argue about the correct answer. One proposes, another critiques, they iterate until convergence. Useful for reducing hallucination and increasing robustness on reasoning tasks.

**Division of labor**: An orchestrator decomposes the task and delegates sub-tasks to specialized agents. Each specialist has its own tools and system prompt. Results are collected and synthesized.

**The reliability trap**: Multi-agent systems compound errors. If each agent has 95% accuracy per step and there are 5 sequential steps: 0.95^5 = 77% overall success. A study in 2025 found multi-agent systems can introduce 17x more errors than single-agent approaches when not carefully designed. The solution is verification at each handoff — never blindly trust an upstream agent's output.

### Hierarchical Agents

A tree of agents where parent agents delegate to children, children can delegate further. Bounded recursion.

```
┌─────────────────────────────────────────────────────────────┐
│                   HIERARCHICAL AGENTS                       │
│                                                             │
│                    ┌──────────┐                             │
│                    │  Master  │  Depth 0                    │
│                    │  Agent   │                             │
│                    └────┬─────┘                             │
│               ┌─────────┼──────────┐                       │
│               v         v          v                       │
│          ┌────────┐┌────────┐┌────────┐                    │
│          │Sub-    ││Sub-    ││Sub-    │  Depth 1           │
│          │agent A ││agent B ││agent C │                    │
│          └────┬───┘└────────┘└────────┘                    │
│               v                                            │
│          ┌────────┐                                        │
│          │Sub-sub ││                     Depth 2 (max)     │
│          │agent   ││                                       │
│          └────────┘                                        │
│                                                             │
│  Rules:                                                     │
│  - Max depth enforced (prevents infinite recursion)         │
│  - Children have isolated context windows                   │
│  - Parent only sees child's final output, not full trace    │
│  - Each level can use different models (cost optimization)  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

Claude Code uses this pattern: the master agent loop can spawn sub-agents (called "Task" agents) for exploration or parallel work. Sub-agents run in isolated context windows with strict depth limits — a sub-agent cannot spawn its own sub-agents beyond a set depth. This prevents runaway recursion while enabling focused parallel exploration.

**Key design principle**: The parent agent should only receive a distilled summary from each child, not the full trajectory. This preserves the parent's context window for high-level reasoning.

## Memory Systems

The central challenge for agents: LLMs are stateless. Every call starts fresh. Memory must be engineered into the scaffold.

### Short-Term Memory (Context Window)

The conversation history itself — the message array passed to the LLM on each call. This is the agent's "working memory."

```
┌──────────────────────────────────────────────────────────┐
│              CONTEXT WINDOW AS WORKING MEMORY             │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │ System prompt (tools, instructions)          ~2K   │  │
│  │ User task                                    ~0.5K │  │
│  │ Step 1: thought + action + observation       ~1K   │  │
│  │ Step 2: thought + action + observation       ~2K   │  │
│  │ Step 3: thought + action + observation       ~5K   │  │
│  │ ...                                                │  │
│  │ Step 47: thought + action + observation      ~3K   │  │
│  │                                                    │  │
│  │ Total consumed: 187K / 200K token limit            │  │
│  │ Remaining for next response: 13K                   │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  Problem: Context fills up. Early observations get       │
│  pushed out or compressed. Agent "forgets" what it       │
│  learned in step 3.                                      │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

**Context window sizes as of March 2026**:

| Model              | Context Window  | Effective for Agents |
|--------------------|-----------------|----------------------|
| Claude Opus 4.6    | 1M tokens       | ~200-400 tool steps  |
| GPT-5.4            | 1M tokens       | ~200-400 tool steps  |
| Gemini 3.1 Pro     | 2M tokens       | ~400-800 tool steps  |
| Qwen3.5-72B        | 256K tokens     | ~50-100 tool steps   |
| Llama 4 Maverick   | 1M tokens       | ~200-400 tool steps  |

Even 1M tokens runs out on complex tasks. A single file read can be 10-50K tokens. Test output can be 5-20K. After 50-100 tool calls, the context is often saturated.

### Working Memory Management

Strategies to keep the context window useful:

**Summarization**: Periodically compress old trajectory steps into a summary. Replace 50 detailed steps with a 500-token summary of what was tried and learned.

**Sliding window**: Drop the oldest messages, keeping only the most recent N steps plus the original task and system prompt. Simple but lossy — the agent may repeat mistakes from forgotten steps.

**Selective retention**: Keep high-value observations (error messages, test results, key findings) and drop low-value ones (file listings, successful but unremarkable operations). Requires heuristics or a second LLM call to judge value.

**Context compaction**: Claude Code's approach — when context gets long, the system compacts the conversation by summarizing prior exchanges while preserving critical information like file paths, error messages, and the current plan.

### Long-Term Memory

Persists across sessions and episodes. Three main approaches:

**RAG / Vector stores**: Embed past observations, solutions, and knowledge into a vector database. At each step, retrieve the most relevant memories based on the current state. Used by many agent frameworks for knowledge retrieval.

```
Agent encounters error → embed error + context → search vector DB
→ find similar past error + solution → inject into prompt
```

**File-based memory**: Write knowledge to files that the agent can later read. Voyager's skill library is exactly this — successful code functions are saved with descriptions and retrieved when similar tasks arise. CLAUDE.md files serve a similar purpose: persistent instructions and knowledge that load into every session.

**Structured databases**: Store memories in schemas — task outcomes, user preferences, codebase facts. More structured than vectors, more queryable.

### Episodic Memory (Reflexion Pattern)

Reflexion (Shinn et al., 2023) introduced the most influential memory pattern for agents:

```
┌──────────────────────────────────────────────────────────┐
│                  REFLEXION MEMORY                         │
│                                                          │
│  Episode 1: Agent attempts task → Fails                  │
│      │                                                   │
│      v                                                   │
│  Self-reflect: "I failed because I didn't check the      │
│  return type of parse(). I assumed it returns a string    │
│  but it returns Optional[str]. Next time, always check   │
│  return types before using results."                     │
│      │                                                   │
│      v                                                   │
│  Store reflection in episodic memory buffer              │
│                                                          │
│  Episode 2: Agent attempts same task                     │
│  Context includes: [task] + [reflection from episode 1]  │
│  → Agent checks return type → Succeeds                   │
│                                                          │
│  Episode 3: Agent attempts new similar task              │
│  Context includes: [task] + [all past reflections]       │
│  → Agent applies learned lessons → Succeeds faster       │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

The key insight: instead of updating model weights (expensive, requires RL infrastructure), store verbal self-critiques as text and inject them into future prompts. The model learns from its mistakes through in-context learning, not gradient updates. This is the "verbal reinforcement learning" idea.

Components:
- **Actor**: Generates actions based on observations (the agent's policy)
- **Evaluator**: Scores the episode outcome (binary success/fail or scalar)
- **Self-Reflection**: An LLM call that analyzes why the episode failed and produces a natural-language lesson
- **Memory buffer**: Stores reflections, capped at a max count to prevent context overflow

Results: ReAct + Reflexion solved 130/134 AlfWorld tasks, vs. ReAct alone solving far fewer. On HumanEval (code generation), Reflexion achieved 91% pass@1.

## Planning

How agents decompose tasks into steps and decide what to do.

### ReAct (Reasoning + Acting)

The simplest planning strategy — interleave one thought with one action, repeat. No upfront plan. The model figures out the next step based on what it has observed so far.

```
Thought: I need to find the bug in the test file. Let me read the test first.
Action:  read_file("tests/test_parser.py")
Observation: [file contents with 3 test functions]

Thought: test_parse_json is failing. Let me run it to see the error.
Action:  bash("python -m pytest tests/test_parser.py::test_parse_json -v")
Observation: AssertionError: expected {'key': 'value'} but got None

Thought: The parser returns None. Let me check the parse() function.
Action:  read_file("src/parser.py")
...
```

**Strength**: Reactive, handles unexpected situations well. The model can change direction immediately based on new information.

**Weakness**: No lookahead. The model may wander, revisit dead ends, or take inefficient paths. For complex multi-step tasks, it can get lost.

### Plan-Then-Execute

Generate a full plan upfront, then execute each step:

```
┌──────────────────────────────────────────────────────────┐
│              PLAN-THEN-EXECUTE                            │
│                                                          │
│  Phase 1: Planning                                       │
│  ┌────────────────────────────────────────────────────┐  │
│  │ Task: "Add rate limiting to the API"               │  │
│  │                                                    │  │
│  │ Plan:                                              │  │
│  │ 1. Read the current API route handlers             │  │
│  │ 2. Design a rate limiter middleware                 │  │
│  │ 3. Implement the middleware in src/middleware/      │  │
│  │ 4. Add configuration (requests/min, per-IP)        │  │
│  │ 5. Wire middleware into the route handlers          │  │
│  │ 6. Write unit tests                                │  │
│  │ 7. Run tests and verify                            │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  Phase 2: Execution (step by step)                       │
│  Execute step 1 → observe → execute step 2 → ...        │
│                                                          │
│  Phase 3: Replanning (if needed)                         │
│  If step 3 reveals the project uses a framework          │
│  with built-in rate limiting → revise plan               │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

**Strength**: More efficient — fewer LLM calls because the model commits to a direction. Better for well-understood tasks.

**Weakness**: The initial plan may be wrong. Requires replanning logic when assumptions are violated.

Cursor's "Plan Mode" uses this pattern: it reads the codebase, asks clarifying questions, then produces a structured plan.md before writing any code.

### Tree-of-Thought Planning

For problems where the first approach might fail, explore multiple possibilities:

```
┌──────────────────────────────────────────────────────────┐
│              TREE OF THOUGHTS                             │
│                                                          │
│  Problem: "Optimize this function (3 possible approaches)"│
│                                                          │
│                    ┌─────────┐                           │
│                    │  Root   │                           │
│                    │ problem │                           │
│                    └────┬────┘                           │
│            ┌────────────┼────────────┐                   │
│            v            v            v                   │
│      ┌──────────┐ ┌──────────┐ ┌──────────┐            │
│      │Approach A│ │Approach B│ │Approach C│            │
│      │ Cache    │ │ Algo     │ │ Parallel │            │
│      │ results  │ │ change   │ │ compute  │            │
│      └────┬─────┘ └────┬─────┘ └────┬─────┘            │
│           │             │             │                   │
│    Eval: 0.7      Eval: 0.9     Eval: 0.5              │
│    (good)         (best)        (risky)                 │
│                         │                                │
│                    Expand B                              │
│                    ┌────┴────┐                           │
│                    v         v                           │
│              ┌────────┐ ┌────────┐                      │
│              │B1: Use │ │B2: Use │                      │
│              │heapq   │ │bisect  │                      │
│              └────────┘ └────────┘                      │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

At each node, the model (or a separate evaluator) scores the partial solution. High-scoring branches get expanded, low-scoring ones get pruned. BFS explores breadth-first (try all approaches one level deep), DFS goes depth-first (follow the most promising path). LATS (Language Agent Tree Search) combines this with MCTS — using the LLM itself as the value function to evaluate states.

**When to use**: Problems with multiple viable approaches where the cost of going down the wrong path is high. Math, optimization, complex refactoring.

### Dynamic Replanning

The most practical pattern for production agents: plan, but revise after every step.

```python
def plan_and_execute_with_replan(task, tools):
    plan = llm_plan(task)  # Initial plan: list of steps

    for i, step in enumerate(plan):
        result = execute_step(step, tools)

        # After each step, ask: is the plan still valid?
        revised_plan = llm_replan(
            original_task=task,
            completed_steps=plan[:i+1],
            results_so_far=results,
            remaining_plan=plan[i+1:],
            latest_observation=result
        )
        plan = plan[:i+1] + revised_plan  # Keep completed, update remaining
```

This combines the efficiency of upfront planning with the adaptability of reactive execution. Most production agents (Claude Code, Devin, Cursor) use variants of this.

## Reflection and Self-Evaluation

### The Self-Critique Loop

Beyond Reflexion's cross-episode learning, agents can self-critique within a single episode:

```
┌──────────────────────────────────────────────────────────┐
│                  SELF-CRITIQUE LOOP                       │
│                                                          │
│  Step 1: Agent produces candidate output                 │
│          "Here's the implementation of merge_sort..."    │
│                                                          │
│  Step 2: Agent critiques its own output                  │
│          "Wait — this doesn't handle empty lists.        │
│           Also, the base case should be len <= 1,        │
│           not len == 1."                                 │
│                                                          │
│  Step 3: Agent revises based on critique                 │
│          "Updated implementation with empty list check   │
│           and corrected base case."                      │
│                                                          │
│  Step 4: Agent verifies (runs tests)                     │
│          bash("python -m pytest test_sort.py")           │
│          → All 12 tests pass                             │
│                                                          │
│  Step 5: Confidence check — "Should I keep going?"       │
│          "Tests pass, edge cases covered. Done."         │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

Modern reasoning models (Claude Opus 4.6 with thinking, GPT-5.4) perform self-critique internally during their extended chain-of-thought. The thinking trace often contains phrases like "Wait, that's wrong" or "Let me reconsider" — the model is self-correcting before producing output. For agents, this means fewer tool calls wasted on obviously wrong approaches.

### Learning from Failures

Practical patterns for agents that improve over sessions:

1. **Error pattern logging**: After a failed episode, extract the error type and root cause. Store in a structured format: `{error: "TypeError", cause: "forgot to handle None return", fix: "always check return type"}`.

2. **Success/failure ratio tracking**: Monitor which tool sequences lead to success vs. failure. Over time, bias the agent toward successful patterns.

3. **Escalation**: If the agent fails the same task type repeatedly, escalate to a human or a more capable model. Don't keep burning compute on a doomed approach.

## Multi-Agent Systems

### Frameworks Landscape (March 2026)

| Framework       | Architecture         | Best For                          | Key Feature                    |
|----------------|----------------------|-----------------------------------|-------------------------------|
| LangGraph      | Graph-based workflow | Complex conditional pipelines     | Explicit state machine        |
| CrewAI         | Role-based teams     | Business process automation       | Intuitive role definitions    |
| AutoGen/MS Agent Framework | Conversational | Research, debate, brainstorming | Multi-turn agent conversations|
| OpenAI Agents SDK | Single/multi-agent | General purpose                  | Native tool use + handoffs    |
| Claude Agent SDK | Orchestrated agents | Production deployments           | Built-in guardrails          |

Microsoft merged AutoGen and Semantic Kernel into a unified "Microsoft Agent Framework" (Oct 2025), with general availability in Q1 2026.

LangChain explicitly shifted focus: "Use LangGraph for agents, not LangChain" — acknowledging that the original chain-based abstraction was insufficient for agent orchestration.

### Design Patterns

**Debate**: Two agents argue opposing positions on a task. A judge agent selects the better answer or synthesizes. Reduces hallucination because each agent is incentivized to find flaws in the other's reasoning.

```
┌────────────────────────────────────────────────────────┐
│                    DEBATE PATTERN                       │
│                                                        │
│  Round 1:                                              │
│  Agent A: "The bug is in the parser — line 42 uses     │
│            split(',') but the CSV has quoted fields."   │
│  Agent B: "I disagree — the parser handles quotes.     │
│            The real issue is the encoding: the file     │
│            is UTF-16 but we read it as UTF-8."         │
│                                                        │
│  Round 2:                                              │
│  Agent A: "Good point about encoding, but even after   │
│            fixing that, the comma-in-quotes issue       │
│            remains. Look at line 42..."                 │
│  Agent B: "You're right — both issues exist. The       │
│            encoding causes the first 3 failures,        │
│            the quoting causes the rest."                │
│                                                        │
│  Judge: Synthesizes — both bugs are real. Fix both.    │
│                                                        │
└────────────────────────────────────────────────────────┘
```

**Division of labor with handoffs**: The orchestrator routes sub-tasks to specialized agents. Each specialist has narrow tools and expertise. OpenAI's Agents SDK uses a "handoff" primitive — one agent can hand control to another, passing along context.

**Supervisor pattern**: A supervisor agent monitors worker agents, provides feedback, and can override or redirect. Useful for maintaining quality in long-running tasks.

### The Compounding Error Problem

A critical consideration for multi-agent systems:

```
Single agent, 10 steps, 95% accuracy per step:
  Overall: 0.95^10 = 59.9%

Multi-agent, 3 agents × 5 steps each, 95% accuracy per step, no verification:
  Overall: 0.95^15 = 46.3%

Multi-agent with verification at each handoff (catches 80% of errors):
  Effective per-step accuracy: 0.95 + (0.05 × 0.80) = 0.99
  Overall: 0.99^15 = 86.0%
```

The lesson: multi-agent systems only win when verification is cheap and reliable. Without verification at handoff points, adding agents makes things worse, not better.

## Code Agents

Code agents are the most commercially successful agent application as of 2026. They operate in sandboxed environments with filesystem, terminal, and browser access.

### Architecture Comparison

```
┌──────────────────────────────────────────────────────────────────┐
│              CODE AGENT ARCHITECTURES                             │
│                                                                  │
│  SWE-agent (2024)          Claude Code (2025)                    │
│  ┌──────────────────┐      ┌──────────────────┐                 │
│  │ Custom ACI       │      │ Standard tools   │                 │
│  │ (Agent-Computer  │      │ (bash, read,     │                 │
│  │  Interface)      │      │  edit, search)   │                 │
│  │ Custom commands:  │      │ Single-threaded  │                 │
│  │  open, goto,     │      │ master loop      │                 │
│  │  edit, scroll    │      │ Sub-agents for   │                 │
│  │ Linter feedback  │      │ parallel tasks   │                 │
│  └──────────────────┘      └──────────────────┘                 │
│                                                                  │
│  Devin (2024)              Cursor (2025)                        │
│  ┌──────────────────┐      ┌──────────────────┐                 │
│  │ Full sandboxed   │      │ IDE-native fork   │                 │
│  │ environment:     │      │ of VS Code        │                 │
│  │  shell + editor  │      │ Composer (MoE     │                 │
│  │  + browser       │      │ agentic model)    │                 │
│  │ Parallel Devins  │      │ RAG over full     │                 │
│  │ Interactive plan │      │ codebase (@symbols│                 │
│  │ RL-trained model │      │ Plan→Execute→     │                 │
│  └──────────────────┘      │ Verify workflow   │                 │
│                            └──────────────────┘                 │
│                                                                  │
│  OpenHands (2024)          Codex (2025)                         │
│  ┌──────────────────┐      ┌──────────────────┐                 │
│  │ Open-source SDK  │      │ Cloud-based       │                 │
│  │ CodeAct: function│      │ Parallel sandbox  │                 │
│  │ calling + bash   │      │ environments      │                 │
│  │ Event-sourced    │      │ codex-1 (o3 for   │                 │
│  │ state            │      │ SWE)              │                 │
│  │ Model-agnostic   │      │ 7+ hour tasks     │                 │
│  │ Sandboxed Docker │      │ PR generation     │                 │
│  └──────────────────┘      └──────────────────┘                 │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### The Edit-Test-Debug Loop

All code agents converge on the same fundamental loop:

```
┌──────────────────────────────────────────────────────────┐
│              EDIT → TEST → DEBUG LOOP                     │
│                                                          │
│  1. Understand                                           │
│     read files, grep codebase, understand structure      │
│           │                                              │
│           v                                              │
│  2. Plan                                                 │
│     identify what to change, draft approach              │
│           │                                              │
│           v                                              │
│  3. Edit                                                 │
│     modify files (create, edit, delete)                  │
│           │                                              │
│           v                                              │
│  4. Test ←──────────────────────────┐                    │
│     run tests, linters, type checks │                    │
│           │                         │                    │
│           v                         │                    │
│  5. Pass? ──── No ──> Debug ────────┘                    │
│           │           read error, analyze,               │
│          Yes          fix, go to step 4                  │
│           │                                              │
│           v                                              │
│  6. Verify                                               │
│     broader test suite, manual check                     │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

A concrete example — fixing a bug with Claude Code:

```
User: "The CSV parser crashes on files with Unicode headers"

Step 1 (Understand):
  Thought: Let me find the CSV parser code and the failing test.
  Action: grep("csv.*pars", type="py")  → src/parser.py, tests/test_parser.py
  Action: read_file("tests/test_parser.py")  → see test_unicode_headers

Step 2 (Reproduce):
  Action: bash("python -m pytest tests/test_parser.py::test_unicode_headers -v")
  Observation: UnicodeDecodeError: 'ascii' codec can't decode byte 0xc3

Step 3 (Diagnose):
  Action: read_file("src/parser.py")
  Thought: Line 23 opens with open(path) — no encoding specified.
           Defaults to system locale, which is ASCII on this machine.

Step 4 (Fix):
  Action: edit_file("src/parser.py", old="open(path)", new="open(path, encoding='utf-8')")

Step 5 (Test):
  Action: bash("python -m pytest tests/test_parser.py -v")
  Observation: 12/12 tests pass ✓

Step 6 (Verify):
  Action: bash("python -m pytest --tb=short")
  Observation: 147/147 tests pass ✓

Done. The bug was a missing encoding parameter.
```

### SWE-bench Performance (March 2026)

Top results on SWE-bench Verified (500 curated real GitHub issues):

| Agent / System                     | Model                  | Score   |
|-----------------------------------|------------------------|---------|
| Claude Opus 4.6 (thinking) + scaffold | Claude Opus 4.6     | 79.2%   |
| GPT-5.4 + scaffold               | GPT-5.4                | 77.2%   |
| Gemini 3 Flash + scaffold        | Gemini 3 Flash         | 76.2%   |
| OpenHands CodeAct 2.1            | Claude Sonnet 4.5      | 72.0%   |
| Codex                            | GPT-5.3-Codex          | ~68%    |
| SWE-agent 1.0                    | Claude 3.5 Sonnet      | 57.6%   |
| Devin (launch)                   | Custom                 | 13.9%   |

The jump from Devin's 13.9% (Mar 2024) to 79.2% (Mar 2026) in two years reflects improvements in both models and scaffolding. The February 2026 SWE-bench scaffold upgrade (better environments, higher token limits) also boosted scores significantly.

## Agent Evaluation

### Benchmarks

| Benchmark     | Domain               | Tasks | Metric       | SOTA (Mar 2026)    |
|--------------|----------------------|-------|--------------|--------------------|
| SWE-bench Verified | Software engineering | 500 | % resolved | 79.2% (Opus 4.6)  |
| SWE-bench Live | Fresh GitHub issues | Rolling | % resolved | ~65%              |
| WebArena     | Web navigation       | 812   | Task success | 61.7% (IBM CUGA)  |
| WebChoreArena| Complex web tasks    | 532   | Task success | 37.8% (Gemini 2.5)|
| OSWorld      | Desktop OS tasks     | 369   | Task success | ~38% (Operator)   |
| GAIA         | General assistance   | 466   | Task success | ~75% (GPT-5)      |
| HumanEval    | Code generation      | 164   | pass@1       | 97%+              |

### Benchmarking Challenges

**Contamination**: Models may have seen benchmark solutions in training data. SWE-bench Live uses fresh issues to mitigate this. SWE-bench Verified manually audited all 500 tasks for clarity.

**Scaffold variance**: The same model can score very differently with different scaffolds. SWE-agent's custom ACI boosted results significantly over naive tool use. This makes it hard to isolate model capability from engineering.

**Overfitting to benchmarks**: Agents optimized for SWE-bench may not generalize to real-world software engineering, which involves understanding vague requirements, navigating large codebases, and communicating with humans.

**Cost normalization**: A system that achieves 70% by spending $50/task is very different from one that achieves 65% at $0.50/task. Most benchmarks don't normalize by cost, but production systems must.

**Reproducibility**: Agent behavior is stochastic. Temperature, sampling, and even API-side changes affect results. SWE-bench runs need multiple trials for reliable numbers.

## Training Agents

### SFT on Trajectories

The simplest approach: collect successful agent trajectories (from humans or a strong model) and fine-tune on them.

```
┌──────────────────────────────────────────────────────────┐
│            SFT ON AGENT TRAJECTORIES                      │
│                                                          │
│  1. Collect trajectories from expert agent (e.g., GPT-5) │
│     ┌────────────────────────────────────────────────┐   │
│     │ Task: "Fix the null pointer in UserService"    │   │
│     │ Step 1: grep("UserService", type="java")       │   │
│     │ Step 2: read("src/UserService.java")           │   │
│     │ Step 3: edit(line 42, add null check)           │   │
│     │ Step 4: bash("mvn test") → pass                │   │
│     │ Outcome: SUCCESS                                │   │
│     └────────────────────────────────────────────────┘   │
│                                                          │
│  2. Format as training data (mask non-assistant tokens)   │
│     ┌────────────────────────────────────────────────┐   │
│     │ system: tools + instructions     MASKED        │   │
│     │ user: task description           MASKED        │   │
│     │ assistant: thought + tool call   TRAIN ←       │   │
│     │ tool: result                     MASKED        │   │
│     │ assistant: thought + tool call   TRAIN ←       │   │
│     │ tool: result                     MASKED        │   │
│     │ assistant: final answer          TRAIN ←       │   │
│     └────────────────────────────────────────────────┘   │
│                                                          │
│  3. Fine-tune with standard SFT (same as your Dolly/     │
│     SlimOrca training, but on trajectory data)            │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

This is the same SFT you did with Dolly 15k and SlimOrca 520k — same loss function, same masking. The difference is the training data format: multi-turn trajectories with tool calls instead of single-turn instruction-response pairs.

**Limitation**: SFT learns to imitate the expert's trajectory but doesn't learn to recover from mistakes (because expert trajectories rarely contain mistakes). The model performs well on easy tasks but struggles when it encounters situations not covered by the training data.

### RL for Agents

Reinforcement learning addresses SFT's limitation by training the model to maximize task success through trial and error. The agent learns from its own experience, including failures.

**WebGPT** (2021) — The earliest example: GPT-3 fine-tuned with RL from human feedback to browse the web. The reward model scored the agent's answers for accuracy and citation quality.

**Agent Q** (Pang et al., 2024, [arxiv 2408.07199](https://arxiv.org/abs/2408.07199)) — Combined Monte Carlo Tree Search with self-critique and iterative fine-tuning. The agent explores multiple action sequences via MCTS, evaluates them with a self-critique mechanism, and uses DPO to update the policy. On WebShop: 0% → 50% improvement over baseline.

**WebAgent-R1** (May 2025, [arxiv 2505.16421](https://arxiv.org/abs/2505.16421)) — End-to-end multi-turn RL for web agents. Learns directly from online interactions using binary success/failure rewards. Boosted Qwen-2.5-3B from 6.1% → 33.9% and Llama-3.1-8B from 8.5% → 44.8% on WebArena-Lite. Outperformed o3.

**Agent-R1** (Nov 2025, [arxiv 2511.14460](https://arxiv.org/abs/2511.14460)) — Extended the MDP framework for LLM agents with a modular RL training framework. Demonstrated that GRPO-style RL (the same algorithm you studied) works for training agents — the reward signal comes from task success rather than human preference.

**ART (Agent Reinforcement Trainer)** (OpenPipe, 2025, [github](https://github.com/OpenPipe/ART)) — Open-source framework for training multi-step agents with GRPO on real-world tasks. Supports Qwen3.5, GPT-OSS, Llama. The key design: asynchronous trajectory generation with binary rewards.

### Reward Design for Multi-Step Tasks

The hardest part of RL for agents: defining what "good" means.

```
┌──────────────────────────────────────────────────────────┐
│              REWARD DESIGN SPECTRUM                        │
│                                                          │
│  Sparse (easy to define, hard to learn from):            │
│    reward = 1 if task_completed else 0                   │
│    Problem: agent gets 0 reward on most episodes,         │
│    no gradient signal to learn from                       │
│                                                          │
│  Dense (harder to define, easier to learn from):         │
│    reward = (                                            │
│      +0.1 for each relevant file read                    │
│      +0.2 for each test that newly passes                │
│      -0.1 for each tool call that errors                 │
│      -0.05 per step (efficiency penalty)                 │
│      +1.0 for full task completion                       │
│    )                                                     │
│    Problem: reward hacking — agent games the proxy        │
│    metrics instead of solving the actual task             │
│                                                          │
│  Process-based (best but expensive):                     │
│    Train a process reward model (PRM) that scores        │
│    each reasoning step, not just the outcome.            │
│    Requires step-level human annotations.                │
│    Your knowledge of PRMs from test-time compute         │
│    applies directly here.                                │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

The WebAgent-R1 result is notable: binary sparse rewards (success/fail) were sufficient to train effective agents when combined with enough trajectory diversity. This mirrors the DeepSeek-R1 finding — sometimes simple rewards work if you generate enough diverse experience.

### The Training Pipeline

```
┌──────────────────────────────────────────────────────────┐
│         AGENT TRAINING PIPELINE (2025-2026)                │
│                                                          │
│  Stage 1: SFT on expert trajectories                     │
│           (bootstrap basic tool-use ability)             │
│           ↓                                              │
│  Stage 2: RL with task rewards                           │
│           (learn to recover from errors,                 │
│            explore better strategies)                    │
│           ↓                                              │
│  Stage 3: (Optional) Iterative self-play                 │
│           (agent generates its own training data          │
│            by attempting tasks, filtered by success)     │
│                                                          │
│  This mirrors the standard LLM pipeline:                 │
│  Pretraining → SFT → RLHF/DPO/GRPO                     │
│  But here: Pretraining → Tool SFT → Agent RL             │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

## Failure Modes

### 1. Context Window Overflow

The most common failure for long-running agents. As the trajectory grows, the context window fills, and the agent either loses track of early observations or hits the limit entirely.

**Symptoms**: Agent repeats actions it already performed, forgets what it learned, starts contradicting earlier reasoning, or crashes with a context length error.

**Numerical example**: An agent debugging a complex issue reads 20 files (avg 5K tokens each) = 100K tokens of file content alone. Add tool calls, observations, and reasoning = 200K+ total. On a 200K context model, it's already full.

**Mitigations**: Context compaction (summarize old steps), sub-agent delegation (offload exploration to a fresh context), selective file reading (read only relevant sections, not whole files), and hard step limits.

### 2. Error Cascades

One mistake compounds through subsequent steps. The agent acts on wrong assumptions and each action moves further from the correct path.

```
Step 1: Agent misreads error message (mistake)
Step 2: Edits wrong file based on misunderstanding
Step 3: Tests fail (different error now)
Step 4: Agent tries to fix the new error (introduced by step 2)
Step 5: More edits, more breakage
...
Step 15: Codebase is now worse than when the agent started
```

**Production data**: Tool hallucination rates of 2-8% in production systems. Even a 2% error rate per step gives 0.98^50 = 36% chance of no errors over 50 steps.

**Mitigations**: Run tests after every edit (not just at the end), use git checkpoints to revert bad changes, implement "circuit breakers" that stop the agent after N consecutive failures.

### 3. Hallucinated Actions

The agent calls tools that don't exist, passes invalid arguments, or fabricates observation data in its reasoning.

```
Agent: "Let me check the database schema."
Action: query_database("SHOW TABLES")    ← tool doesn't exist
Error: Unknown tool: query_database

Agent: "The database has tables: users, orders, products"
                                           ← hallucinated from training data,
                                              not from any observation
```

**Mitigations**: Strict tool validation (reject unknown tools immediately), always ground reasoning in actual observations (never trust the model's "memory" of data it didn't actually read this session), and include tool names in the system prompt.

### 4. Infinite Loops

The agent gets stuck in a cycle — trying the same approach, getting the same error, retrying without changing anything meaningful.

```
Step 10: edit line 42 to fix TypeError
Step 11: run tests → TypeError on line 42
Step 12: edit line 42 (same fix) to fix TypeError
Step 13: run tests → TypeError on line 42
Step 14: edit line 42...
```

**Mitigations**: Track action history and detect repetition, hard step limits, escalation after N failures ("You've tried this 3 times. Try a completely different approach or ask for help.").

### 5. Cost Runaway

Every tool call = one or more LLM API calls. Complex tasks can consume thousands of dollars in API costs.

```
Typical cost per agent task (March 2026):
  Simple question answering:  $0.01 - $0.05
  Bug fix (5-10 steps):       $0.10 - $0.50
  Feature implementation:      $0.50 - $5.00
  Complex multi-file refactor: $2.00 - $20.00
  Failed task (100 steps, gives up): $5.00 - $50.00

The failure case is the expensive one. An agent that loops for
100 steps and fails costs 10-100x more than one that succeeds in 10 steps.
```

**Mitigations**: Per-task budgets (max tokens, max steps, max cost), monitoring and alerting, fast-fail heuristics (stop early if the approach isn't working), use cheaper models for exploration and expensive models for execution.

### 6. The "Bag of Agents" Trap

For multi-agent systems: adding more agents doesn't necessarily help. A 2025 study found that naive multi-agent systems produce 17x more errors than well-designed single-agent systems. Each agent handoff is a point of information loss and potential error injection.

**Mitigations**: Use multi-agent only when the task genuinely requires it (different expertise, true parallelism). Always verify at handoff points. Prefer a single capable agent over a committee of weaker ones.

## Practical Patterns

### When to Use Agents vs. Simple Prompting

```
┌──────────────────────────────────────────────────────────┐
│       DECISION FRAMEWORK: AGENT VS. PROMPT                │
│                                                          │
│  Use simple prompting when:                              │
│  ✓ Task can be answered in one LLM call                  │
│  ✓ No external data or tools needed                      │
│  ✓ Input/output is well-defined (classification, etc.)   │
│  ✓ Cost sensitivity is high                              │
│  ✓ Latency must be < 2 seconds                           │
│                                                          │
│  Use an agent when:                                      │
│  ✓ Task requires multiple steps with uncertain order     │
│  ✓ External tools are needed (code execution, search,    │
│    file system, APIs)                                    │
│  ✓ The solution path depends on intermediate results     │
│  ✓ Error recovery and iteration are expected             │
│  ✓ The task would take a human multiple steps too        │
│                                                          │
│  Rule of thumb: if you can write the solution as a       │
│  fixed script (no conditionals based on LLM output),     │
│  you don't need an agent. If the next step depends on    │
│  what the model finds, you need an agent.                │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### Cost Management

**Token budgets**: Set per-task limits. A coding task might get a budget of 500K input tokens and 50K output tokens. If the agent approaches the limit, it must produce a best-effort response.

**Model routing**: Use a cheap model (Claude Haiku, GPT-4o-mini) for exploration and information gathering, then route to an expensive model (Claude Opus, GPT-5) for critical reasoning and decision-making. Some scaffolds do this automatically based on task complexity.

**Caching**: Cache tool results. If the agent reads the same file twice, return the cached version. If it runs the same search, return cached results. This is trivially implemented and can save 20-40% of costs.

**Early termination**: If the agent is spinning (same error 3 times, no progress in 5 steps), stop it. A failed agent that stops early costs much less than one that grinds through 100 steps.

### Human-in-the-Loop Designs

```
┌──────────────────────────────────────────────────────────┐
│         HUMAN-IN-THE-LOOP PATTERNS                        │
│                                                          │
│  1. Approval gates (Devin, Claude Code)                  │
│     Agent proposes action → Human approves/rejects       │
│     Best for: destructive actions (delete, deploy, push) │
│                                                          │
│  2. Checkpoint review                                    │
│     Agent works autonomously for N steps → pauses        │
│     Human reviews progress → continues or redirects      │
│     Best for: long tasks where full autonomy is risky    │
│                                                          │
│  3. Escalation                                           │
│     Agent runs autonomously → detects it's stuck         │
│     → asks human a specific question → continues         │
│     Best for: tasks where the agent needs domain info    │
│                                                          │
│  4. Async review (Codex, Devin)                          │
│     Agent completes task fully → creates PR/artifact     │
│     Human reviews finished work, provides feedback       │
│     Best for: batch/background work                      │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

The trend in 2025-2026 is toward pattern 4 (async review): agents work in the background on multiple tasks in parallel, humans review outputs. This maximizes throughput — the human becomes a reviewer/architect rather than a step-by-step supervisor.

### Production Deployment Checklist

1. **Sandboxing**: Never let an agent run on your host machine. Use containers (Docker) or cloud sandboxes. Claude Code, Devin, OpenHands, and Codex all use sandboxed environments.

2. **Observability**: Log every tool call, observation, and decision. 89% of production agent teams have implemented observability (LangChain survey, 2025). Without it, debugging failures is nearly impossible.

3. **Guardrails**: Prevent the agent from accessing sensitive files, running dangerous commands, or making network requests to unauthorized endpoints. Cursor's sandbox reduces approval prompts by 40% while maintaining security.

4. **Idempotency**: If an agent task fails partway, you should be able to retry it safely. Use git branches for code changes, database transactions for data changes.

5. **Cost monitoring**: Set alerts for tasks exceeding cost thresholds. A runaway agent at 2am can burn through significant budget.

6. **Evaluation**: Don't just ship the agent — measure it. Track success rates, cost per task, user satisfaction, and failure mode distribution over time.

## Key Papers

| Paper | Authors | Year | Key Contribution |
|-------|---------|------|-----------------|
| [WebGPT](https://arxiv.org/abs/2112.09332) | Nakano et al. (OpenAI) | 2021 | First LLM trained to browse the web via structured actions |
| [ReAct](https://arxiv.org/abs/2210.03629) | Yao et al. (Princeton/Google) | 2022 | Thought-Action-Observation loop; backbone of all agent frameworks |
| [Toolformer](https://arxiv.org/abs/2302.04761) | Schick et al. (Meta) | 2023 | Self-supervised tool learning via perplexity filtering |
| [Reflexion](https://arxiv.org/abs/2303.11366) | Shinn et al. | 2023 | Verbal reinforcement learning; episodic memory from self-reflection |
| [Voyager](https://arxiv.org/abs/2305.16291) | Wang et al. (NVIDIA) | 2023 | First embodied lifelong learning agent; skill library pattern |
| [Tree of Thoughts](https://arxiv.org/abs/2305.10601) | Yao et al. (Princeton) | 2023 | Tree-structured deliberation with search for LLM planning |
| [LATS](https://arxiv.org/abs/2310.04406) | Zhou et al. (Princeton) | 2023 | Unified reasoning + acting + planning via MCTS with LLM |
| [SWE-agent](https://arxiv.org/abs/2405.15793) | Yang et al. (Princeton) | 2024 | Agent-computer interface design for software engineering |
| [OpenHands](https://arxiv.org/abs/2407.16741) | Wang et al. (UIUC) | 2024 | Open platform for AI software developers; CodeAct architecture |
| [Agent Q](https://arxiv.org/abs/2408.07199) | Pang et al. | 2024 | MCTS + self-critique + DPO for autonomous web agents |
| [WebAgent-R1](https://arxiv.org/abs/2505.16421) | Qi et al. | 2025 | End-to-end multi-turn RL for web agents; binary reward sufficient |
| [Agent-R1](https://arxiv.org/abs/2511.14460) | Yuan et al. | 2025 | MDP framework + modular RL training for LLM agents |
| [LLM Agent Survey](https://arxiv.org/abs/2503.21460) | Luo et al. | 2025 | Comprehensive survey: methodology, applications, challenges |

**Related work**: [Lilian Weng's "LLM Powered Autonomous Agents"](https://lilianweng.github.io/posts/2023-06-23-agent/) (Jun 2023) — The blog post that crystallized the field's thinking. Framed agents as: planning + memory + tool use. [Anthropic's "Building Effective Agents"](https://www.anthropic.com/engineering/building-effective-agents) (Dec 2024) — Practical guide from Claude's developers on when and how to build agents.
