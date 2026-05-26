# Tool Use and Function Calling in LLMs

A step-by-step guide to how LLMs learn to use external tools. Covers the research lineage, training methods, and practical implementation.

## Background

**Research lineage** — Tool use in LLMs evolved from prompting tricks to native model capabilities:

1. **WebGPT** (Nakano et al., OpenAI, Dec 2021, [arxiv 2112.09332](https://arxiv.org/abs/2112.09332)) — Fine-tuned GPT-3 to browse the web via a text-based browser environment. The model learned special actions (search, click, scroll, quote) through imitation learning on human demonstrations, then was optimized with human feedback. First major demonstration that LLMs could learn to interact with external systems through structured actions.

2. **ReAct** (Yao et al., Princeton/Google, Oct 2022, [arxiv 2210.03629](https://arxiv.org/abs/2210.03629)) — Introduced the Thought-Action-Observation loop. Instead of training, ReAct used few-shot prompting to get LLMs to interleave reasoning traces with tool actions. Published at ICLR 2023. Showed that even without fine-tuning, LLMs could use tools effectively if given the right prompting structure.

3. **Toolformer** (Schick et al., Meta, Feb 2023, [arxiv 2302.04761](https://arxiv.org/abs/2302.04761)) — The first self-supervised approach to tool learning. The model taught itself when and how to insert API calls by filtering for calls that reduced perplexity. No human demonstrations of tool use needed. Published at NeurIPS 2023.

4. **OpenAI Function Calling** (June 2023) — GPT-3.5-turbo and GPT-4 gained native function calling. Models were fine-tuned to detect when a function should be called and output structured JSON with function name and arguments. Made tool use a first-class API feature rather than a prompting hack.

5. **Gorilla** (Patil et al., UC Berkeley, May 2023, [arxiv 2305.15334](https://arxiv.org/abs/2305.15334)) — Fine-tuned LLaMA on 11,000+ instruction-API pairs. Surpassed GPT-4 on API call accuracy. Demonstrated that SFT on tool-use data could create specialized tool-calling models. Published at NeurIPS 2024.

6. **Claude Tool Use, Gemini Function Calling** (2023-2024) — Anthropic, Google, and others added native tool use to their models, converging on a standard: tools defined via JSON Schema, model outputs structured calls, system executes and feeds results back.

**The arc**: prompting (ReAct) → self-supervised learning (Toolformer) → supervised fine-tuning on curated data (Gorilla, GPT-4, Claude) → native model capability with structured APIs.

## What Problem Does Tool Use Solve?

LLMs have fundamental limitations that no amount of scaling fixes:

- **No real-time information**: Training data has a cutoff. The model can't tell you today's weather or stock price.
- **Imprecise computation**: LLMs approximate math via token prediction. "What is 1847 * 3921?" is unreliable. A calculator is exact.
- **No external interaction**: Can't send emails, query databases, call APIs, or modify files.
- **No private data access**: Can't look up your calendar, your company's internal docs, or a customer's order status.

Tool use bridges this gap. Instead of trying to do everything with next-token prediction, the model learns to **delegate** to specialized systems:

```
Without tools:  User asks "What's AAPL stock price?" → Model guesses or says "I don't know"
With tools:     User asks "What's AAPL stock price?" → Model calls get_stock_price("AAPL")
                → System returns $187.42 → Model says "Apple's current stock price is $187.42"
```

The model becomes an **orchestrator** — it understands what the user needs, picks the right tool, formats the call correctly, and synthesizes the result into a natural response.

## Key Terms

**Tool / Function**: An external capability the model can invoke. Defined by a name, description, and parameter schema. Examples: `get_weather(city, unit)`, `search_web(query)`, `run_sql(query)`.

**Tool Schema (JSON Schema)**: A structured definition of what a tool does and what arguments it accepts. This is what the model sees in its context:

```json
{
  "name": "get_weather",
  "description": "Get current weather for a city",
  "parameters": {
    "type": "object",
    "properties": {
      "city": {"type": "string", "description": "City name"},
      "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
    },
    "required": ["city"]
  }
}
```

**Function Calling / Tool Calling**: The model's act of generating a structured output that specifies which tool to call and with what arguments. The model doesn't execute anything — it produces JSON that the system parses and executes.

**Tool Invocation**: The system (not the model) executing the actual function call based on the model's structured output.

**Tool Result Injection**: Feeding the tool's output back into the model's context as a new message, so the model can incorporate the result into its response.

**Agentic Loop**: A cycle where the model repeatedly reasons, calls tools, observes results, and decides whether to call more tools or produce a final answer. This enables multi-step problem solving.

## How Tool Use Works End-to-End

The complete cycle from user query to final response:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TOOL USE CYCLE                               │
│                                                                     │
│  ┌──────────┐    ┌──────────────────┐    ┌───────────────────────┐  │
│  │          │    │                  │    │                       │  │
│  │   User   │───>│   LLM decides:   │───>│  Generate structured  │  │
│  │  Query   │    │  "I need a tool" │    │  tool call (JSON)     │  │
│  │          │    │                  │    │                       │  │
│  └──────────┘    └──────────────────┘    └───────────┬───────────┘  │
│                                                      │              │
│                                                      v              │
│  ┌──────────┐    ┌──────────────────┐    ┌───────────────────────┐  │
│  │          │    │                  │    │                       │  │
│  │  Final   │<───│  LLM synthesizes │<───│  System executes      │  │
│  │ Response │    │  tool result     │    │  tool, returns result │  │
│  │          │    │                  │    │                       │  │
│  └──────────┘    └──────────────────┘    └───────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Concrete Example: Walking Through Each Step

**User query**: "What's the weather in Tokyo and should I bring an umbrella?"

**Step 1 — System prompt with tool definitions**:

The model's context includes tool schemas before the conversation starts:

```
System: You have access to the following tools:

1. get_weather(city: string, unit: string) - Get current weather for a city
2. get_forecast(city: string, days: int) - Get weather forecast

When you need to use a tool, output a JSON tool call.
```

**Step 2 — Model decides to call a tool**:

The model processes the user query and its context (including tool definitions). Instead of generating a text response, it generates a structured tool call:

```json
{
  "tool_calls": [
    {
      "id": "call_001",
      "function": {
        "name": "get_weather",
        "arguments": "{\"city\": \"Tokyo\", \"unit\": \"celsius\"}"
      }
    }
  ]
}
```

The model chose `get_weather` over `get_forecast` because the user asked about current conditions. It inferred "celsius" because Tokyo is in Japan.

**Step 3 — System executes the tool**:

The application layer (not the model) parses the JSON, validates the arguments, and calls the actual function:

```python
result = get_weather(city="Tokyo", unit="celsius")
# Returns: {"temperature": 18, "condition": "light rain", "humidity": 78}
```

**Step 4 — Result injected back into context**:

The tool result is added as a new message with role "tool":

```json
{
  "role": "tool",
  "tool_call_id": "call_001",
  "content": "{\"temperature\": 18, \"condition\": \"light rain\", \"humidity\": 78}"
}
```

**Step 5 — Model generates final response**:

With the tool result in context, the model now has factual data to work with:

> "It's currently 18°C in Tokyo with light rain and 78% humidity. Yes, you should definitely bring an umbrella — it's actively raining right now."

### What the Message History Looks Like

The full conversation from the API's perspective:

```python
messages = [
    # 1. System prompt with tool definitions
    {"role": "system", "content": "You have access to: get_weather, get_forecast..."},

    # 2. User's question
    {"role": "user", "content": "What's the weather in Tokyo and should I bring an umbrella?"},

    # 3. Assistant's tool call (model output)
    {"role": "assistant", "tool_calls": [
        {"id": "call_001", "function": {"name": "get_weather",
         "arguments": "{\"city\": \"Tokyo\", \"unit\": \"celsius\"}"}}
    ]},

    # 4. Tool result (injected by system)
    {"role": "tool", "tool_call_id": "call_001",
     "content": "{\"temperature\": 18, \"condition\": \"light rain\", \"humidity\": 78}"},

    # 5. Final response (model output)
    {"role": "assistant", "content": "It's currently 18°C in Tokyo with light rain..."}
]
```

Messages 1, 2, and 4 are inputs to the model. Messages 3 and 5 are model outputs. The model generates message 3, the system handles execution, then the model generates message 5 in a second forward pass.

## Training for Tool Use

Models learn tool use through supervised fine-tuning (SFT) on conversations that contain tool calls. This is the same SFT process you've done before — the model learns to predict the next token — but the training data now includes structured tool call tokens alongside natural language.

### Training Data Format

Each training example is a multi-turn conversation with tool calls embedded:

```python
# One training example — a full tool-use conversation
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant with access to the following tools:\n\n"
                 "[{\"name\": \"get_weather\", \"description\": \"Get current weather\", "
                 "\"parameters\": {\"type\": \"object\", \"properties\": "
                 "{\"city\": {\"type\": \"string\"}, "
                 "\"unit\": {\"type\": \"string\", \"enum\": [\"celsius\", \"fahrenheit\"]}}, "
                 "\"required\": [\"city\"]}}]"
    },
    {
      "role": "user",
      "content": "What's the weather like in Paris?"
    },
    {
      "role": "assistant",
      "tool_calls": [
        {
          "id": "call_abc",
          "type": "function",
          "function": {
            "name": "get_weather",
            "arguments": "{\"city\": \"Paris\", \"unit\": \"celsius\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_abc",
      "content": "{\"temperature\": 22, \"condition\": \"sunny\"}"
    },
    {
      "role": "assistant",
      "content": "It's 22°C and sunny in Paris right now — a beautiful day!"
    }
  ]
}
```

### What the Model Learns

During SFT, the model learns to predict all assistant tokens (both tool calls and natural responses) while the system, user, and tool messages are masked (loss not computed on them):

```
Tokens:                      Loss computed?
─────────────────────────    ─────────────
system: "You have tools..."  NO  (input context)
user: "Weather in Paris?"    NO  (input context)
assistant: <tool_call>       YES (model must learn to produce this)
  get_weather(Paris, C)      YES
</tool_call>                 YES
tool: {temp: 22, sunny}      NO  (injected result — input context)
assistant: "It's 22°C..."    YES (model must learn to synthesize result)
```

This is identical to how SFT works for regular instruction following — the model learns to generate the right output given the context. The only difference is that "right output" now sometimes means a structured tool call instead of natural text.

### What the Model Must Learn

Through SFT on thousands of these examples, the model learns several capabilities simultaneously:

1. **Tool selection**: Given a user query and available tools, pick the right one (or none)
2. **Argument extraction**: Parse the user's intent into the correct JSON arguments
3. **Format compliance**: Generate syntactically valid JSON matching the schema
4. **Result synthesis**: Incorporate tool results into natural, helpful responses
5. **Abstention**: When no tool is needed, just respond normally

### Scale of Training Data

- **Gorilla** (2023): 11,000+ instruction-API pairs across 1,600+ APIs
- **ToolACE** (2024): 26,507 diverse APIs with synthetic instruction-API pairs
- **Production models** (GPT-4, Claude): Likely trained on hundreds of thousands of tool-use conversations, both human-written and synthetically generated

## The Toolformer Approach

Toolformer (Schick et al., 2023) took a fundamentally different approach: instead of curating training data with human-labeled tool calls, the model **taught itself** when and how to use tools.

### The Problem with Supervised Tool Data

Manually annotating "here is where you should call a calculator" across thousands of texts is expensive and brittle. Toolformer's insight: the model itself can figure out where tool calls would be helpful, using a simple criterion — **does the tool call help predict future tokens?**

### The Toolformer Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                   TOOLFORMER PIPELINE                        │
│                                                             │
│  Step 1: Sample      Step 2: Execute     Step 3: Filter     │
│                                                             │
│  For each text       Call the actual     Keep only calls     │
│  position, sample    APIs to get         where the result    │
│  candidate API       results             REDUCES loss on     │
│  calls                                   future tokens       │
│                                                             │
│  "The pop of        Calculator(          Loss with result:   │
│   Toronto is         pop_toronto)         1.2                │
│   [CALC]2.8M         → 2,794,356        Loss without:       │
│   people"                                 3.8                │
│                                          Δ = -2.6 → KEEP    │
│                                                             │
│  Step 4: Fine-tune on filtered data                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Step 1 — Sample candidate positions**: For each position in a text sequence, compute the probability that the model would start an API call (using a few-shot prompt with API call examples). Positions above a threshold are kept. For each position, sample up to *m* different API calls.

**Step 2 — Execute the API calls**: Actually run each sampled call (calculator, search engine, Q&A system, etc.) and record the result.

**Step 3 — Filter by perplexity reduction**: This is the key step. For each candidate API call, compute two losses on the tokens following the call:

```
L_with_result    = cross-entropy loss on future tokens, given the API call + result in context
L_without_result = cross-entropy loss on future tokens, without the API call

Keep the call if: L_without_result - L_with_result > threshold
```

If inserting `[Calculator(18*7) → 126]` before "... equals 126" reduces the model's loss on predicting "126", the call is kept. If the tool result doesn't help predict future tokens, it's discarded.

**Step 4 — Fine-tune**: The model is fine-tuned on the original texts augmented with the filtered API calls. The API calls are represented as special token sequences inline in the text.

### Toolformer's Inline Format

Unlike modern function-calling APIs, Toolformer embeds tool calls directly in the text stream:

```
The population of Toronto is [QA("What is the population of Toronto?") → 2,794,356] around 2.8 million.
```

The model learns to insert `[API_NAME(args) → result]` at positions where the tool helps. During inference, the model generates up to the `→` token, the system executes the API call, injects the result, and generation continues.

### Why Toolformer Matters

Toolformer showed that tool use doesn't require explicit human supervision. The model identified, from its own loss signal, that a calculator helps with arithmetic and a search engine helps with factual questions. This self-supervised approach inspired subsequent work, though production systems ultimately converged on curated SFT data for reliability.

## Function Calling Architecture

How modern models (GPT-4, Claude, Llama, Mistral) implement tool calling in practice.

### Special Tokens

Models use special tokens to delimit tool calls, separating them from natural language output. Different model families use different conventions:

```
ChatML / Hermes style:
<tool_call>
{"name": "get_weather", "arguments": {"city": "Tokyo"}}
</tool_call>

Llama 3 style:
<|python_tag|>get_weather.call(city="Tokyo")

Mistral style:
[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Tokyo"}}]

OpenAI (internal):
Uses dedicated fields in the API response object — the model outputs tokens that
the API layer parses into the structured tool_calls field.
```

These special tokens are added to the tokenizer's vocabulary (like `<|endoftext|>` or `<|im_start|>`) and the model learns their meaning through SFT.

### JSON Schema in System Prompt

Tool definitions are injected into the system prompt as JSON Schema. The model has been trained to understand this format and map user queries to the correct tool:

```
System prompt (what the model sees):

You are a helpful assistant. You have access to the following tools:

[
  {
    "type": "function",
    "function": {
      "name": "search_web",
      "description": "Search the web for current information",
      "parameters": {
        "type": "object",
        "properties": {
          "query": {"type": "string", "description": "Search query"},
          "num_results": {"type": "integer", "default": 5}
        },
        "required": ["query"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "run_python",
      "description": "Execute Python code and return the output",
      "parameters": {
        "type": "object",
        "properties": {
          "code": {"type": "string", "description": "Python code to execute"}
        },
        "required": ["code"]
      }
    }
  }
]

When you need to use a tool, respond with a tool call.
When you have enough information, respond directly.
```

The tool definitions count as input tokens — more tools means less context for the conversation. This is why tool descriptions should be concise.

### Parsing and Validation

After the model generates a tool call, the system must:

1. **Parse**: Extract the JSON from the model's output (between special tokens)
2. **Validate**: Check that the function name matches a defined tool and arguments match the schema (correct types, required fields present, enum values valid)
3. **Execute**: Call the actual function with the validated arguments
4. **Handle errors**: If parsing fails or arguments are invalid, feed an error message back to the model

```python
# Simplified tool execution loop
def execute_tool_call(tool_call, available_tools):
    name = tool_call["function"]["name"]
    args = json.loads(tool_call["function"]["arguments"])

    if name not in available_tools:
        return {"error": f"Unknown tool: {name}"}

    schema = available_tools[name]["parameters"]
    if not validate_json_schema(args, schema):
        return {"error": f"Invalid arguments for {name}"}

    result = available_tools[name]["execute"](**args)
    return result
```

### Parallel Tool Calling

Modern models can generate multiple tool calls in a single response, enabling concurrent execution:

```python
# Model output with parallel tool calls
{
  "tool_calls": [
    {"id": "call_1", "function": {"name": "get_weather",
     "arguments": "{\"city\": \"Tokyo\"}"}},
    {"id": "call_2", "function": {"name": "get_weather",
     "arguments": "{\"city\": \"London\"}"}},
    {"id": "call_3", "function": {"name": "get_exchange_rate",
     "arguments": "{\"from\": \"JPY\", \"to\": \"GBP\"}"}}
  ]
}
```

All three calls execute in parallel. Each result is injected as a separate tool message with the matching `tool_call_id`. The model then synthesizes all results into a single response.

The model learns to emit parallel calls when the tools are independent (no data dependency between them). When calls depend on each other (e.g., search for a restaurant, then get directions to it), the model emits them sequentially across multiple turns.

## ReAct Framework

ReAct (Reasoning + Acting) formalized the pattern of interleaving internal reasoning with external actions. Published by Yao et al. (2022) at ICLR 2023.

### The Thought-Action-Observation Loop

```
┌──────────────────────────────────────────────────────────┐
│                     ReAct LOOP                            │
│                                                          │
│    ┌─────────┐     ┌─────────┐     ┌─────────────────┐  │
│    │         │     │         │     │                 │  │
│ ──>│ Thought │────>│ Action  │────>│  Observation    │──┤
│ │  │(reason) │     │(tool    │     │  (tool result)  │  │
│ │  │         │     │ call)   │     │                 │  │
│ │  └─────────┘     └─────────┘     └─────────────────┘  │
│ │                                                        │
│ └──────────── repeat until done ─────────────────────────│
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### Concrete Example

**Question**: "Who was the president of the US when the iPhone was released?"

```
Thought 1: I need to find when the iPhone was released, then find who was
           president at that time. Let me search for the iPhone release date.

Action 1:  search("iPhone first release date")

Observation 1: The first iPhone was released on June 29, 2007.

Thought 2: The iPhone was released in June 2007. I need to find who was the
           US president in 2007.

Action 2:  search("US president 2007")

Observation 2: George W. Bush was the 43rd president, serving from 2001 to 2009.

Thought 3: I now have both pieces of information. George W. Bush was president
           when the iPhone was released in 2007.

Final Answer: George W. Bush was the president of the United States when the
             first iPhone was released on June 29, 2007.
```

### Why "Thought" Matters

Without reasoning traces, the model might search for "president iPhone" and get confused. The explicit thought step lets the model:

- **Decompose** multi-step problems into sub-questions
- **Track** what information it has vs. still needs
- **Recover** from errors ("that search didn't help, let me try a different query")
- **Decide** when it has enough information to answer

ReAct was originally implemented via few-shot prompting (no fine-tuning). You give the model 2-3 examples of the Thought/Action/Observation format, and it follows the pattern. Modern models have internalized this pattern through SFT on thousands of ReAct-style conversations.

### ReAct vs Chain-of-Thought

```
Chain-of-Thought (CoT):  Reason → Reason → Reason → Answer
                         (pure thinking, no external data)

ReAct:                   Think → Act → Observe → Think → Act → Observe → Answer
                         (interleaves reasoning with real-world data)
```

CoT fails when the model needs facts it doesn't have. ReAct bridges internal reasoning with external information retrieval.

## Agentic Patterns

Tool use enables **agents** — systems that autonomously loop through reasoning and acting to accomplish goals. This goes beyond single tool calls to multi-step workflows.

### The Basic Agent Loop

```python
def agent_loop(user_query, tools, max_steps=10):
    messages = [
        {"role": "system", "content": f"You have tools: {tools}. "
         "Use them to answer the user's question. "
         "When done, respond with your final answer."},
        {"role": "user", "content": user_query}
    ]

    for step in range(max_steps):
        response = llm(messages)

        if response.has_tool_calls:
            # Model wants to use a tool
            messages.append(response.message)
            for tool_call in response.tool_calls:
                result = execute(tool_call)
                messages.append({"role": "tool",
                                 "tool_call_id": tool_call.id,
                                 "content": str(result)})
        else:
            # Model produced a final text response
            return response.content

    return "Max steps reached"
```

The model is in control — it decides when to call tools and when to stop. The system just executes what the model asks for.

### Multi-Step Tool Chains

Real tasks often require sequential tool calls where each step depends on the previous:

```
User: "Find the top Hacker News story today and summarize the linked article."

Step 1 → Tool call: get_hacker_news_top_stories(limit=1)
         Result: {"title": "New Rust compiler...", "url": "https://blog.rust-lang.org/..."}

Step 2 → Tool call: fetch_webpage(url="https://blog.rust-lang.org/...")
         Result: {"content": "Today we're announcing... [full article text]"}

Step 3 → Model synthesizes: "The top Hacker News story today is about..."
```

The model couldn't know the URL for step 2 until step 1 completed. This sequential dependency is why agentic loops need multiple LLM calls.

### Common Agentic Patterns

**Tool fan-out**: Call multiple independent tools, then synthesize:
```
User: "Compare weather in NYC, London, and Tokyo"
→ 3 parallel get_weather calls → single comparative response
```

**Iterative refinement**: Use results to decide the next action:
```
User: "Find a Python bug in my code"
→ read_file → identify suspicious function → run_tests → read error → suggest fix
```

**Error recovery**: Handle tool failures gracefully:
```
→ search("obscure topic") → no results
→ Model thinks: "Let me try a broader query"
→ search("related broader topic") → useful results
```

## Practical Implementation: Adding Tool Use via SFT

If you have a fine-tuned model (e.g., your GPT-2 124M after SFT) and want to add tool-use capabilities, the process is straightforward SFT on tool-use conversations.

### Step 1 — Choose a Tool Call Format

Define the special tokens and format your model will use:

```python
# Add special tokens to tokenizer
special_tokens = ["<tool_call>", "</tool_call>", "<tool_result>", "</tool_result>"]
tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
model.resize_token_embeddings(len(tokenizer))
```

### Step 2 — Create Training Data

Build conversations that include tool calls. You can generate these synthetically:

```python
training_example = {
    "messages": [
        {"role": "system", "content": (
            "You have access to these tools:\n"
            "- calculator(expression: str) → computes math\n"
            "- search(query: str) → searches the web\n"
            "Use <tool_call>...</tool_call> to call a tool."
        )},
        {"role": "user", "content": "What is 847 * 29?"},
        {"role": "assistant", "content": (
            '<tool_call>{"name": "calculator", '
            '"arguments": {"expression": "847 * 29"}}</tool_call>'
        )},
        {"role": "tool", "content": (
            '<tool_result>24563</tool_result>'
        )},
        {"role": "assistant", "content": "847 × 29 = 24,563."},
    ]
}
```

Include a mix of:
- Conversations that require tool calls (teaches when to use tools)
- Conversations that don't require tools (teaches when NOT to use tools)
- Multi-tool conversations (teaches sequential tool use)
- Conversations where the model should refuse or ask for clarification

### Step 3 — SFT Training

Standard SFT — compute loss only on assistant tokens:

```python
# Tokenize the conversation
input_ids = tokenize_conversation(training_example)

# Create labels — mask everything except assistant turns
labels = input_ids.clone()
for span in non_assistant_spans:
    labels[span.start:span.end] = -100  # ignored by cross-entropy

# Forward pass + loss (same as your existing SFT code)
outputs = model(input_ids, labels=labels)
loss = outputs.loss
loss.backward()
optimizer.step()
```

The model learns to generate `<tool_call>{"name": "calculator", ...}</tool_call>` when the context calls for it, through the same next-token prediction objective.

### Step 4 — Inference Loop

At inference time, implement the agentic loop:

```python
def inference_with_tools(model, tokenizer, user_query, tools):
    prompt = format_system_prompt(tools) + format_user_message(user_query)
    input_ids = tokenizer.encode(prompt)

    while True:
        output_ids = model.generate(input_ids, max_new_tokens=256,
                                     stop_strings=["</tool_call>", tokenizer.eos_token])
        output_text = tokenizer.decode(output_ids)

        if "<tool_call>" in output_text:
            # Parse and execute the tool call
            tool_call_json = extract_between(output_text, "<tool_call>", "</tool_call>")
            result = execute_tool(json.loads(tool_call_json))

            # Append result and continue generation
            input_ids = tokenizer.encode(
                output_text + f"<tool_result>{result}</tool_result>"
            )
        else:
            # No tool call — this is the final response
            return output_text
```

## Challenges

### Hallucinated Tool Calls

The model calls a function that doesn't exist, or invents arguments that don't match the schema.

```
Available tool:  get_weather(city: str)
Model generates: get_weather(city="Paris", include_forecast=True, language="fr")
                                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                          These arguments don't exist in the schema
```

**Mitigation**: Strict JSON Schema validation before execution. Return a clear error message to the model so it can retry. SFT on diverse tool schemas helps the model generalize.

### Incorrect Argument Values

The model calls the right tool with the wrong arguments — confusing entities, using wrong types, or misinterpreting the user's request.

```
User: "What's the weather where I am?"
Model: get_weather(city="where I am")   # Passes the phrase literally
```

**Mitigation**: Better tool descriptions ("city must be a real city name"), and training data that includes clarification examples.

### Over-Eager Tool Use

The model calls tools when it doesn't need to. "What is 2 + 2?" doesn't need a calculator.

```
User: "What is 2 + 2?"
Bad:  <tool_call>calculator("2+2")</tool_call>
Good: "2 + 2 = 4."
```

**Mitigation**: Training data must include examples where the model answers directly without tools. The model needs negative examples — questions it could answer by calling a tool but shouldn't.

### Knowing When to Stop

In agentic loops, the model might call tools indefinitely, never producing a final answer. Or it might stop too early, before gathering enough information.

**Mitigation**: Maximum step limits, explicit instructions in the system prompt ("when you have enough information, respond directly"), and training data demonstrating appropriate stopping behavior.

### Security Considerations

Tool use introduces real-world side effects. A model with access to `send_email()` or `delete_file()` can cause damage.

- **Prompt injection**: A malicious webpage fetched by a search tool could contain instructions that trick the model into calling dangerous tools
- **Argument injection**: SQL injection via `run_sql(query="DROP TABLE users")` if the tool doesn't sanitize
- **Excessive permissions**: Giving the model access to tools it doesn't need for the task

**Mitigation**: Principle of least privilege (only expose needed tools), human-in-the-loop confirmation for destructive actions, sandboxed execution environments, input sanitization in tool implementations.

### Latency

Each tool call adds a round trip: LLM inference → tool execution → LLM inference. Multi-step agentic tasks can take 10-30 seconds, far longer than a single response.

**Mitigation**: Parallel tool calling, caching, faster tools, and minimizing the number of steps through better tool design (one powerful tool vs. many small ones).

## Key Papers

| Paper | Authors | Year | Key Contribution |
|-------|---------|------|-----------------|
| [WebGPT](https://arxiv.org/abs/2112.09332) | Nakano et al. (OpenAI) | 2021 | First LLM trained to browse the web via actions |
| [ReAct](https://arxiv.org/abs/2210.03629) | Yao et al. (Princeton/Google) | 2022 | Thought-Action-Observation loop for interleaving reasoning and tool use |
| [Toolformer](https://arxiv.org/abs/2302.04761) | Schick et al. (Meta) | 2023 | Self-supervised tool learning via perplexity filtering |
| [Gorilla](https://arxiv.org/abs/2305.15334) | Patil et al. (UC Berkeley) | 2023 | SFT on 11K+ API pairs; surpassed GPT-4 on API accuracy |
| [ToolACE](https://arxiv.org/abs/2409.00920) | Liu et al. | 2024 | Automated pipeline generating tool-use training data at scale (26K+ APIs) |
| [Tool Zero](https://arxiv.org/abs/2511.01934) | Zeng et al. | 2025 | Training tool-augmented LLMs via pure RL from scratch (no SFT) |

**Related work**: [ART](https://arxiv.org/abs/2303.09014) (automatic reasoning and tool-use), [TaskMatrix.AI](https://arxiv.org/abs/2303.16434) (connecting foundation models with APIs), [HuggingGPT](https://arxiv.org/abs/2303.17580) (LLM as controller for specialized models).
