# OpenAI《Research acceleration: The jagged frontier》读后笔记

2026-11-24

刚读完 OpenAI 11 月 18 日的研究加速报告。我最想继续想的是，agent 能接手越来越长的工作以后，研究员怎样消化这么多结果，以及模型什么时候能自己提出值得做的研究问题。 [OpenAI 原文](https://openai.com/index/research-acceleration-jagged-frontier/)

## 1. Agent 能独立完成更长的任务了

从 3 月到 10 月中旬，人工预计要做 2–4 小时的研究任务，agent 无需人工介入就成功的比例从 23% 升到 46%。人工预计要做 4–8 小时的任务，从 13% 升到 35%。同期委派的任务里，2–4 小时和 4–8 小时这两档的占比上升，15–30 分钟这档从约 23% 降到 17%。Agent 能完成更多长任务，研究员也愿意交给它更多长任务。同一个时长档里，大家挑选的任务也可能变了。 [任务时长与成功率](https://openai.com/index/research-acceleration-jagged-frontier/#measures-of-research-activity)

## 2. 一天的用量，以及谁来审阅结果

最让我焦虑的是资源差距。10 月，按 API 标价折算，OpenAI 研究员的 agent 日用量中位数是每人 2,400 美元，第 90 百分位是 23,000 美元。个人通常拿 100 或 200 美元的订阅用一个月，包月价格和实际用量之间也有很大差别。 [OpenAI 的内部用量](https://openai.com/index/research-acceleration-jagged-frontier/#internal-agent-usage)

**100 美元的 Pro 订阅如果每周接近用满，光 Codex 就粗估相当于每月 2,000 美元的 API 用量。** 一组公开日志里，10 月 29 日额度翻倍前，八个接近用满的周额度窗口按 API 标价平均约 237 美元。沿用类似的模型和缓存比例，按现在的两倍额度、每月 4.33 周折算，约 2,050 美元。这个估算按正常周刷新计算。按相同口径，OpenAI 研究员中位的一天略超过我这里估的重度个人一个月，第 90 百分位的一天约抵 11 个月。 [Codex 用量实测](https://finopsllm.com/research/codex-pro-100-30-days-14-resets) · [OpenAI 的额度变更](https://help.openai.com/en/articles/6825453-chatgpt-release-notes)

预训练和后训练团队平均每人每天约用 8 个 agent workdays。按 8 小时一个工作日算，就是累计约 64 小时的 agent 运行时间。10 月中旬，样本里超过三分之二的研究员日均并发峰值至少有 4 个 agent，超过 10% 的至少有 16 个。 [OpenAI 的内部用量](https://openai.com/index/research-acceleration-jagged-frontier/#internal-agent-usage)

我很想知道他们怎么审。就算 16 个 agent 各跑 4–8 小时，一天只各交一份结果，一个人也得面对 16 份。一位做后训练评估的研究员说，他会先让 agent 做一份，自己审阅修正，再让它批量做剩下的。这个办法能处理同类的工作。要是并行跑的是 16 条不同的研究线，怎么判断哪些结果值得细看？ [研究员访谈](https://openai.com/index/research-acceleration-jagged-frontier/#researchers-perspectives-on-ai-use)

## 3. 人的时间用在哪里

在 101 名研究人员的调查里，中位回答是，如果团队没有 coding agent，只能用 Copilot 加 ChatGPT，需要 3 个月才能做出最近 1 个月同等价值的工作。按他们自己的估计，用 agent 后的用时大约是原来的三分之一。 [研究员调查](https://openai.com/index/research-acceleration-jagged-frontier/#researchers-perspectives-on-ai-use)

与此同时，1 到 9 月，OpenAI 记录到的 Safety & Alignment 工作周数增加了 77%，其中有新招的人，也有从别处转来的研究员。7 月 Hugging Face 事件之后，这部分投入的增速明显上升。我想到一个外部研究者可能有的优势。做范围较小的研究，承担较少的前沿训练和部署责任，也许能腾出更多时间探索新问题。 [OpenAI 的研究人力分布](https://openai.com/index/research-acceleration-jagged-frontier/#where-researchers-allocate-their-time)

## 4. RSI 会卡在哪些任务上

按任务类别看，沟通的零干预成功率约 69%，分析 58%，构建 43%。具体差距更大。监控训练运行约 79%，启动训练约 35%，写推理和模型服务代码约 25%。搭建评估、强化学习和 agent 环境，以及建设训练框架基础设施，都在 29% 左右。按水桶理论看，RSI 能走多远可能取决于这些最弱的环节。这也是我觉得还有机会的地方。 [任务类别与成功率](https://openai.com/index/research-acceleration-jagged-frontier/#measures-of-research-activity)

设计和分析还有一个特点，就是研究员委派得很少。10 月 18 日的图里，每人每天估计委派约 0.5 项设计、1.1 项分析、13.2 项沟通任务。设计包括写技术规格和设计实验。分析包括评阅外部研究、分析产品用量、模型行为和实验结果。对我来说，这些比单纯的沟通更接近研究里要下判断的地方。 [任务委派分布](https://openai.com/index/research-acceleration-jagged-frontier/#measures-of-research-activity)

## 5. 好问题是怎么来的

回顾 GPT-6 Astra 和 GPT-6.2 Astra 时，受访的后训练和安全监控研究员说，他们列出的关键点子都来自人，模型帮忙实现。GPT-6.2 Astra 的预训练里，有人和 Codex 聊着聊着得到关键洞察，也有人先有自己的想法，后来 Codex 建议的一处改动才让它真正奏效。 [关键点子的来源](https://openai.com/index/research-acceleration-jagged-frontier/#researchers-perspectives-on-ai-use)

最让我在意的是一位做强化学习环境的研究员说，模型很难一次就提出一个定义清楚、自己却解不了的问题。让它生成一百次数据点，产出的东西又常常很像。这两点连在一起，是我觉得现在 RSI 最大的痛点之一。能不能让模型持续提出不同的、定义清楚的研究问题，包括它现在自己还解决不了的问题？我甚至会把这件事放到接下来最优先研究的位置。 [研究员访谈](https://openai.com/index/research-acceleration-jagged-frontier/#researchers-perspectives-on-ai-use)
