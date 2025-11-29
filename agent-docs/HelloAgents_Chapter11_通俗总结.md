# Hello Agents 第十一章：Agentic RL（通俗总结）

> **本章核心思想**：如果说 Prompt Engineering 是"教"模型做事（给它指令），那么 Agentic RL 就是让模型在"试错"中自己学会做事。我们不再只关注"下一个词预测得准不准"，而是关注"任务完成得好不好"。

---

## 📖 目录

- [1. 从 LLM 到 Agentic RL：思维的转变](#1-从-llm-到-agentic-rl思维的转变)
- [2. LLM 训练全景图](#2-llm-训练全景图)
- [3. 数据集与奖励：告诉模型什么是"好"](#3-数据集与奖励告诉模型什么是好)
- [4. SFT 训练：先学会"走路"](#4-sft-训练先学会走路)
- [5. GRPO 训练：在奔跑中"���我进化"](#5-grpo-训练在奔跑中自我进化)
- [6. 本章总结](#6-本章总结)

---

## 1. 从 LLM 到 Agentic RL：思维的转变

### 🤔 传统 LLM 的局限

现在的 LLM（如 ChatGPT）很强，但它们本质上是"预测下一个词"的机器。
*   **被动**：只能根据你的 Prompt 回答，缺乏自主探索。
*   **短视**：只关注当前的回答好不好，不关心长远任务是否完成。
*   **模仿**：SFT（监督微调）只是模仿人类的数据，很难超越人类。

### 💡 Agentic RL 的核心

Agentic RL (基于强化学习的智能体训练) 把 LLM 放入一个"环境"中，让它像玩游戏一样去完成任务。
*   **多步决策**：为了完成任务，模型可能需要查资料、写代码、运行代码、修正错误。
*   **延迟奖励**：中间步骤可能没有奖励，只有最终任务完成了才有糖吃。
*   **自我进化**：通过试错，发现人类教不了的新策略（比如更聪明的解题路径）。

> **比喻**：
> *   **Pre-training**：在学校死记硬背书本知识。
> *   **SFT**：老师手把手教你做题（模仿）。
> *   **Agentic RL**：把你扔进社会，在实战中摸爬滚打，自己总结经验（试错与进化）。

---

## 2. LLM 训练全景图

训练一个强大的 Agent��通常要经历三个阶段：

1.  **预训练 (Pre-training)** 📚
    *   **目标**：学会语言和世界知识。
    *   **方法**：海量文本自监督学习（Next Token Prediction）。
    *   **产出**：Base Model（懂语言，但不听话）。

2.  **监督微调 (SFT)** 👩‍🏫
    *   **目标**：学会听指令，学会对话格式。
    *   **方法**：高质量问答对（Prompt-Completion）。
    *   **产出**：Chat Model（听话，但能力有限）。

3.  **强化学习 (RLHF / RLAIF)** 🚀
    *   **目标**：对齐人类价值观，提升复杂任务能力。
    *   **方法**：PPO, GRPO, DPO 等算法。
    *   **产出**：Agent Model（聪明，能自主解决问题）。

---

## 3. 数据集与奖励：告诉模型什么是"好"

在 RL 中，最重要的就是**奖励函数 (Reward Function)**。模型是"趋利"的，你奖励什么，它就学什么。

### 📊 GSM8K 数据集

我们以数学推理任务为例，使用经典的 **GSM8K** 数据集。
*   **SFT 格式**：包含完整的解题步骤（Step-by-step）。
*   **RL 格式**：只包含问题和最终答案。模型必须自己想出中间步骤。

### 🏆 奖励函数的设计

HelloAgents 提供了三种内置奖励，可以组合使用：

1.  **准确率奖励 (AccuracyReward)**：
    *   答案对不对？对得 1 分，错得 0 分。
    *   *最基础，但太稀疏（只有最后一步才有反馈）。*

2.  **长度惩罚 (LengthPenalty)**：
    *   答案是否太啰嗦？越啰嗦分越低。
    *   *鼓励模型简洁高效。*

3.  **步骤奖励 (StepReward)**：
    *   有没有清晰的推理步骤（Step 1, Step 2...）？
    *   *鼓励模型展示思考过程（CoT），而不是瞎猜。*

---

## 4. SFT 训练：先学会"走路"

在让模型自我进化之前，必须先让它学会基本的"走路"姿势（输出格式、基本逻辑）。这就是 SFT 的作用。

### 🛠️ LoRA：参数高效微调

全量微调太贵了（显存杀手）。我们使用 **LoRA (Low-Rank Adaptation)** 技术。
*   **原理**：冻结原模型，只训练旁边挂的一个小小的低秩矩阵。
*   **效果**：显存占用减少几百倍，效果几乎不降。

### 💻 代码演示：SFT 训练

```python
from hello_agents.tools import RLTrainingTool

rl_tool = RLTrainingTool()

# 一键启动 SFT 训练
result = rl_tool.run({
    "action": "train",
    "algorithm": "sft",
    "model_name": "Qwen/Qwen3-0.6B",  # 使用 Qwen 小模型
    "output_dir": "./models/sft_model",
    "use_lora": True,  # 开启 LoRA
    "num_epochs": 3
})
```

---

## 5. GRPO ��练：在奔跑中"自我进化"

SFT 只是模仿，**GRPO (Group Relative Policy Optimization)** 才是真正的强化学习。

### 🚀 为什么选 GRPO？

传统的 PPO 算法太复杂（需要 4 个模型，显存爆炸，训练不稳定）。
**GRPO** 是 DeepSeek-R1 背后的核心技术之一，它做了极简优化：
*   **不需要 Value Model**：省显存。
*   **组内相对奖励**：对于同一个问题，生成 N 个答案，让它们"内卷"。比平均分高的给正反馈，低的给负反馈。
*   **更稳定**：训练收敛更快。

### 💻 代码演示：GRPO 训练

```python
result = rl_tool.run({
    "action": "train",
    "algorithm": "grpo",
    "model_name": "./models/sft_model",  # 基于 SFT 模型继续训练
    "output_dir": "./models/grpo_model",
    "num_generations": 4,  # 每个问题生成 4 个答案进行PK
    "reward_type": "accuracy", # 以准确率为导向
    "use_lora": True
})
```

### 📈 效果评估

训练完后，你会发现：
*   **SFT 模型**：能回答问题，但难题经常算错，或者逻辑不通。
*   **GRPO 模型**：准确率明显提升，而且学会了更复杂的推理链路，甚至展现出一些"顿悟"时刻。

---

## 6. 本章总结

### 🌟 核心收获

1.  **思维升级**：从"优化 Prompt"升级到"优化模型权重"，从"模仿"升级到"进化"。
2.  **完整 Pipeline**：掌握了从数据准备 -> SFT -> GRPO -> 评估的完整 RLHF 流程。
3.  **技术落地**：学会了使用 LoRA 和 GRPO 这些前沿且高效的训练技术。

### 🚀 下一步是什么？

现在我们已经学会了如何训练一个强大的单体 Agent。但是，我们怎么知道这个 Agent 到底有多好？除了做数学题，它在通用任务上表现如何？

下一章（第十二章），我们将进入**智能体性能评估**领域，学习如何科学、全面地评测我们的 Agent 系统！

---

### 🔗 快速传送门
- **GitHub 源码**: [hello-agents/chapter11](https://github.com/jjyaoao/helloagents)
- **安装命令**: `pip install "hello-agents[rl]"`
