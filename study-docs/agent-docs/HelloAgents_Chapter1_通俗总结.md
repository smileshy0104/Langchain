# 第一章《初识智能体》通俗总结

> 从零开始理解智能体：定义、类型、范式与应用

面向刚接触智能体的同学，这份笔记用通俗易懂的方式解释智能体的核心概念，配上生动的例子和实战案例，帮助你快速建立整体认知。

---

## 开篇：什么是智能体时代？

**2025年，我们正式进入"智能体时代"**

过去几年，AI的进步主要体现在：
- 2020-2022：训练更大的模型（GPT-3 → GPT-4）
- 2023-至今：**让模型变成能自主行动的智能体**

**类比：**
- **传统AI**：像一本百科全书，你问它答，但不会主动帮你做事
- **智能体**：像一个助理，理解你的目标后，自己规划、执行、反馈，直到完成任务

---

## 第一部分：智能体到底是什么？

### 1.1 最简定义

> **智能体（Agent）= 能够感知环境、自主决策、采取行动的实体**

**三个关键要素：**

```
┌──────────────────────────────┐
│         环境（Environment）      │
│   (外部世界：网页、数据库、用户等)  │
└──────────────────────────────┘
         ↑          ↓
      感知器      执行器
    (Sensors)  (Actuators)
         ↑          ↓
┌──────────────────────────────┐
│        智能体（Agent）          │
│   感知 → 思考 → 行动            │
└──────────────────────────────┘
```

---

### 1.2 生活中的智能体例子

| 智能体 | 传感器（感知） | 决策器（思考） | 执行器（行动） | 环境 |
|-------|-------------|-------------|-------------|------|
| **扫地机器人** | 摄像头、碰撞传感器 | 路径规划算法 | 轮子、吸尘器 | 房间地板 |
| **自动驾驶汽车** | 雷达、摄像头、GPS | 深度学习模型 | 方向盘、刹车、油门 | 道路、行人、车辆 |
| **智能客服** | 用户文本输入 | LLM（如GPT-4） | 文本回复、工具调用 | 聊天对话、订单系统 |
| **交易机器人** | 股票价格API | 强化学习策略 | 买入/卖出指令 | 金融市场 |
| **游戏AI** | 游戏画面像素 | 神经网络 | 操作指令（上下左右） | 游戏世界 |

---

### 1.3 智能体的核心特征

#### 特征1：自主性（Autonomy）

**不是简单的脚本执行，而是自己决定下一步做什么**

**对比：**
```python
# ❌ 这不是智能体（死板的脚本）
def process_order(order):
    if order.amount < 100:
        approve()
    else:
        reject()

# ✅ 这是智能体（自主决策）
def intelligent_agent(order):
    # 1. 分析订单历史
    user_history = query_database(order.user_id)

    # 2. 评估风险
    risk_score = fraud_detection_model(order, user_history)

    # 3. 综合决策
    if risk_score < 0.3 and user_history['trust_level'] == 'high':
        approve()
    elif risk_score < 0.7:
        manual_review()
    else:
        reject()
```

---

#### 特征2：反应性（Reactivity）

**能及时感知环境变化并作出响应**

**例子：自动驾驶**
```
场景：高速公路行驶
时间 0秒：前方车辆正常行驶
时间 1秒：前方车辆突然刹车（环境变化）
智能体反应：
  - 0.1秒内检测到刹车灯亮起（感知）
  - 0.2秒内决定减速或变道（决策）
  - 0.3秒内执行刹车（行动）
```

---

#### 特征3：主动性（Proactivity）

**不仅被动响应，还能主动追求目标**

**例子：智能日程助理**
```
被动响应：用户问"明天几点开会" → 助理查询日历
主动行为：
  - 检测到两个会议时间冲突 → 主动提醒用户
  - 发现明天有会议但没订会议室 → 自动预订
  - 预测路上堵车 → 提前建议出发时间
```

---

#### 特征4：学习能力（Learning）

**从经验中改进自己的行为**

**例子：推荐系统**
```
第1天：推荐用户可能喜欢的电影（基于协同过滤）
用户反馈：看了A片，没看B片
第7天：学习到用户偏好动作片，不喜欢文艺片
第30天：推荐准确率从60%提升到85%
```

---

### 1.4 智能体 vs 普通程序

| 对比维度 | 普通程序 | 智能体 |
|---------|---------|--------|
| **决策方式** | IF-THEN固定规则 | 根据环境动态调整 |
| **目标设定** | 代码写死 | 理解自然语言目标 |
| **学习能力** | 无（除非重新编程） | 从数据/经验中学习 |
| **适应性** | 环境变化需修改代码 | 自动适应新环境 |
| **交互方式** | 固定接口（API调用） | 自然语言对话 |

**例子：**
```
任务：帮用户订机票

普通程序：
  input: 出发地、目的地、日期
  output: 航班列表
  → 用户需要自己比较价格、时间、转机等

智能体：
  User: "帮我订一张去北京的机票，越便宜越好"
  Agent:
    1. 理解目标：价格优先
    2. 查询：对比多个网站价格
    3. 考虑：检查用户历史偏好（经济舱、早班飞机）
    4. 推荐：CA1234，早上7点，￥680（比其他航班便宜15%）
    5. 主动：需要我同时订酒店吗？
```

---

## 第二部分：智能体的演进范式

### 2.1 传统范式：从简单到复杂的四个层次

#### 层次1：简单反射式智能体（Simple Reflex Agent）

**核心思想：** `IF (感知) THEN (行动)`

**类比：** 恒温器
```
IF 温度 > 25°C THEN 开空调
IF 温度 < 20°C THEN 关空调
```

**优点：**
- ✅ 响应快速（毫秒级）
- ✅ 简单可靠

**缺点：**
- ❌ 只看当前状态，没有记忆
- ❌ 无法处理复杂场景

**代码示例：**
```python
class SimpleReflexAgent:
    def __init__(self):
        self.rules = {
            "温度过高": lambda: self.turn_on_ac(),
            "温度过低": lambda: self.turn_off_ac(),
        }

    def perceive(self, temperature):
        if temperature > 25:
            return "温度过高"
        elif temperature < 20:
            return "温度过低"
        return "正常"

    def act(self, temperature):
        condition = self.perceive(temperature)
        if condition in self.rules:
            self.rules[condition]()
```

---

#### 层次2：基于模型的反射式智能体（Model-Based Reflex Agent）

**核心思想：** 维护一个"世界模型"，记住环境状态

**类比：** 自动驾驶在隧道中
```
场景：进入隧道，GPS信号丢失
简单反射：检测不到前车 → 误以为没车
基于模型：记住进隧道前，前车在50米处 → 推测前车仍在前方
```

**内部状态示例：**
```python
class ModelBasedAgent:
    def __init__(self):
        self.world_model = {
            "前车位置": None,
            "车道": 2,
            "速度": 80,
        }

    def update_model(self, perception):
        """根据感知更新世界模型"""
        if perception.get("前车可见"):
            self.world_model["前车位置"] = perception["前车位置"]
        else:
            # GPS丢失时，根据速度推测位置
            self.world_model["前车位置"] += self.world_model["速度"] * 0.1

    def decide(self):
        """基于世界模型决策"""
        if self.world_model["前车位置"] < 30:
            return "减速"
        return "保持速度"
```

---

#### 层次3：基于目标的智能体（Goal-Based Agent）

**核心思想：** 有明确目标，会规划达成目标的步骤

**类比：** 导航系统
```
目标：从A地到B地
规划：
  1. 搜索所有可能路径
  2. 评估每条路径（距离、时间、路况）
  3. 选择最优路径
  4. 动态调整（如遇到堵车）
```

**与反射式的区别：**
```
反射式：IF 前方有障碍 THEN 停车
目标式：IF 前方有障碍 THEN
         评估：绕路能否更快到达目的地？
         决策：如果绕路时间<5分钟，选择绕路
              否则等待障碍清除
```

---

#### 层次4：基于效用的智能体（Utility-Based Agent）

**核心思想：** 不仅追求目标，还要最大化"效用"（多目标权衡）

**类比：** 打车软件选择路线
```
目标：从家到机场
考虑因素：
  - 时间：30分钟（权重0.5）
  - 费用：￥80（权重0.3）
  - 舒适度：不堵车（权重0.2）

路线A：高速，25分钟，￥100，堵车概率10%
  效用 = 0.5×(35/25) + 0.3×(80/100) + 0.2×0.9 = 0.7 + 0.24 + 0.18 = 1.12

路线B：国道，35分钟，￥60，堵车概率30%
  效用 = 0.5×(35/35) + 0.3×(80/60) + 0.2×0.7 = 0.5 + 0.4 + 0.14 = 1.04

选择：路线A（效用更高）
```

**代码框架：**
```python
class UtilityBasedAgent:
    def utility(self, state):
        """计算状态的效用值"""
        return (
            0.5 * self.time_score(state) +
            0.3 * self.cost_score(state) +
            0.2 * self.comfort_score(state)
        )

    def decide(self, possible_actions):
        """选择效用最大的行动"""
        best_action = None
        best_utility = -float('inf')

        for action in possible_actions:
            next_state = self.predict_result(action)
            u = self.utility(next_state)
            if u > best_utility:
                best_utility = u
                best_action = action

        return best_action
```

---

#### 层次5：学习型智能体（Learning Agent）

**核心思想：** 从经验中学习，持续改进

**经典例子：AlphaGo Zero**
```
Day 1：随机下棋（完全不会）
Day 3：通过自我对弈学习了基本规则
Day 7：达到业余水平
Day 40：击败所有人类棋手
```

**学习循环：**
```
1. 执行行动 → 观察结果
2. 评估：这个行动好还是坏？
3. 更新策略：好的行动增加概率，坏的减少
4. 重复
```

**强化学习示例：**
```python
class LearningAgent:
    def __init__(self):
        self.Q = {}  # Q表：存储"在状态s采取行动a的预期回报"

    def learn(self, state, action, reward, next_state):
        """Q-learning算法"""
        # 当前Q值
        current_q = self.Q.get((state, action), 0)

        # 最优未来Q值
        max_future_q = max([self.Q.get((next_state, a), 0)
                           for a in self.available_actions(next_state)])

        # 更新Q值
        self.Q[(state, action)] = current_q + 0.1 * (
            reward + 0.9 * max_future_q - current_q
        )

    def choose_action(self, state):
        """选择Q值最大的行动（有一定随机探索）"""
        if random.random() < 0.1:  # 10%概率探索
            return random.choice(self.available_actions(state))
        else:  # 90%概率利用已知知识
            return max(self.available_actions(state),
                      key=lambda a: self.Q.get((state, a), 0))
```

---

### 2.2 新范式：LLM驱动的智能体

**革命性变化：** 从"手工编程"到"自然语言指令"

#### 传统智能体 vs LLM智能体

| 维度 | 传统智能体 | LLM智能体 |
|------|-----------|----------|
| **知识来源** | 人工编写规则 或 训练专用模型 | 预训练的通用知识（互联网文本） |
| **任务理解** | 需要形式化定义 | 直接理解自然语言 |
| **规划能力** | 需要手写搜索算法（A*、MCTS） | LLM自己分解任务 |
| **工具使用** | 硬编码API调用 | 自己决定何时调用哪个工具 |
| **泛化能力** | 只能做训练过的任务 | 零样本处理新任务 |
| **开发成本** | 数月开发+训练 | 几小时写Prompt |

---

#### LLM智能体的核心能力

**1. 自然语言接口**
```
传统：robot.navigate(x=10.5, y=23.7, theta=1.57)
LLM：  "帮我把桌上的咖啡杯拿过来"
```

**2. 任务分解**
```
用户："帮我策划一场生日派对"

LLM Agent自动分解：
  1. 确定预算和人数
  2. 选择场地
  3. 准备食物和装饰
  4. 发送邀请
  5. 安排活动流程
```

**3. 工具调用（Function Calling）**
```python
# LLM可以自己决定调用哪个工具
tools = {
    "web_search": "搜索互联网",
    "calculator": "精确计算",
    "send_email": "发送邮件",
}

User: "帮我算一下，比特币3年前是$43000，现在$45000，涨了百分之几？"

LLM思考：
  Thought: 我需要计算 (45000-43000)/43000 * 100，但我不擅长算数
  Action: calculator("(45000-43000)/43000*100")
  Observation: 4.65

  Answer: 涨了约4.65%
```

**4. 记忆与上下文**
```
第1次对话：
User: 我喜欢历史文化景点
Agent: [记住偏好]

第10次对话：
User: 推荐一个旅游目的地
Agent: 根据您的偏好，推荐西安（兵马俑、古城墙等历史景点）
```

---

## 第三部分：如何描述智能体的任务环境？

### 3.1 PEAS模型

**PEAS = Performance + Environment + Actuators + Sensors**

这是设计智能体的标准框架，就像建筑师的蓝图。

---

#### 示例1：自动驾驶汽车

| 要素 | 内容 |
|------|------|
| **P - Performance（绩效指标）** | • 安全性：零事故 <br> • 效率：到达时间最短 <br> • 舒适性：平稳驾驶 <br> • 守法性：遵守交通规则 |
| **E - Environment（环境）** | • 道路：高速/城市/乡村 <br> • 其他车辆、行人 <br> • 天气：晴天/雨雪/雾霾 <br> • 交通灯、路标 |
| **A - Actuators（执行器）** | • 方向盘 <br> • 油门 <br> • 刹车 <br> • 转向灯 |
| **S - Sensors（传感器）** | • 摄像头 <br> • 雷达 <br> • 激光雷达（LiDAR） <br> • GPS |

---

#### 示例2：智能旅行助手

| 要素 | 内容 |
|------|------|
| **P - Performance** | • 推荐准确度（用户满意度>90%） <br> • 响应速度（<5秒） <br> • 完成度（完整的旅行计划） |
| **E - Environment** | • 天气API <br> • 景点数据库 <br> • 票务系统 <br> • 用户对话历史 |
| **A - Actuators** | • 调用天气API <br> • 搜索景点信息 <br> • 生成文本回复 <br> • 预订服务 |
| **S - Sensors** | • 用户文本输入 <br> • API返回结果 <br> • 用户反馈（满意/不满意） |

---

### 3.2 环境的特性分类

理解环境特性，可以帮助选择合适的智能体设计。

| 特性 | 说明 | 例子 |
|------|------|------|
| **完全可观察 vs 部分可观察** | 传感器能否感知全部状态？ | • 下棋：完全可观察（棋盘透明） <br> • 扑克牌：部分可观察（看不到对手的牌） |
| **确定性 vs 随机性** | 行动的结果是否确定？ | • 数独：确定（规则固定） <br> • 股票交易：随机（价格波动） |
| **离散 vs 连续** | 状态和行动是离散还是连续？ | • 围棋：离散（361个位置） <br> • 机器人：连续（位置、速度） |
| **静态 vs 动态** | 环境是否随时间变化？ | • 拼图：静态 <br> • 足球比赛：动态（对手在移动） |
| **单智能体 vs 多智能体** | 环境中有其他智能体吗？ | • 扫地机器人：单智能体 <br> • 自动驾驶：多智能体（其他车） |

---

#### 环境特性对智能体设计的影响

**部分可观察 → 需要记忆**
```python
# 自动驾驶进入隧道（GPS失效）
class Agent:
    def __init__(self):
        self.memory = {}  # 记住进隧道前的车辆位置
```

**随机性 → 需要概率推理**
```python
# 股票交易
def decide(self, price):
    # 不是简单的IF price>100 THEN buy
    # 而是基于概率分布
    expected_return = self.predict_distribution(price)
    if expected_return.mean() > threshold:
        buy()
```

**多智能体 → 需要博弈论**
```python
# 拍卖系统
def bid(self, current_price):
    # 考虑其他竞拍者的策略
    competitors_strategy = self.estimate_competitors()
    optimal_bid = self.game_theory_solver(competitors_strategy)
```

---

## 第四部分：智能体的工作循环

### 4.1 经典Agent Loop

```
┌─────────────────────────────────┐
│   1. 感知（Perceive）             │
│      收集环境信息                  │
└──────────┬──────────────────────┘
           ↓
┌──────────┴──────────────────────┐
│   2. 思考（Think）                │
│      分析、推理、规划               │
└──────────┬──────────────────────┘
           ↓
┌──────────┴──────────────────────┐
│   3. 行动（Act）                  │
│      执行决策                     │
└──────────┬──────────────────────┘
           ↓
         环境变化
           ↓
         （循环）
```

---

### 4.2 ReAct范式：Thought-Action-Observation

**ReAct = Reasoning + Acting**

这是LLM智能体最常用的模式，让模型边思考边行动。

#### 格式规范

```
Thought: [推理过程，解释为什么这样做]
Action: [具体的行动，如调用工具]
Observation: [行动的结果]
... (重复多轮)
Thought: 任务完成
Action: finish(answer="最终答案")
```

---

#### 实战案例：智能旅行助手

**任务：** 推荐北京旅游景点

```
【第1轮】
Thought: 用户想去北京旅游，我应该先查一下北京的天气，
         根据天气情况推荐合适的景点（晴天推荐户外，雨天推荐室内）
Action: get_weather(city="北京", date="2025-11-20")
Observation: 北京 2025年11月20日 晴天 15°C ☀️

【第2轮】
Thought: 天气不错，适合户外活动。我现在搜索北京的热门景点，
         优先考虑户外景点。
Action: get_attraction(city="北京", type="户外")
Observation:
  1. 颐和园 - 皇家园林，风景优美
  2. 长城 - 世界奇迹，需要体力
  3. 天坛 - 明清皇帝祭天的地方

【第3轮】
Thought: 我已经获得了天气和景点信息，可以给用户一个完整的推荐了
Action: finish(answer="""
推荐您去颐和园！

理由：
1. 天气：明天晴天15°C，非常适合户外游览
2. 景点：颐和园是中国最大的皇家园林，秋季景色宜人
3. 建议：早上9点到达，避开人流高峰，游览时间约3-4小时

其他备选：如果您体力好，也可以考虑爬长城（需要一整天）
""")
```

---

### 4.3 为什么需要结构化输出？

**问题：** LLM的自然语言输出难以被程序解析

**解决方案：** 规定固定格式

```python
# ❌ 不规范的输出（难以解析）
"我觉得应该先查天气，然后可能调用get_weather函数..."

# ✅ 规范的输出（容易解析）
"""
Thought: 需要查询天气
Action: get_weather(city="北京")
"""

# 解析代码
import re

def parse_action(text):
    match = re.search(r'Action:\s*(\w+)\((.*?)\)', text)
    if match:
        function_name = match.group(1)
        args = match.group(2)
        return function_name, args
```

---

### 4.4 完整的Agent主循环实现

```python
def agent_loop(user_query, max_iterations=5):
    """
    智能体主循环
    """
    # 初始化
    prompt_history = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": user_query}
    ]

    for iteration in range(max_iterations):
        # 1. LLM生成Thought和Action
        response = llm.call(prompt_history)

        # 2. 解析Action
        thought = extract_thought(response)
        action_name, action_args = parse_action(response)

        # 3. 执行Action
        if action_name == "finish":
            # 任务完成
            return action_args["answer"]

        # 4. 调用工具
        observation = execute_tool(action_name, action_args)

        # 5. 更新历史
        prompt_history.append({
            "role": "assistant",
            "content": f"Thought: {thought}\nAction: {action_name}({action_args})"
        })
        prompt_history.append({
            "role": "user",
            "content": f"Observation: {observation}"
        })

    return "达到最大迭代次数，任务未完成"
```

---

## 第五部分：智能体的应用模式

### 5.1 智能体作为工具（Copilot模式）

**特点：** 辅助人类，但最终决策权在人类

**代表产品：**
- **GitHub Copilot**：代码补全、函数生成
- **Claude Code**：理解需求、编写代码、调试
- **Cursor**：AI配对编程

**工作流程：**
```
程序员：写代码遇到困难
   ↓
Copilot：建议代码片段
   ↓
程序员：审查、修改、采纳
   ↓
最终代码由人类确认
```

**适用场景：**
- ✅ 提高效率（重复性工作）
- ✅ 学习新技术（AI解释代码）
- ✅ 人类保持控制权（安全关键场景）

---

### 5.2 智能体作为协作者（Autonomous模式）

**特点：** 给定目标，智能体自主完成全流程

**代表产品：**
- **AutoGPT**：长时间自主运行
- **BabyAGI**：任务分解与执行
- **MetaGPT**：多角色协作开发软件

**工作流程：**
```
用户：给一个高层目标（"调研电动车市场"）
   ↓
Agent：自己分解任务
   → 搜索资料
   → 整理数据
   → 写报告
   → 自我审查
   ↓
用户：收到完整报告
```

**适用场景：**
- ✅ 明确且重复的任务（写周报、数据分析）
- ✅ 人类没时间处理的琐事
- ❌ 不适合：高风险决策（如医疗诊断）

---

### 5.3 Workflow vs Agent的对比

| 维度 | Workflow（流程自动化） | Agent（智能体） |
|------|---------------------|---------------|
| **决策方式** | 预定义流程图 | 动态规划 |
| **灵活性** | 低（固定路径） | 高（根据情况调整） |
| **适用场景** | 明确规则（报销审批） | 开放任务（内容创作） |
| **可预测性** | 高（每次相同） | 中（有随机性） |
| **开发成本** | 高（需要穷举所有分支） | 低（只需写Prompt） |
| **维护成本** | 高（规则变化需修改代码） | 低（调整Prompt） |

**例子：电商退款**

**Workflow方式：**
```python
if 金额 < 100 and 时间 <= 7天:
    自动通过
elif 金额 < 500:
    转客服审核
else:
    转主管审批
```

**Agent方式：**
```python
prompt = """
退款政策：7天无理由退款，定制品除外
请分析：
- 用户历史行为（是否恶意退款）
- 商品状况
- 退款原因
决策：批准/拒绝/转人工
"""
```

**混合方案（最佳实践）：**
```
简单case（80%）→ Workflow快速处理
复杂case（20%）→ Agent智能决策
高风险case → 人工审核
```

---

## 第六部分：5分钟实战 - 构建旅行助手

### 6.1 系统架构

```
┌──────────────────────────────────┐
│         工具层（Tools）             │
│  get_weather() | get_attraction() │
└──────────────┬───────────────────┘
               ↓
┌──────────────┴───────────────────┐
│      LLM推理层（Brain）            │
│   OpenAI GPT-4 / Claude Sonnet   │
└──────────────┬───────────────────┘
               ↓
┌──────────────┴───────────────────┐
│     主循环（Agent Loop）           │
│  Thought → Action → Observation   │
└──────────────────────────────────┘
```

---

### 6.2 工具定义

```python
def get_weather(city: str, date: str) -> str:
    """
    查询天气

    Args:
        city: 城市名，如"北京"
        date: 日期，格式"YYYY-MM-DD"

    Returns:
        天气描述，如"北京 2025-11-20 晴天 15°C"
    """
    # 实际调用天气API（这里用wttr.in演示）
    url = f"https://wttr.in/{city}?format=%C+%t"
    response = requests.get(url)
    return f"{city} {date} {response.text}"


def get_attraction(city: str, keyword: str = "") -> str:
    """
    搜索景点

    Args:
        city: 城市名
        keyword: 关键词（可选），如"历史"、"自然"

    Returns:
        景点列表
    """
    # 实际调用搜索API（如Tavily、Google）
    query = f"{city} {keyword} 旅游景点"
    results = search_api.search(query)
    return "\n".join([f"{i+1}. {r['title']}" for i, r in enumerate(results[:5])])


# 注册工具
available_tools = {
    "get_weather": get_weather,
    "get_attraction": get_attraction,
}
```

---

### 6.3 系统提示词

```python
AGENT_SYSTEM_PROMPT = """
你是一个专业的旅行助手。你的任务是根据用户需求，推荐合适的旅游景点。

【可用工具】
1. get_weather(city, date) - 查询指定城市和日期的天气
2. get_attraction(city, keyword) - 搜索景点

【工作流程】
1. 理解用户需求
2. 如果需要，查询天气信息
3. 根据天气和用户偏好，搜索景点
4. 给出详细的推荐理由

【输出格式】
你必须按以下格式输出：

Thought: [你的分析和计划]
Action: function_name(arg1="value1", arg2="value2")

当工具返回结果后，你会收到：
Observation: [工具返回的结果]

你需要继续思考，直到可以给出最终答案：
Thought: 我已经收集到足够信息
Action: finish(answer="你的推荐内容")

【示例】
User: 推荐一个北京的景点

Thought: 先查天气，根据天气推荐景点
Action: get_weather(city="北京", date="2025-11-20")

Observation: 北京 2025-11-20 晴天 15°C

Thought: 天气不错，推荐户外景点
Action: get_attraction(city="北京", keyword="户外")

Observation: 1. 颐和园 2. 长城 3. 天坛

Thought: 已收集到足够信息，可以推荐了
Action: finish(answer="推荐颐和园，天气晴朗适合户外...")
"""
```

---

### 6.4 完整代码

```python
import re
import requests

class TravelAgent:
    def __init__(self, llm_client):
        self.llm = llm_client
        self.tools = {
            "get_weather": self.get_weather,
            "get_attraction": self.get_attraction,
        }

    def get_weather(self, city, date):
        """查询天气"""
        url = f"https://wttr.in/{city}?format=%C+%t"
        response = requests.get(url)
        return f"{city} {date} {response.text}"

    def get_attraction(self, city, keyword=""):
        """搜索景点"""
        # 这里简化，实际应调用搜索API
        mock_results = {
            "北京": ["颐和园", "长城", "故宫", "天坛"],
            "上海": ["外滩", "东方明珠", "豫园", "迪士尼"],
        }
        attractions = mock_results.get(city, ["景点1", "景点2"])
        return "\n".join([f"{i+1}. {a}" for i, a in enumerate(attractions)])

    def parse_action(self, text):
        """解析Action"""
        match = re.search(r'Action:\s*(\w+)\((.*?)\)', text, re.DOTALL)
        if not match:
            return None, None

        func_name = match.group(1)
        args_str = match.group(2)

        # 简单解析参数（实际应用中用更robust的方法）
        args = {}
        for arg in args_str.split(','):
            if '=' in arg:
                key, value = arg.split('=')
                args[key.strip()] = value.strip(' "\'')

        return func_name, args

    def run(self, user_query, max_iterations=5):
        """主循环"""
        messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_query}
        ]

        for i in range(max_iterations):
            print(f"\n{'='*50}")
            print(f"迭代 {i+1}")
            print(f"{'='*50}")

            # 1. LLM生成响应
            response = self.llm.chat(messages)
            print(f"\nLLM输出:\n{response}")

            # 2. 解析Action
            func_name, args = self.parse_action(response)

            # 3. 检查是否完成
            if func_name == "finish":
                return args.get("answer", response)

            # 4. 执行工具
            if func_name in self.tools:
                observation = self.tools[func_name](**args)
                print(f"\nObservation:\n{observation}")

                # 5. 更新消息历史
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": f"Observation: {observation}"})
            else:
                print(f"⚠️ 未知工具: {func_name}")
                break

        return "达到最大迭代次数"


# 使用示例
if __name__ == "__main__":
    # 初始化LLM客户端（使用OpenAI或其他兼容API）
    from openai import OpenAI
    llm = OpenAI(api_key="your-api-key")

    # 创建智能体
    agent = TravelAgent(llm)

    # 运行
    result = agent.run("帮我推荐一个北京的旅游景点")
    print(f"\n{'='*50}")
    print(f"最终答案:\n{result}")
```

---

### 6.5 运行效果

```
用户输入: 帮我推荐一个北京的旅游景点

==================================================
迭代 1
==================================================

LLM输出:
Thought: 用户想要北京的景点推荐，我应该先查询天气，
         根据天气情况推荐合适的景点
Action: get_weather(city="北京", date="2025-11-20")

Observation:
北京 2025-11-20 晴天 15°C ☀️

==================================================
迭代 2
==================================================

LLM输出:
Thought: 天气晴朗，温度适中，非常适合户外活动。
         我现在搜索北京的景点
Action: get_attraction(city="北京")

Observation:
1. 颐和园
2. 长城
3. 故宫
4. 天坛

==================================================
迭代 3
==================================================

LLM输出:
Thought: 我已经获得了天气和景点信息，可以给出推荐了
Action: finish(answer="推荐您去颐和园！

【推荐理由】
1. 天气优势：明天晴天15°C，非常适合户外游览
2. 景点特色：颐和园是中国最大的皇家园林，集山水园林之大成
3. 游览建议：
   - 最佳时间：早上9点到达，避开人流高峰
   - 游览时长：3-4小时
   - 必看景点：万寿山、昆明湖、长廊

【其他备选】
- 如果您想感受皇家建筑的恢弘，可以选择故宫
- 如果您体力充沛，可以考虑爬长城（需要一整天）")

==================================================
最终答案:
推荐您去颐和园！

【推荐理由】
1. 天气优势：明天晴天15°C，非常适合户外游览
2. 景点特色：颐和园是中国最大的皇家园林，集山水园林之大成
3. 游览建议：
   - 最佳时间：早上9点到达，避开人流高峰
   - 游览时长：3-4小时
   - 必看景点：万寿山、昆明湖、长廊

【其他备选】
- 如果您想感受皇家建筑的恢弘，可以选择故宫
- 如果您体力充沛，可以考虑爬长城（需要一整天）
```

---

### 6.6 这个例子展示了什么能力？

| 能力 | 表现 |
|------|------|
| **任务分解** | 自动拆分成"查天气"和"搜景点"两步 |
| **工具调用** | 正确调用get_weather和get_attraction |
| **上下文记忆** | 记住天气信息，结合景点推荐 |
| **推理综合** | 根据天气（晴天）推荐户外景点 |
| **结构化输出** | 按Thought-Action格式输出，易于解析 |

---

## 第七部分：延伸思考

### 思考题1：PEAS模型练习

**任务：** 设计一个"智能健身教练"

提示：
- Performance：用户的健身目标达成率、受伤率
- Environment：可穿戴设备、健身房器械
- Actuators：语音指导、调整训练计划
- Sensors：心率、运动轨迹

---

### 思考题2：Workflow vs Agent

**场景：** 电商退款审批

- Workflow方案：写死规则（金额<100自动通过）
- Agent方案：AI分析用户历史、商品状况
- 混合方案：如何结合两者优势？

---

### 思考题3：功能增强

为旅行助手添加：
1. **记忆功能**：记住用户偏好（喜欢历史文化）
2. **备选方案**：门票售罄时自动推荐替代景点
3. **反思能力**：连续拒绝3次后调整推荐策略

---

### 思考题4：系统1 vs 系统2

卡尼曼的双系统理论在智能体中的应用：
- **系统1（快速直觉）**：神经网络快速识别模式
- **系统2（慢速推理）**：符号系统逻辑推理

设计一个医疗诊断智能体，如何结合两者？

---

### 思考题5：智能体的局限

1. **为什么会产生幻觉？**
   - LLM本质是统计模型，而非知识库
   - 缺乏事实验证机制

2. **没有循环限制的风险？**
   - 无限循环（重复调用同一工具）
   - 成本爆炸（每次调用GPT-4都要钱）

3. **如何评估智能体？**
   - 准确率、效率、成本、用户满意度
   - 鲁棒性、安全性、可解释性

---

## 总结：本章核心要点

### ✅ 核心概念

1. **智能体定义**：感知→思考→行动的自主实体
2. **PEAS模型**：设计智能体的标准框架
3. **ReAct范式**：Thought-Action-Observation循环
4. **LLM革命**：从编程到Prompting

---

### ✅ 关键技能

1. 能够用PEAS描述任务环境
2. 理解不同智能体范式的适用场景
3. 实现基础的Agent Loop
4. 知道何时用Workflow、何时用Agent

---

### ✅ 实践建议

1. **从简单开始**：先做单工具调用，再做多步规划
2. **明确边界**：什么任务适合智能体，什么不适合
3. **人机协作**：高风险决策保留人工审核
4. **持续监控**：记录日志、分析失败case

---

## 推荐资源

**在线课程：**
- Datawhale《Hello-Agents》完整教程
- DeepLearning.AI《LangChain for LLM Application Development》

**开源框架：**
- LangChain / LangGraph
- AutoGPT
- MetaGPT

**学术论文：**
- ReAct: Synergizing Reasoning and Acting in Language Models
- Toolformer: Language Models Can Teach Themselves to Use Tools

---

如果需要把这份总结扩展成幻灯片或进一步细化代码实现，可以在此基础上继续补充示例和图示！
