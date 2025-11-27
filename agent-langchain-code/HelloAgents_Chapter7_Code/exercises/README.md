# Hello Agents 第七章习题解答

本目录包含第七章的习题解答和扩展实现。

## 📚 习题列表

### 习题 1: 框架设计理念分析
**题目**：思考"万物皆工具"的优缺点

**解答文件**：[exercise_01_framework_analysis.md](exercise_01_framework_analysis.md)

---

### 习题 2: 多模型支持
**题目**：实践添加新的模型供应商

**解答文件**：[exercise_02_new_model_provider.py](exercise_02_new_model_provider.py)

**实现内容**：
- 添加 Anthropic Claude 模型支持
- 添加 Moonshot AI 模型支持
- 统一的模型接口封装

---

### 习题 3: Agent 实现对比
**题目**：对比不同 Agent 的适用场景

**解答文件**：[exercise_03_agent_comparison.md](exercise_03_agent_comparison.md)

**对比内容**：
- SimpleAgent vs ReActAgent vs ReflectionAgent vs PlanAndSolveAgent
- 每种 Agent 的优缺点
- 实际应用场景示例

---

### 习题 4: 工具开发
**题目**：实现一个实用的自定义工具

**解答文件**：[exercise_04_custom_tools.py](exercise_04_custom_tools.py)

**实现内容**：
- 天气查询工具（接入真实 API）
- 文件操作工具
- 数据库查询工具
- API 调用工具

---

### 习题 5: 系统扩展
**题目**：设计插件系统架构

**解答文件**：[exercise_05_plugin_system.py](exercise_05_plugin_system.py)

**实现内容**：
- 插件加载机制
- 插件生命周期管理
- 插件依赖管理
- 插件配置系统

---

## 🚀 运行习题

### 运行单个习题

```bash
# 习题 2: 新模型供应商
python exercises/exercise_02_new_model_provider.py

# 习题 4: 自定义工具
python exercises/exercise_04_custom_tools.py

# 习题 5: 插件系统
python exercises/exercise_05_plugin_system.py
```

### 运行所有习题

```bash
python exercises/run_all_exercises.py
```

---

## 📖 学习建议

1. **按顺序学习**：习题难度递进，建议按顺序完成
2. **动手实践**：不要只看代码，要自己动手修改和实验
3. **扩展思考**：每道习题都可以继续深入扩展
4. **对比学习**：对比 Hello-Agents 原始实现和 LangChain 实现的区别

---

## 💡 扩展挑战

完成基础习题后，可以尝试：

1. **实现 ReflectionAgent** - 带自我反思能力的 Agent
2. **实现 PlanAndSolveAgent** - 计划-执行模式的 Agent
3. **添加工具链功能** - 让多个工具协同工作
4. **实现异步工具调用** - 提升性能
5. **添加工具错误恢复** - 提高鲁棒性

---

**Happy Learning! 🎓**
