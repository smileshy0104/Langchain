# 更新日志 (Changelog)

## [1.0.0] - 2024-11-30

### 🎉 初始发布

#### ✨ 新增功能

**代码示例**:
- ✅ `01_basic_model_usage.py` - 5个 Model 基础用法示例
- ✅ `02_agent_usage.py` - 5个 Agent 使用示例（后处理方式）
- ✅ `03_real_world_applications.py` - 5个实际应用场景

**文档**:
- ✅ `README.md` - 项目主文档
- ✅ `QUICK_REFERENCE.md` - 快速参考指南
- ✅ `PROJECT_SUMMARY.md` - 项目概览
- ✅ `IMPLEMENTATION_NOTES.md` - 实现说明和技术细节
- ✅ `CHANGELOG.md` - 本文件

**配置文件**:
- ✅ `requirements.txt` - Python 依赖列表
- ✅ `.gitignore` - Git 忽略规则

#### 🔧 技术实现

**Pydantic V2 兼容**:
- 将所有 `@validator` 更新为 `@field_validator`
- 添加 `@classmethod` 装饰器
- 更新验证器方法签名

**ChatZhipuAI 适配**:
- 发现 ChatZhipuAI 不支持 `ToolStrategy`
- 实现后处理方式作为替代方案
- 添加详细的说明文档

**嵌套结构优化**:
- 复杂嵌套字段改为可选 (`default_factory=list`)
- 增强字段描述以引导模型生成
- 避免验证失败的同时保持数据质量

#### 🐛 修复问题

**验证器语法**:
- 修复 Pydantic V1 废弃警告
- 更新所有验证器到 V2 语法

**API 配置**:
- 统一使用环境变量读取 API Key
- 移除硬编码的测试密钥

**导入语句**:
- 移除未使用的 `ToolStrategy` 和 `ProviderStrategy` 导入
- 添加必要的 `field_validator` 导入

#### 📊 项目统计

- **代码文件**: 3个
- **文档文件**: 5个
- **示例总数**: 15个
- **代码行数**: ~930行
- **Schema 定义**: 18个
- **工具函数**: 5个

#### ⚠️ 已知限制

**ChatZhipuAI 限制**:
- 不支持 `ToolStrategy`（仅支持 `auto` tool choice）
- Agent 示例使用后处理方式实现
- 相比原生 ToolStrategy 需要额外调用

**建议**:
- Model 层面的结构化输出完全支持
- Agent 层面建议使用 OpenAI 等支持 ToolStrategy 的模型

#### 📚 示例清单

**01_basic_model_usage.py**:
1. 基础 Pydantic Model - 电影信息
2. 嵌套结构 - 电影详情含演员阵容
3. 字段验证器 - 联系信息验证和清洗
4. 获取原始响应 - 包含 token 使用信息
5. 提取多个实例 - 团队成员列表

**02_agent_usage.py**:
1. 基础 Agent 结构化输出 - 天气查询
2. 复杂查询 - 研究结果总结
3. 多工具协作 - 计算 + 文件 + 搜索
4. 带记忆的 Agent - 对话摘要
5. Pydantic 验证错误处理

**03_real_world_applications.py**:
1. 数据提取 - 邮件签名 → 联系信息
2. 内容分类 - 新闻文章分类
3. 表单填充 - 自然语言 → 求职表单
4. 评分系统 - 作文评分
5. 产品信息提取 - 描述 → 结构化数据

#### 🎯 设计决策

**后处理方式实现 Agent 结构化输出**:

```python
# 两步法
# 1. Agent 执行任务
agent = create_agent(model=model, tools=[...])
result = agent.invoke({"messages": [...]})

# 2. 提取结构化数据
model_with_structure = model.with_structured_output(Schema)
structured = model_with_structure.invoke(f"提取结构化信息：\n{result['messages'][-1].content}")
```

**优势**:
- ✅ 兼容 ChatZhipuAI
- ✅ 清晰的两阶段处理
- ✅ 易于调试和理解
- ✅ 可以应用于任何 LLM

**劣势**:
- ❌ 需要两次 API 调用
- ❌ 可能增加延迟
- ❌ 成本稍高

#### 🔮 未来计划

**v1.1 (短期)**:
- [ ] 添加 OpenAI 模型的 ToolStrategy 对比示例
- [ ] 添加流式输出示例
- [ ] 性能基准测试

**v1.2 (中期)**:
- [ ] 批量处理示例
- [ ] 错误重试机制
- [ ] 缓存优化

**v2.0 (长期)**:
- [ ] 多模态支持
- [ ] 向量数据库集成
- [ ] Web UI 演示平台

---

## 版本说明

### 语义化版本

本项目遵循[语义化版本规范](https://semver.org/lang/zh-CN/):

- **主版本号**: 不兼容的 API 修改
- **次版本号**: 向下兼容的功能性新增
- **修订号**: 向下兼容的问题修正

### 更新类型标记

- ✨ `新增功能` - 新增特性或功能
- 🔧 `技术实现` - 技术实现细节
- 🐛 `修复问题` - Bug 修复
- 📚 `文档` - 文档更新
- ⚠️ `已知限制` - 已知的限制或问题
- 🎯 `设计决策` - 重要的设计决策
- 🔮 `未来计划` - 规划的功能

---

**项目开始**: 2024-11-30
**当前版本**: v1.0.0
**状态**: ✅ 生产可用
