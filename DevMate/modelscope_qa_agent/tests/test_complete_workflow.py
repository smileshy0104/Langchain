#!/usr/bin/env python3
"""
Complete Workflow Verification Test
完整工作流验证测试

测试从文件上传到问答的完整端到端流程:
1. 文件上传 → 文档处理 → 向量化 → Milvus 存储
2. 用户提问 → 检索 → 答案生成 → 返回结果
"""

import requests
import json
import time
from pathlib import Path
from datetime import datetime
import sys

# API 配置
API_BASE = "http://localhost:8000"

class Colors:
    """终端颜色"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text):
    """打印标题"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(80)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.ENDC}\n")


def print_section(text):
    """打印章节"""
    print(f"\n{Colors.OKBLUE}{Colors.BOLD}{'-' * 80}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}{Colors.BOLD}{text}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}{Colors.BOLD}{'-' * 80}{Colors.ENDC}\n")


def print_success(text):
    """打印成功信息"""
    print(f"{Colors.OKGREEN}✅ {text}{Colors.ENDC}")


def print_error(text):
    """打印错误信息"""
    print(f"{Colors.FAIL}❌ {text}{Colors.ENDC}")


def print_warning(text):
    """打印警告信息"""
    print(f"{Colors.WARNING}⚠️  {text}{Colors.ENDC}")


def print_info(text):
    """打印信息"""
    print(f"{Colors.OKCYAN}ℹ️  {text}{Colors.ENDC}")


class WorkflowTester:
    """工作流测试器"""

    def __init__(self):
        self.api_base = API_BASE
        self.test_results = {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "warnings": 0
        }
        self.uploaded_files = []
        self.test_session_id = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def run_all_tests(self):
        """运行所有测试"""
        print_header("魔搭社区智能答疑系统 - 完整工作流验证测试")
        print_info(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print_info(f"API 地址: {self.api_base}")
        print_info(f"会话 ID: {self.test_session_id}\n")

        try:
            # Phase 1: 系统检查
            self.test_system_health()
            self.test_system_status()

            # Phase 2: 文件上传和向量化
            self.test_file_upload_workflow()

            # Phase 3: 问答流程
            self.test_qa_workflow()

            # Phase 4: 数据验证
            self.test_data_persistence()

            # 打印测试报告
            self.print_test_report()

            return self.test_results["failed"] == 0

        except KeyboardInterrupt:
            print_warning("\n测试被用户中断")
            return False
        except Exception as e:
            print_error(f"测试过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_system_health(self):
        """测试 1: 系统健康检查"""
        print_section("测试 1: 系统健康检查")

        try:
            response = requests.get(f"{self.api_base}/api/health", timeout=5)

            if response.status_code == 200:
                data = response.json()
                print_success(f"系统健康检查通过")
                print_info(f"   状态: {data.get('status')}")
                print_info(f"   时间戳: {data.get('timestamp')}")
                self.test_results["passed"] += 1
            else:
                print_error(f"健康检查失败: HTTP {response.status_code}")
                self.test_results["failed"] += 1

        except requests.exceptions.ConnectionError:
            print_error("无法连接到 API 服务,请确保服务已启动")
            print_info("   启动命令: cd modelscope_qa_agent && python api/main.py")
            self.test_results["failed"] += 1
            raise
        except Exception as e:
            print_error(f"健康检查异常: {e}")
            self.test_results["failed"] += 1

        self.test_results["total_tests"] += 1

    def test_system_status(self):
        """测试 2: 系统状态检查"""
        print_section("测试 2: 系统状态检查")

        try:
            response = requests.get(f"{self.api_base}/api/status", timeout=10)

            if response.status_code == 200:
                data = response.json()
                print_success("系统状态检查通过")
                print_info(f"   状态: {data['status']}")
                print_info(f"   Milvus 连接: {data['milvus_connected']}")
                print_info(f"   文档数量: {data['document_count']}")
                print_info(f"   向量维度: {data['vector_dim']}")
                print_info(f"   存储类型: {data['storage_type']}")
                print_info(f"   AI 提供商: {data['ai_provider']}")

                # 验证关键组件
                if not data['milvus_connected']:
                    print_warning("Milvus 未连接,向量存储功能可能不可用")
                    self.test_results["warnings"] += 1

                self.test_results["passed"] += 1
            else:
                print_error(f"状态检查失败: HTTP {response.status_code}")
                error = response.json()
                print_error(f"   错误信息: {error.get('detail')}")
                self.test_results["failed"] += 1

        except Exception as e:
            print_error(f"状态检查异常: {e}")
            self.test_results["failed"] += 1

        self.test_results["total_tests"] += 1

    def test_file_upload_workflow(self):
        """测试 3: 文件上传和向量化工作流"""
        print_section("测试 3: 文件上传和向量化工作流")

        # 创建测试文件
        test_files_dir = Path("/tmp/workflow_test_files")
        test_files_dir.mkdir(parents=True, exist_ok=True)

        # 测试文件列表
        test_files = {
            "test_qwen.md": """# Qwen 模型使用指南

## 简介
Qwen 是阿里巴巴开发的大规模语言模型,支持多种自然语言处理任务。

## 安装
```bash
pip install dashscope
```

## 基本使用
```python
from dashscope import Generation

response = Generation.call(
    model='qwen-turbo',
    prompt='你好,请介绍一下自己'
)
print(response.output.text)
```

## 参数说明
- model: 模型名称
- prompt: 输入文本
- temperature: 温度参数 (0-1)
- top_p: 核采样参数

## 高级功能
### 1. 流式输出
使用 stream=True 参数可以实现流式输出

### 2. Few-shot Learning
通过示例引导模型生成更准确的结果

### 3. Function Calling
支持函数调用,可以集成外部工具
""",
            "test_api.txt": """ModelScope API 使用文档

1. 认证
所有 API 请求需要在 Header 中包含 API Key:
Authorization: Bearer YOUR_API_KEY

2. 端点列表
- /api/models - 获取模型列表
- /api/inference - 推理接口
- /api/finetune - 微调接口

3. 示例代码
curl -X POST https://api.modelscope.cn/api/inference \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -d '{"model": "qwen-turbo", "input": "你好"}'

4. 错误码
- 400: 请求参数错误
- 401: 认证失败
- 429: 请求频率超限
- 500: 服务器错误
""",
            "test_faq.json": json.dumps({
                "faqs": [
                    {
                        "question": "如何获取 API Key?",
                        "answer": "登录 ModelScope 官网,在个人中心 - API 管理中创建新的 API Key"
                    },
                    {
                        "question": "支持哪些编程语言?",
                        "answer": "官方支持 Python、Java、Node.js,社区还有其他语言的 SDK"
                    },
                    {
                        "question": "如何提高推理速度?",
                        "answer": "1) 使用更小的模型 2) 减小 max_tokens 3) 使用批量推理 4) 选择更近的服务区域"
                    }
                ]
            }, ensure_ascii=False, indent=2)
        }

        # 创建测试文件
        for filename, content in test_files.items():
            file_path = test_files_dir / filename
            file_path.write_text(content, encoding='utf-8')
            print_info(f"创建测试文件: {filename}")

        # 测试每个文件的上传
        initial_doc_count = self._get_document_count()
        print_info(f"上传前文档数量: {initial_doc_count}\n")

        for filename in test_files.keys():
            self._test_single_file_upload(test_files_dir / filename)
            time.sleep(1)  # 避免请求过快

        # 验证文档数量增加
        final_doc_count = self._get_document_count()
        added_count = final_doc_count - initial_doc_count if final_doc_count > 0 else 0

        print_section("文件上传汇总")
        print_info(f"上传前文档数: {initial_doc_count}")
        print_info(f"上传后文档数: {final_doc_count}")
        print_info(f"新增文档数: {added_count}")

        if added_count > 0:
            print_success(f"成功上传并向量化了 {added_count} 个文档块")
            self.test_results["passed"] += 1
        else:
            print_warning("文档上传成功但未增加文档数,可能未启用 store_to_db")
            self.test_results["warnings"] += 1

        self.test_results["total_tests"] += 1

    def _test_single_file_upload(self, file_path):
        """测试单个文件上传"""
        print_info(f"\n上传文件: {file_path.name}")

        try:
            with open(file_path, 'rb') as f:
                files = {'file': (file_path.name, f, 'application/octet-stream')}
                data = {
                    'category': 'test',
                    'store_to_db': 'true'  # 确保存储到数据库
                }

                response = requests.post(
                    f"{self.api_base}/api/upload",
                    files=files,
                    data=data,
                    timeout=60
                )

                if response.status_code == 200:
                    result = response.json()
                    print_success(f"  上传成功: {result['message']}")
                    print_info(f"    文件大小: {result['file_size']} 字节")
                    print_info(f"    文档块数: {result['document_count']}")
                    print_info(f"    存储到DB: {result['stored_to_db']}")

                    if result['stored_to_db'] and result.get('document_ids'):
                        print_info(f"    文档 IDs: {result['document_ids'][:3]}..." if len(result['document_ids']) > 3 else f"    文档 IDs: {result['document_ids']}")

                    self.uploaded_files.append({
                        "filename": file_path.name,
                        "document_count": result['document_count'],
                        "document_ids": result.get('document_ids', [])
                    })

                else:
                    error = response.json()
                    print_error(f"  上传失败: {error.get('detail')}")

        except Exception as e:
            print_error(f"  上传异常: {e}")

    def test_qa_workflow(self):
        """测试 4: 问答工作流"""
        print_section("测试 4: 问答工作流")

        # 测试问题列表
        test_questions = [
            {
                "question": "如何使用 Qwen 模型?",
                "expected_keywords": ["Qwen", "模型", "使用", "安装", "pip"],
                "description": "测试基本问答功能"
            },
            {
                "question": "ModelScope API 的认证方式是什么?",
                "expected_keywords": ["API", "Key", "Authorization", "Bearer"],
                "description": "测试 API 文档检索"
            },
            {
                "question": "如何提高推理速度?",
                "expected_keywords": ["模型", "max_tokens", "批量", "推理"],
                "description": "测试 FAQ 检索"
            }
        ]

        qa_success_count = 0

        for i, test_case in enumerate(test_questions, 1):
            print_info(f"\n问题 {i}: {test_case['question']}")
            print_info(f"描述: {test_case['description']}")

            try:
                response = requests.post(
                    f"{self.api_base}/api/question",
                    headers={'Content-Type': 'application/json'},
                    json={
                        "question": test_case['question'],
                        "session_id": self.test_session_id,
                        "top_k": 3
                    },
                    timeout=60
                )

                if response.status_code == 200:
                    result = response.json()
                    answer = result['answer']
                    sources = result['sources']
                    confidence = result['confidence']

                    print_success("  问答成功")
                    print_info(f"    置信度: {confidence:.2%}")
                    print_info(f"    来源数: {len(sources)}")
                    print_info(f"    答案长度: {len(answer)} 字符")

                    # 显示答案前200个字符
                    answer_preview = answer[:200] + "..." if len(answer) > 200 else answer
                    print(f"\n    {Colors.OKCYAN}答案预览:{Colors.ENDC}")
                    for line in answer_preview.split('\n'):
                        print(f"    {line}")

                    # 显示来源
                    if sources:
                        print(f"\n    {Colors.OKCYAN}来源文档:{Colors.ENDC}")
                        for j, source in enumerate(sources, 1):
                            print(f"    {j}. {source['source']} (相似度: {source['score']:.2%})")

                    # 检查关键词
                    answer_lower = answer.lower()
                    found_keywords = [kw for kw in test_case['expected_keywords'] if kw.lower() in answer_lower]

                    if found_keywords:
                        print_info(f"    找到关键词: {', '.join(found_keywords)}")
                        qa_success_count += 1
                    else:
                        print_warning(f"    未找到预期关键词: {', '.join(test_case['expected_keywords'])}")

                else:
                    error = response.json()
                    print_error(f"  问答失败: {error.get('detail')}")

            except Exception as e:
                print_error(f"  问答异常: {e}")

            time.sleep(2)  # 避免请求过快

        # 汇总问答测试结果
        print_section("问答测试汇总")
        print_info(f"总问题数: {len(test_questions)}")
        print_info(f"成功回答: {qa_success_count}")
        print_info(f"成功率: {qa_success_count/len(test_questions):.1%}")

        if qa_success_count >= len(test_questions) * 0.6:  # 60% 通过率
            print_success("问答测试通过")
            self.test_results["passed"] += 1
        else:
            print_warning("问答测试部分通过,成功率低于预期")
            self.test_results["warnings"] += 1

        self.test_results["total_tests"] += 1

    def test_data_persistence(self):
        """测试 5: 数据持久性验证"""
        print_section("测试 5: 数据持久性验证")

        try:
            # 再次获取系统状态
            response = requests.get(f"{self.api_base}/api/status", timeout=10)

            if response.status_code == 200:
                data = response.json()
                doc_count = data['document_count']

                print_success("数据持久性验证通过")
                print_info(f"   当前文档总数: {doc_count}")
                print_info(f"   Milvus 连接状态: {data['milvus_connected']}")
                print_info(f"   向量维度: {data['vector_dim']}")

                if doc_count > 0 and data['milvus_connected']:
                    print_success("✅ 文档已成功存储到 Milvus 向量数据库")
                    self.test_results["passed"] += 1
                else:
                    print_warning("文档可能未正确存储")
                    self.test_results["warnings"] += 1
            else:
                print_error("数据持久性验证失败")
                self.test_results["failed"] += 1

        except Exception as e:
            print_error(f"数据持久性验证异常: {e}")
            self.test_results["failed"] += 1

        self.test_results["total_tests"] += 1

    def _get_document_count(self):
        """获取当前文档数量"""
        try:
            response = requests.get(f"{self.api_base}/api/status", timeout=10)
            if response.status_code == 200:
                return response.json()['document_count']
        except:
            pass
        return 0

    def print_test_report(self):
        """打印测试报告"""
        print_header("测试报告")

        total = self.test_results["total_tests"]
        passed = self.test_results["passed"]
        failed = self.test_results["failed"]
        warnings = self.test_results["warnings"]

        print(f"{Colors.BOLD}测试统计:{Colors.ENDC}")
        print(f"  总测试数: {total}")
        print(f"  {Colors.OKGREEN}通过: {passed}{Colors.ENDC}")
        print(f"  {Colors.FAIL}失败: {failed}{Colors.ENDC}")
        print(f"  {Colors.WARNING}警告: {warnings}{Colors.ENDC}")

        if total > 0:
            pass_rate = (passed / total) * 100
            print(f"\n  通过率: {pass_rate:.1f}%")

        print(f"\n{Colors.BOLD}上传文件汇总:{Colors.ENDC}")
        for file_info in self.uploaded_files:
            print(f"  📄 {file_info['filename']}: {file_info['document_count']} 个文档块")

        print(f"\n{Colors.BOLD}工作流验证结果:{Colors.ENDC}")

        if failed == 0:
            if warnings == 0:
                print(f"{Colors.OKGREEN}{Colors.BOLD}✅ 所有测试通过! 工作流运行正常!{Colors.ENDC}")
            else:
                print(f"{Colors.WARNING}{Colors.BOLD}⚠️  测试通过但有 {warnings} 个警告{Colors.ENDC}")
        else:
            print(f"{Colors.FAIL}{Colors.BOLD}❌ {failed} 个测试失败{Colors.ENDC}")

        print(f"\n{Colors.BOLD}完整工作流说明:{Colors.ENDC}")
        print("  1. 文件上传 → 文档处理 → 清洗分块 → 质量评分")
        print("  2. 向量化(Embedding) → Milvus 存储")
        print("  3. 用户提问 → 向量检索 → LLM 生成答案")
        print("  4. 返回答案 + 来源 + 置信度")

        print(f"\n{Colors.OKCYAN}详细文档: {Colors.ENDC}modelscope_qa_agent/WORKFLOW.md")


def main():
    """主函数"""
    tester = WorkflowTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
