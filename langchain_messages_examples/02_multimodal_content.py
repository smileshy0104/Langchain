"""
LangChain Messages - 多模态内容示例
演示图像、视频、音频等多模态内容的使用
使用 GLM 模型

注意:
- ChatZhipuAI 支持图像输入(GLM-4V 系列)
- 本示例展示多模态内容的标准格式
- 实际使用时需要确保模型支持对应的模态
"""

from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import HumanMessage, SystemMessage
import base64
import os

os.environ["ZHIPUAI_API_KEY"] = os.getenv("ZHIPUAI_API_KEY", "your-api-key-here")


# ==================== 1. 图像 URL 输入 ====================

def image_url_example():
    """图像 URL 输入示例"""
    print("=" * 60)
    print("图像 URL 输入示例")
    print("=" * 60)

    # GLM-4V 支持图像理解
    model = ChatZhipuAI(model="glm-4v", temperature=0.5)

    # 使用 content blocks 格式
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": "请描述这张图片的内容"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://example.com/image.jpg"
                }
            }
        ]
    )

    print("\n消息结构:")
    print(f"  内容块数量: {len(message.content)}")
    print(f"  文本块: {message.content[0]}")
    print(f"  图像块: {message.content[1]['type']}")

    print("\n注意: 实际调用需要有效的图像 URL 和支持图像的模型")


# ==================== 2. Base64 编码图像 ====================

def image_base64_example():
    """Base64 编码图像示例"""
    print("\n" + "=" * 60)
    print("Base64 编码图像示例")
    print("=" * 60)

    # 示例: 读取本地图像并转换为 base64
    def encode_image_to_base64(image_path: str) -> str:
        """将图像编码为 base64 字符串"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    # 构建包含 base64 图像的消息
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": "分析这张图片"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."  # 实际的 base64 字符串
                }
            }
        ]
    )

    print("\n消息结构:")
    print(f"  图像格式: Base64 编码")
    print(f"  URL 前缀: data:image/jpeg;base64,...")

    print("\n使用场景: 本地图像、临时图像、不方便托管的图像")


# ==================== 3. 多图像输入 ====================

def multiple_images_example():
    """多图像输入示例"""
    print("\n" + "=" * 60)
    print("多图像输入示例")
    print("=" * 60)

    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": "比较这两张图片的区别"
            },
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/image1.jpg"}
            },
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/image2.jpg"}
            }
        ]
    )

    print("\n消息结构:")
    print(f"  内容块总数: {len(message.content)}")
    print(f"  文本块: 1")
    print(f"  图像块: 2")

    for i, block in enumerate(message.content):
        if block.get("type") == "image_url":
            print(f"  图像 {i}: {block['image_url']['url'][:50]}...")


# ==================== 4. 图像详细级别控制 ====================

def image_detail_control():
    """图像详细级别控制示例"""
    print("\n" + "=" * 60)
    print("图像详细级别控制示例")
    print("=" * 60)

    # 低详细度 - 更快、更便宜
    message_low = HumanMessage(
        content=[
            {"type": "text", "text": "快速识别图片内容"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://example.com/image.jpg",
                    "detail": "low"  # 低详细度
                }
            }
        ]
    )

    # 高详细度 - 更准确、更慢
    message_high = HumanMessage(
        content=[
            {"type": "text", "text": "详细分析图片内容"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://example.com/image.jpg",
                    "detail": "high"  # 高详细度
                }
            }
        ]
    )

    print("\n低详细度模式:")
    print("  - 更快的处理速度")
    print("  - 更低的 token 消耗")
    print("  - 适合简单识别任务")

    print("\n高详细度模式:")
    print("  - 更准确的理解")
    print("  - 更多的 token 消耗")
    print("  - 适合复杂分析任务")


# ==================== 5. 文本与图像交错 ====================

def interleaved_text_image():
    """文本与图像交错示例"""
    print("\n" + "=" * 60)
    print("文本与图像交错示例")
    print("=" * 60)

    message = HumanMessage(
        content=[
            {"type": "text", "text": "第一步,看这张图:"},
            {"type": "image_url", "image_url": {"url": "https://example.com/step1.jpg"}},
            {"type": "text", "text": "第二步,再看这张图:"},
            {"type": "image_url", "image_url": {"url": "https://example.com/step2.jpg"}},
            {"type": "text", "text": "请说明两个步骤的区别"}
        ]
    )

    print("\n消息结构 (文本和图像交错):")
    for i, block in enumerate(message.content, 1):
        block_type = block.get("type")
        if block_type == "text":
            print(f"  {i}. 文本: {block['text']}")
        elif block_type == "image_url":
            print(f"  {i}. 图像: {block['image_url']['url'][:40]}...")


# ==================== 6. 视频内容 (标准格式) ====================

def video_content_example():
    """视频内容示例 (标准格式)"""
    print("\n" + "=" * 60)
    print("视频内容示例")
    print("=" * 60)

    message = HumanMessage(
        content=[
            {"type": "text", "text": "请分析这个视频的内容"},
            {
                "type": "video",
                "video": {
                    "url": "https://example.com/video.mp4"
                }
            }
        ]
    )

    print("\n视频消息结构:")
    print(f"  内容类型: video")
    print(f"  视频 URL: {message.content[1]['video']['url']}")

    print("\n注意: 视频支持取决于具体模型")


# ==================== 7. 音频内容 (标准格式) ====================

def audio_content_example():
    """音频内容示例 (标准格式)"""
    print("\n" + "=" * 60)
    print("音频内容示例")
    print("=" * 60)

    message = HumanMessage(
        content=[
            {"type": "text", "text": "请转录这段音频"},
            {
                "type": "audio",
                "audio": {
                    "url": "https://example.com/audio.mp3"
                }
            }
        ]
    )

    print("\n音频消息结构:")
    print(f"  内容类型: audio")
    print(f"  音频 URL: {message.content[1]['audio']['url']}")

    print("\n使用场景: 语音转文字、音频分析")


# ==================== 8. 文档内容 (标准格式) ====================

def document_content_example():
    """文档内容示例 (标准格式)"""
    print("\n" + "=" * 60)
    print("文档内容示例")
    print("=" * 60)

    message = HumanMessage(
        content=[
            {"type": "text", "text": "请总结这份文档"},
            {
                "type": "document",
                "document": {
                    "url": "https://example.com/document.pdf",
                    "mime_type": "application/pdf"
                }
            }
        ]
    )

    print("\n文档消息结构:")
    print(f"  内容类型: document")
    print(f"  文档 URL: {message.content[1]['document']['url']}")
    print(f"  MIME 类型: {message.content[1]['document']['mime_type']}")

    print("\n支持的文档类型: PDF, Word, Excel 等")


# ==================== 9. 混合多模态内容 ====================

def mixed_multimodal_example():
    """混合多模态内容示例"""
    print("\n" + "=" * 60)
    print("混合多模态内容示例")
    print("=" * 60)

    message = HumanMessage(
        content=[
            {"type": "text", "text": "这是一个综合任务:"},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/chart.jpg"}
            },
            {"type": "text", "text": "上图是销售数据图表"},
            {
                "type": "document",
                "document": {
                    "url": "https://example.com/report.pdf",
                    "mime_type": "application/pdf"
                }
            },
            {"type": "text", "text": "这是详细报告,请对比分析"}
        ]
    )

    print("\n混合内容结构:")
    content_types = [block.get("type") for block in message.content]
    print(f"  内容块类型: {content_types}")

    print("\n应用场景: 复杂的多模态分析任务")


# ==================== 10. Content Blocks 最佳实践 ====================

def content_blocks_best_practices():
    """Content Blocks 最佳实践"""
    print("\n" + "=" * 60)
    print("Content Blocks 最佳实践")
    print("=" * 60)

    print("\n1. 使用标准化格式 (推荐)")
    print("   - 使用 content blocks 格式")
    print("   - 明确指定 type 字段")
    print("   - 提供完整的元数据")

    standard_message = HumanMessage(
        content=[
            {"type": "text", "text": "分析图片"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://example.com/image.jpg",
                    "detail": "high"
                }
            }
        ]
    )

    print("\n2. 避免的做法 (不推荐)")
    print("   - 混合字符串和字典格式")
    print("   - 缺少 type 字段")
    print("   - 使用非标准键名")

    print("\n3. 模型兼容性检查")
    print("   - 确认模型支持的模态")
    print("   - GLM-4V: 支持图像")
    print("   - GPT-4V: 支持图像")
    print("   - Gemini: 支持图像、视频、音频")

    print("\n4. 性能优化")
    print("   - 图像: 使用合适的分辨率")
    print("   - 视频: 考虑帧采样")
    print("   - 文档: 提取关键页面")


# ==================== 11. 实用工具函数 ====================

def utility_functions():
    """实用工具函数示例"""
    print("\n" + "=" * 60)
    print("实用工具函数")
    print("=" * 60)

    def create_text_block(text: str) -> dict:
        """创建文本块"""
        return {"type": "text", "text": text}

    def create_image_block(url: str, detail: str = "auto") -> dict:
        """创建图像块"""
        return {
            "type": "image_url",
            "image_url": {"url": url, "detail": detail}
        }

    def create_multimodal_message(text: str, image_urls: list) -> HumanMessage:
        """创建多模态消息"""
        content = [create_text_block(text)]
        for url in image_urls:
            content.append(create_image_block(url))
        return HumanMessage(content=content)

    # 使用工具函数
    message = create_multimodal_message(
        text="分析这些图片",
        image_urls=[
            "https://example.com/img1.jpg",
            "https://example.com/img2.jpg"
        ]
    )

    print("\n使用工具函数创建的消息:")
    print(f"  内容块数量: {len(message.content)}")
    for i, block in enumerate(message.content, 1):
        print(f"  块 {i}: {block.get('type')}")


# ==================== 12. 错误处理和验证 ====================

def error_handling_example():
    """错误处理和验证示例"""
    print("\n" + "=" * 60)
    print("错误处理和验证")
    print("=" * 60)

    def validate_image_url(url: str) -> bool:
        """验证图像 URL"""
        valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
        return any(url.lower().endswith(ext) for ext in valid_extensions)

    def safe_create_image_message(text: str, image_url: str) -> HumanMessage:
        """安全创建图像消息"""
        if not validate_image_url(image_url):
            print(f"  警告: 图像 URL 可能无效: {image_url}")

        return HumanMessage(
            content=[
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        )

    # 测试
    print("\n测试 URL 验证:")

    test_urls = [
        "https://example.com/image.jpg",
        "https://example.com/file.txt",
        "https://example.com/photo.png"
    ]

    for url in test_urls:
        is_valid = validate_image_url(url)
        print(f"  {url}: {'✓ 有效' if is_valid else '✗ 无效'}")


if __name__ == "__main__":
    try:
        image_url_example()
        image_base64_example()
        multiple_images_example()
        image_detail_control()
        interleaved_text_image()
        video_content_example()
        audio_content_example()
        document_content_example()
        mixed_multimodal_example()
        content_blocks_best_practices()
        utility_functions()
        error_handling_example()

        print("\n" + "=" * 60)
        print("所有多模态内容示例完成!")
        print("=" * 60)
        print("\n注意: 实际使用时需要:")
        print("  1. 确保模型支持对应的模态")
        print("  2. 提供有效的 URL 或 Base64 数据")
        print("  3. 注意 token 消耗和成本")

    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()
