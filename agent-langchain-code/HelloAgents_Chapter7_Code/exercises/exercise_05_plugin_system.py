"""
习题 5: 插件系统设计
设计一个可扩展的插件系统架构

本文件实现了一个完整的插件系统,包括:
1. 插件加载机制 (动态导入)
2. 插件生命周期管理 (init, start, stop, cleanup)
3. 插件依赖管理
4. 插件配置系统
5. 插件市场概念
"""

import os
import sys
import importlib
import inspect
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum


# ============================================================================
# 1. 插件状态和元数据
# ============================================================================

class PluginState(Enum):
    """插件状态"""
    UNLOADED = "unloaded"      # 未加载
    LOADED = "loaded"          # 已加载
    INITIALIZED = "initialized" # 已初始化
    STARTED = "started"        # 已启动
    STOPPED = "stopped"        # 已停止
    ERROR = "error"            # 错误状态


@dataclass
class PluginMetadata:
    """插件元数据"""
    name: str                          # 插件名称
    version: str                       # 版本号
    description: str = ""              # 描述
    author: str = ""                   # 作者
    dependencies: List[str] = field(default_factory=list)  # 依赖的其他插件
    tags: List[str] = field(default_factory=list)          # 标签
    config_schema: Dict[str, Any] = field(default_factory=dict)  # 配置模式


# ============================================================================
# 2. 插件基类
# ============================================================================

class Plugin(ABC):
    """
    插件抽象基类
    所有插件必须继承此类并实现相应方法
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化插件

        Args:
            config: 插件配置
        """
        self.config = config or {}
        self.state = PluginState.LOADED
        self._metadata: Optional[PluginMetadata] = None

    @abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """获取插件元数据(必须实现)"""
        pass

    def initialize(self) -> bool:
        """
        初始化插件
        插件加载后调用,用于设置初始状态

        Returns:
            是否成功初始化
        """
        try:
            print(f"  🔧 初始化插件: {self.get_metadata().name}")
            self.state = PluginState.INITIALIZED
            return True
        except Exception as e:
            print(f"  ❌ 初始化失败: {e}")
            self.state = PluginState.ERROR
            return False

    def start(self) -> bool:
        """
        启动插件
        初始化后调用,开始提供功能

        Returns:
            是否成功启动
        """
        try:
            print(f"  ▶️  启动插件: {self.get_metadata().name}")
            self.state = PluginState.STARTED
            return True
        except Exception as e:
            print(f"  ❌ 启动失败: {e}")
            self.state = PluginState.ERROR
            return False

    def stop(self) -> bool:
        """
        停止插件
        停止提供功能,但保留状态

        Returns:
            是否成功停止
        """
        try:
            print(f"  ⏸️  停止插件: {self.get_metadata().name}")
            self.state = PluginState.STOPPED
            return True
        except Exception as e:
            print(f"  ❌ 停止失败: {e}")
            return False

    def cleanup(self):
        """
        清理插件
        释放所有资源
        """
        try:
            print(f"  🧹 清理插件: {self.get_metadata().name}")
            self.state = PluginState.UNLOADED
        except Exception as e:
            print(f"  ❌ 清理失败: {e}")

    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """
        执行插件功能(必须实现)

        Args:
            *args, **kwargs: 插件特定的参数

        Returns:
            插件执行结果
        """
        pass

    def __str__(self) -> str:
        metadata = self.get_metadata()
        return f"Plugin(name={metadata.name}, version={metadata.version}, state={self.state.value})"


# ============================================================================
# 3. 插件管理器
# ============================================================================

class PluginManager:
    """插件管理器 - 负责插件的加载、管理和执行"""

    def __init__(self, plugin_dirs: Optional[List[str]] = None):
        """
        初始化插件管理器

        Args:
            plugin_dirs: 插件目录列表
        """
        self.plugin_dirs = plugin_dirs or []
        self.plugins: Dict[str, Plugin] = {}  # name -> plugin
        self.plugin_instances: Dict[str, Any] = {}  # name -> instance
        print("🔌 插件管理器已初始化")

    def add_plugin_dir(self, dir_path: str):
        """添加插件目录"""
        if dir_path not in self.plugin_dirs:
            self.plugin_dirs.append(dir_path)
            print(f"✅ 添加插件目录: {dir_path}")

    def discover_plugins(self) -> List[str]:
        """
        发现所有可用的插件

        Returns:
            插件文件路径列表
        """
        plugin_files = []

        for plugin_dir in self.plugin_dirs:
            path = Path(plugin_dir)
            if not path.exists():
                print(f"⚠️  插件目录不存在: {plugin_dir}")
                continue

            # 查找所有 .py 文件
            for file in path.glob("*.py"):
                if file.name.startswith("_"):
                    continue
                plugin_files.append(str(file))

        print(f"🔍 发现 {len(plugin_files)} 个插件文件")
        return plugin_files

    def load_plugin(self, plugin_path: str) -> bool:
        """
        加载单个插件

        Args:
            plugin_path: 插件文件路径

        Returns:
            是否成功加载
        """
        try:
            # 动态导入模块
            spec = importlib.util.spec_from_file_location("plugin_module", plugin_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # 查找 Plugin 子类
            plugin_class = None
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and
                    issubclass(obj, Plugin) and
                    obj is not Plugin):
                    plugin_class = obj
                    break

            if not plugin_class:
                print(f"⚠️  未找到插件类: {plugin_path}")
                return False

            # 实例化插件
            plugin_instance = plugin_class()
            metadata = plugin_instance.get_metadata()

            # 检查依赖
            if not self._check_dependencies(metadata):
                print(f"❌ 插件依赖不满足: {metadata.name}")
                return False

            # 注册插件
            self.plugins[metadata.name] = plugin_instance

            print(f"✅ 加载插件: {metadata.name} v{metadata.version}")
            return True

        except Exception as e:
            print(f"❌ 加载插件失败 ({plugin_path}): {e}")
            import traceback
            traceback.print_exc()
            return False

    def _check_dependencies(self, metadata: PluginMetadata) -> bool:
        """检查插件依赖是否满足"""
        for dep in metadata.dependencies:
            if dep not in self.plugins:
                print(f"  ⚠️  缺少依赖: {dep}")
                return False
        return True

    def load_all_plugins(self):
        """加载所有发现的插件"""
        plugin_files = self.discover_plugins()

        for plugin_file in plugin_files:
            self.load_plugin(plugin_file)

        print(f"\n✅ 成功加载 {len(self.plugins)} 个插件")

    def initialize_all(self) -> bool:
        """初始化所有插件"""
        print("\n🔧 初始化所有插件...")
        success = True

        for name, plugin in self.plugins.items():
            if not plugin.initialize():
                success = False

        return success

    def start_all(self) -> bool:
        """启动所有插件"""
        print("\n▶️  启动所有插件...")
        success = True

        for name, plugin in self.plugins.items():
            if not plugin.start():
                success = False

        return success

    def stop_all(self):
        """停止所有插件"""
        print("\n⏸️  停止所有插件...")

        for name, plugin in self.plugins.items():
            plugin.stop()

    def cleanup_all(self):
        """清理所有插件"""
        print("\n🧹 清理所有插件...")

        for name, plugin in self.plugins.items():
            plugin.cleanup()

        self.plugins.clear()

    def get_plugin(self, name: str) -> Optional[Plugin]:
        """获取指定插件"""
        return self.plugins.get(name)

    def execute_plugin(self, name: str, *args, **kwargs) -> Any:
        """
        执行指定插件

        Args:
            name: 插件名称
            *args, **kwargs: 插件参数

        Returns:
            插件执行结果
        """
        plugin = self.get_plugin(name)

        if not plugin:
            raise ValueError(f"插件不存在: {name}")

        if plugin.state != PluginState.STARTED:
            raise RuntimeError(f"插件未启动: {name} (状态: {plugin.state.value})")

        return plugin.execute(*args, **kwargs)

    def list_plugins(self) -> List[Dict[str, Any]]:
        """列出所有插件信息"""
        plugins_info = []

        for name, plugin in self.plugins.items():
            metadata = plugin.get_metadata()
            plugins_info.append({
                "name": metadata.name,
                "version": metadata.version,
                "description": metadata.description,
                "author": metadata.author,
                "state": plugin.state.value,
                "dependencies": metadata.dependencies,
                "tags": metadata.tags,
            })

        return plugins_info

    def __str__(self) -> str:
        return f"PluginManager(plugins={len(self.plugins)}, dirs={self.plugin_dirs})"


# ============================================================================
# 4. 示例插件实现
# ============================================================================

class GreetingPlugin(Plugin):
    """问候插件 - 示例插件1"""

    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="greeting",
            version="1.0.0",
            description="提供多语言问候功能",
            author="LangChain Team",
            tags=["greeting", "i18n"]
        )

    def execute(self, name: str = "World", lang: str = "en") -> str:
        """
        生成问候语

        Args:
            name: 名字
            lang: 语言 (en/zh/es/fr)

        Returns:
            问候语
        """
        greetings = {
            "en": f"Hello, {name}!",
            "zh": f"你好,{name}!",
            "es": f"¡Hola, {name}!",
            "fr": f"Bonjour, {name}!",
        }

        return greetings.get(lang, greetings["en"])


class CalculatorPlugin(Plugin):
    """计算器插件 - 示例插件2"""

    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="calculator",
            version="1.0.0",
            description="提供基本计算功能",
            author="LangChain Team",
            tags=["math", "calculator"]
        )

    def execute(self, expression: str) -> str:
        """
        计算数学表达式

        Args:
            expression: 数学表达式

        Returns:
            计算结果
        """
        try:
            result = eval(expression, {"__builtins__": {}}, {})
            return f"{expression} = {result}"
        except Exception as e:
            return f"计算错误: {e}"


class WeatherPlugin(Plugin):
    """天气插件 - 示例插件3 (有依赖)"""

    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="weather",
            version="1.0.0",
            description="提供天气查询功能",
            author="LangChain Team",
            dependencies=[],  # 实际可能依赖其他插件
            tags=["weather", "api"]
        )

    def initialize(self) -> bool:
        """初始化天气数据"""
        self.weather_data = {
            "北京": {"temp": 25, "condition": "晴"},
            "上海": {"temp": 28, "condition": "多云"},
            "深圳": {"temp": 30, "condition": "小雨"},
        }
        return super().initialize()

    def execute(self, city: str) -> str:
        """
        查询城市天气

        Args:
            city: 城市名称

        Returns:
            天气信息
        """
        if city in self.weather_data:
            data = self.weather_data[city]
            return f"{city}: {data['condition']}, {data['temp']}°C"
        else:
            return f"未找到 {city} 的天气信息"


# ============================================================================
# 5. 演示和测试
# ============================================================================

def demo_basic_plugin_system():
    """演示基本插件系统"""
    print("\n" + "=" * 60)
    print("演示 1: 基本插件系统")
    print("=" * 60)

    # 创建插件管理器
    manager = PluginManager()

    # 手动注册插件
    print("\n📦 注册插件...")
    greeting_plugin = GreetingPlugin()
    calculator_plugin = CalculatorPlugin()
    weather_plugin = WeatherPlugin()

    manager.plugins["greeting"] = greeting_plugin
    manager.plugins["calculator"] = calculator_plugin
    manager.plugins["weather"] = weather_plugin

    # 初始化和启动
    manager.initialize_all()
    manager.start_all()

    # 执行插件
    print("\n🎯 执行插件...")
    print("1. Greeting:")
    print(f"   {manager.execute_plugin('greeting', name='Alice', lang='zh')}")

    print("2. Calculator:")
    print(f"   {manager.execute_plugin('calculator', '2 + 3 * 4')}")

    print("3. Weather:")
    print(f"   {manager.execute_plugin('weather', '北京')}")

    # 列出所有插件
    print("\n📋 插件列表:")
    for info in manager.list_plugins():
        print(f"  - {info['name']} v{info['version']} ({info['state']})")
        print(f"    {info['description']}")

    # 清理
    manager.stop_all()
    manager.cleanup_all()


def demo_plugin_lifecycle():
    """演示插件生命周期"""
    print("\n" + "=" * 60)
    print("演示 2: 插件生命周期")
    print("=" * 60)

    plugin = GreetingPlugin()

    print(f"\n初始状态: {plugin.state.value}")

    print("\n📍 生命周期演示:")
    plugin.initialize()
    print(f"状态: {plugin.state.value}")

    plugin.start()
    print(f"状态: {plugin.state.value}")

    result = plugin.execute("Bob")
    print(f"执行结果: {result}")

    plugin.stop()
    print(f"状态: {plugin.state.value}")

    plugin.cleanup()
    print(f"状态: {plugin.state.value}")


def demo_plugin_dependency():
    """演示插件依赖管理"""
    print("\n" + "=" * 60)
    print("演示 3: 插件依赖管理")
    print("=" * 60)

    class DependentPlugin(Plugin):
        """依赖其他插件的插件"""

        def get_metadata(self) -> PluginMetadata:
            return PluginMetadata(
                name="dependent",
                version="1.0.0",
                description="依赖 greeting 和 calculator",
                dependencies=["greeting", "calculator"]
            )

        def execute(self, *args, **kwargs):
            return "Dependent plugin executed"

    manager = PluginManager()

    # 先加载依赖
    print("\n1. 加载基础插件:")
    manager.plugins["greeting"] = GreetingPlugin()
    manager.plugins["calculator"] = CalculatorPlugin()

    # 再加载依赖插件
    print("\n2. 加载依赖插件:")
    dependent = DependentPlugin()
    metadata = dependent.get_metadata()

    if manager._check_dependencies(metadata):
        manager.plugins["dependent"] = dependent
        print("✅ 依赖检查通过")
    else:
        print("❌ 依赖检查失败")

    manager.cleanup_all()


def create_sample_plugins(plugin_dir: str):
    """创建示例插件文件"""
    os.makedirs(plugin_dir, exist_ok=True)

    # 创建示例插件1
    plugin1_code = '''"""示例插件: 文本转换"""
from exercise_05_plugin_system import Plugin, PluginMetadata

class TextTransformPlugin(Plugin):
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="text_transform",
            version="1.0.0",
            description="文本转换工具",
            author="Example"
        )

    def execute(self, text: str, operation: str = "upper") -> str:
        if operation == "upper":
            return text.upper()
        elif operation == "lower":
            return text.lower()
        elif operation == "title":
            return text.title()
        return text
'''

    plugin1_path = os.path.join(plugin_dir, "text_transform_plugin.py")
    with open(plugin1_path, 'w', encoding='utf-8') as f:
        f.write(plugin1_code)

    print(f"✅ 创建示例插件: {plugin1_path}")


def demo_dynamic_loading():
    """演示动态加载插件"""
    print("\n" + "=" * 60)
    print("演示 4: 动态加载插件")
    print("=" * 60)

    plugin_dir = "/tmp/langchain_plugins"

    # 创建示例插件
    print("\n📝 创建示例插件文件...")
    create_sample_plugins(plugin_dir)

    # 创建管理器并加载
    print("\n🔌 动态加载插件...")
    manager = PluginManager(plugin_dirs=[plugin_dir])
    manager.load_all_plugins()

    if manager.plugins:
        manager.initialize_all()
        manager.start_all()

        # 列出插件
        print("\n📋 已加载的插件:")
        for info in manager.list_plugins():
            print(f"  - {info['name']} v{info['version']}")

        manager.cleanup_all()


if __name__ == "__main__":
    print("=" * 60)
    print("  习题 5: 插件系统架构设计")
    print("=" * 60)

    # 运行演示
    demo_basic_plugin_system()
    demo_plugin_lifecycle()
    demo_plugin_dependency()
    demo_dynamic_loading()

    print("\n" + "=" * 60)
    print("  所有演示完成")
    print("=" * 60)
    print("""
💡 扩展思考:
1. 如何实现插件的热加载和热卸载?
2. 如何实现插件的版本管理和升级?
3. 如何实现插件间的通信机制?
4. 如何实现插件的安全沙箱?
5. 如何设计插件市场,支持插件的发布、下载和评分?
6. 如何实现插件的配置界面?
7. 如何处理插件冲突?

🏗️ 进阶设计:
- 插件优先级系统
- 插件事件系统 (pub-sub)
- 插件权限管理
- 插件性能监控
- 插件文档自动生成
    """)
