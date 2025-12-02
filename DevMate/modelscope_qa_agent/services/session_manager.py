"""
Session Manager
管理用户会话和对话历史的 Redis 存储
"""
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import List, Optional
import json
import uuid
import redis
from config.config_loader import get_config


@dataclass
class ConversationTurn:
    """
    对话轮次数据类
    表示一次完整的问答交互
    """
    question: str
    answer: str
    timestamp: datetime
    sources: Optional[List[dict]] = None
    confidence: Optional[float] = None

    def to_dict(self) -> dict:
        """转换为字典格式"""
        data = asdict(self)
        # 将 datetime 转换为 ISO 格式字符串
        data['timestamp'] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> 'ConversationTurn':
        """从字典创建对象"""
        # 将 ISO 格式字符串转换回 datetime
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class SessionManager:
    """
    会话管理器
    使用 Redis 存储和管理用户会话
    """

    def __init__(self, redis_client: redis.Redis = None):
        """
        初始化会话管理器

        Args:
            redis_client: Redis 客户端实例，如果为 None 则自动创建
        """
        if redis_client is None:
            config = get_config()
            self.redis_client = redis.Redis(
                host=config.redis.host,
                port=config.redis.port,
                db=config.redis.db,
                password=config.redis.password,
                decode_responses=True
            )
        else:
            self.redis_client = redis_client

        self.config = get_config()
        self.session_ttl = self.config.session.ttl
        self.max_sessions_per_user = self.config.session.max_sessions_per_user

    def create_session(self, user_id: Optional[str] = None) -> str:
        """
        创建新会话

        Args:
            user_id: 用户ID，如果为 None 则使用匿名会话

        Returns:
            会话ID
        """
        session_id = str(uuid.uuid4())
        session_key = f"session:{session_id}"

        session_data = {
            "session_id": session_id,
            "user_id": user_id or "anonymous",
            "created_at": datetime.now().isoformat(),
            "last_active": datetime.now().isoformat(),
            "turn_count": 0
        }

        # 保存会话数据
        self.redis_client.hset(session_key, mapping=session_data)
        # 设置过期时间
        self.redis_client.expire(session_key, self.session_ttl)

        # 如果有用户ID，维护用户的会话列表
        if user_id and user_id != "anonymous":
            user_sessions_key = f"user:{user_id}:sessions"
            self.redis_client.sadd(user_sessions_key, session_id)
            # 限制每个用户的最大会话数
            self._cleanup_user_sessions(user_id)

        return session_id

    def get_session(self, session_id: str) -> Optional[dict]:
        """
        获取会话信息

        Args:
            session_id: 会话ID

        Returns:
            会话数据字典，如果不存在则返回 None
        """
        session_key = f"session:{session_id}"
        session_data = self.redis_client.hgetall(session_key)

        if not session_data:
            return None

        # 更新最后活跃时间
        self.redis_client.hset(
            session_key,
            "last_active",
            datetime.now().isoformat()
        )
        # 刷新过期时间
        self.redis_client.expire(session_key, self.session_ttl)

        return session_data

    def delete_session(self, session_id: str) -> bool:
        """
        删除会话

        Args:
            session_id: 会话ID

        Returns:
            是否成功删除
        """
        session_key = f"session:{session_id}"
        history_key = f"session:{session_id}:history"

        # 获取用户ID以便从用户会话列表中移除
        session_data = self.redis_client.hgetall(session_key)
        if session_data and session_data.get("user_id"):
            user_id = session_data["user_id"]
            user_sessions_key = f"user:{user_id}:sessions"
            self.redis_client.srem(user_sessions_key, session_id)

        # 删除会话和历史
        deleted = self.redis_client.delete(session_key, history_key)
        return deleted > 0

    def add_conversation_turn(
        self,
        session_id: str,
        turn: ConversationTurn
    ) -> bool:
        """
        添加对话轮次到会话历史

        Args:
            session_id: 会话ID
            turn: 对话轮次对象

        Returns:
            是否成功添加
        """
        session_key = f"session:{session_id}"
        history_key = f"session:{session_id}:history"

        # 检查会话是否存在
        if not self.redis_client.exists(session_key):
            return False

        # 添加对话轮次到历史列表
        turn_json = json.dumps(turn.to_dict())
        self.redis_client.rpush(history_key, turn_json)

        # 更新会话的轮数计数
        self.redis_client.hincrby(session_key, "turn_count", 1)

        # 更新最后活跃时间
        self.redis_client.hset(
            session_key,
            "last_active",
            datetime.now().isoformat()
        )

        # 刷新过期时间
        self.redis_client.expire(session_key, self.session_ttl)
        self.redis_client.expire(history_key, self.session_ttl)

        return True

    def get_conversation_history(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[ConversationTurn]:
        """
        获取会话的对话历史

        Args:
            session_id: 会话ID
            limit: 限制返回的轮数，None 表示返回全部

        Returns:
            对话轮次列表
        """
        history_key = f"session:{session_id}:history"

        # 获取历史记录
        if limit:
            # 获取最近的 N 条记录
            history_json = self.redis_client.lrange(history_key, -limit, -1)
        else:
            # 获取全部记录
            history_json = self.redis_client.lrange(history_key, 0, -1)

        # 解析 JSON 并转换为 ConversationTurn 对象
        history = []
        for turn_json in history_json:
            turn_dict = json.loads(turn_json)
            history.append(ConversationTurn.from_dict(turn_dict))

        return history

    def _cleanup_user_sessions(self, user_id: str):
        """
        清理用户的旧会话，保持在最大会话数限制内

        Args:
            user_id: 用户ID
        """
        user_sessions_key = f"user:{user_id}:sessions"
        session_ids = self.redis_client.smembers(user_sessions_key)

        if len(session_ids) <= self.max_sessions_per_user:
            return

        # 获取所有会话的最后活跃时间
        sessions_with_time = []
        for session_id in session_ids:
            session_key = f"session:{session_id}"
            last_active = self.redis_client.hget(session_key, "last_active")
            if last_active:
                sessions_with_time.append((session_id, last_active))

        # 按最后活跃时间排序
        sessions_with_time.sort(key=lambda x: x[1])

        # 删除最旧的会话
        to_delete = len(sessions_with_time) - self.max_sessions_per_user
        for i in range(to_delete):
            session_id, _ = sessions_with_time[i]
            self.delete_session(session_id)

    def list_sessions(self, user_id: Optional[str] = None) -> List[dict]:
        """
        列出会话

        Args:
            user_id: 用户ID，如果为 None 则列出所有会话（仅用于管理）

        Returns:
            会话信息列表
        """
        sessions = []

        if user_id:
            # 获取特定用户的会话
            user_sessions_key = f"user:{user_id}:sessions"
            session_ids = self.redis_client.smembers(user_sessions_key)
        else:
            # 获取所有会话 (扫描所有 session:* 键)
            session_keys = []
            cursor = 0
            while True:
                cursor, keys = self.redis_client.scan(cursor, match="session:*", count=100)
                # 过滤出会话键（不包括历史键）
                session_keys.extend([k for k in keys if ":history" not in k])
                if cursor == 0:
                    break

            # 从键中提取 session_id
            session_ids = [key.split("session:")[1] for key in session_keys]

        # 获取每个会话的详细信息
        for session_id in session_ids:
            session_data = self.get_session(session_id)
            if session_data:
                sessions.append(session_data)

        # 按最后活跃时间倒序排序
        sessions.sort(key=lambda x: x.get("last_active", ""), reverse=True)

        return sessions

    def ping(self) -> bool:
        """
        测试 Redis 连接

        Returns:
            连接是否正常
        """
        try:
            return self.redis_client.ping()
        except Exception:
            return False
