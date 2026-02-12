# network_manager.py - 网络管理器
"""
本模块实现网络通信管理。

网络通信负责：
- WebSocket服务器
- HTTP服务器
- 会话管理
- 数据发布
"""

import asyncio
import json
from typing import Dict, Set, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.websockets import WebSocketDisconnect

from radar.common.logger import get_logger
from radar.protocol.messages import (
    BaseMessage, PlotDataMessage, TrackDataMessage,
    BeamDataMessage, SimulationStateMessage, SystemStateMessage
)
from radar.protocol.serializer import serialize_message


@dataclass
class ClientSession:
    """
    客户端会话

    Attributes:
        websocket: WebSocket连接
        session_id: 会话ID
        connected_time: 连接时间
        last_heartbeat: 最后心跳时间
        subscriptions: 订阅的消息类型
    """
    websocket: WebSocket
    session_id: str
    connected_time: datetime = field(default_factory=datetime.now)
    last_heartbeat: datetime = field(default_factory=datetime.now)
    subscriptions: Set[str] = field(default_factory=set)


class NetworkManager:
    """
    网络管理器

    管理所有网络连接和数据通信。
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8000):
        """
        初始化网络管理器

        Args:
            host: 服务器地址
            port: 服务器端口
        """
        self._logger = get_logger("network")
        self._host = host
        self._port = port

        # FastAPI应用
        self._app = FastAPI(title="Radar Simulation Backend")

        # 连接管理
        self._sessions: Dict[str, ClientSession] = {}
        self._websocket_clients: Dict[WebSocket, str] = {}

        # 数据队列
        self._plot_queue: asyncio.Queue = None
        self._track_queue: asyncio.Queue = None
        self._beam_queue: asyncio.Queue = None
        self._state_queue: asyncio.Queue = None

        # 回调函数
        self._command_handler = None

        # 统计信息
        self._stats = {
            'total_connections': 0,
            'active_connections': 0,
            'messages_sent': 0,
            'messages_received': 0,
        }

        # 设置路由
        self._setup_routes()

    def _setup_routes(self) -> None:
        """设置HTTP路由"""
        self._logger.info("设置HTTP路由")

        @self._app.get("/")
        async def root():
            """根路径：返回基本信息"""
            return HTMLResponse("""
            <html>
                <head><title>Radar Simulation Backend</title></head>
                <body>
                    <h1>相控阵雷达仿真平台 - 后端</h1>
                    <p>版本: 2.0</p>
                    <p>WebSocket: ws://{host}:{port}/ws</p>
                </body>
            </html>
            """)

        @self._app.get("/api/status")
        async def status():
            """获取系统状态"""
            return JSONResponse({
                'status': 'running',
                'active_sessions': len(self._sessions),
                'statistics': self._stats,
            })

        @self._app.get("/api/config")
        async def get_config():
            """获取配置信息"""
            return JSONResponse({
                'version': '2.0',
                'host': self._host,
                'port': self._port,
            })

        @self._app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket端点"""
            await self._handle_websocket(websocket)

    async def _handle_websocket(self, websocket: WebSocket) -> None:
        """
        处理WebSocket连接

        Args:
            websocket: WebSocket连接对象
        """
        # 生成会话ID
        session_id = f"session_{datetime.now().timestamp()}"

        # 接受连接
        await websocket.accept()

        # 创建会话
        session = ClientSession(
            websocket=websocket,
            session_id=session_id,
            subscriptions=set(['all'])  # 默认订阅所有消息
        )

        self._sessions[session_id] = session
        self._websocket_clients[websocket] = session_id

        # 更新统计
        self._stats['total_connections'] += 1
        self._stats['active_connections'] = len(self._sessions)

        self._logger.info(f"新WebSocket连接: {session_id}")

        try:
            # 启动心跳
            heartbeat_task = asyncio.create_task(
                self._heartbeat_loop(session)
            )

            # 消息处理循环
            await self._message_loop(session)

        except WebSocketDisconnect:
            self._logger.info(f"WebSocket断开: {session_id}")
        except Exception as e:
            self._logger.error(f"WebSocket错误: {e}")
        finally:
            # 清理会话
            await self._cleanup_session(session_id, heartbeat_task)

    async def _message_loop(self, session: ClientSession) -> None:
        """
        消息处理循环

        Args:
            session: 客户端会话
        """
        try:
            while True:
                # 接收消息
                data = await session.websocket.receive()

                # 解析消息
                try:
                    if isinstance(data, str):
                        # JSON格式
                        msg_data = json.loads(data)
                    elif isinstance(data, bytes):
                        # 二进制格式
                        msg_data = json.loads(data.decode('utf-8'))
                    else:
                        self._logger.warning(f"未知消息类型: {type(data)}")
                        continue

                    self._stats['messages_received'] += 1

                    # 处理消息
                    await self._handle_client_message(session, msg_data)

                except json.JSONDecodeError as e:
                    self._logger.error(f"JSON解析失败: {e}")
                except Exception as e:
                    self._logger.error(f"消息处理失败: {e}")

        except WebSocketDisconnect:
            raise
        except Exception as e:
            self._logger.error(f"消息循环错误: {e}")

    async def _handle_client_message(self, session: ClientSession,
                                   msg_data: Dict[str, Any]) -> None:
        """
        处理客户端消息

        Args:
            session: 客户端会话
            msg_data: 消息数据
        """
        msg_type = msg_data.get('type', '')
        self._logger.debug(f"收到消息: {msg_type}")

        # 处理不同类型的消息
        if msg_type == 'command':
            # 控制指令
            if self._command_handler:
                await self._command_handler(msg_data)

        elif msg_type == 'subscribe':
            # 订阅消息
            message_types = msg_data.get('message_types', ['all'])
            session.subscriptions = set(message_types)
            self._logger.info(f"会话订阅: {message_types}")

        elif msg_type == 'unsubscribe':
            # 取消订阅
            message_types = msg_data.get('message_types', [])
            for msg_type in message_types:
                session.subscriptions.discard(msg_type)

        elif msg_type == 'heartbeat':
            # 心跳
            session.last_heartbeat = datetime.now()

        else:
            self._logger.warning(f"未知消息类型: {msg_type}")

    async def _heartbeat_loop(self, session: ClientSession,
                            interval: float = 30.0) -> None:
        """
        心跳循环

        Args:
            session: 客户端会话
            interval: 心跳间隔 [秒]
        """
        try:
            while True:
                await asyncio.sleep(interval)

                # 检查超时
                elapsed = (datetime.now() - session.last_heartbeat).total_seconds()
                if elapsed > 3 * interval:
                    self._logger.warning(f"会话超时: {session.session_id}")
                    break

                # 发送心跳
                heartbeat_msg = {
                    'type': 'heartbeat',
                    'timestamp': int(datetime.now().timestamp() * 1e6),
                    'server_time': datetime.now().isoformat()
                }
                await session.websocket.send_json(heartbeat_msg)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            self._logger.error(f"心跳循环错误: {e}")

    async def _cleanup_session(self, session_id: str,
                             heartbeat_task: asyncio.Task) -> None:
        """
        清理会话

        Args:
            session_id: 会话ID
            heartbeat_task: 心跳任务
        """
        # 取消心跳任务
        heartbeat_task.cancel()

        # 移除会话
        if session_id in self._sessions:
            session = self._sessions[session_id]
            websocket = session.websocket

            if websocket in self._websocket_clients:
                del self._websocket_clients[websocket]

            del self._sessions[session_id]
            self._stats['active_connections'] = len(self._sessions)

            self._logger.info(f"会话清理: {session_id}")

    async def broadcast(self, message: BaseMessage,
                     message_type: str = 'data') -> None:
        """
        广播消息到所有连接的客户端

        Args:
            message: 要广播的消息
            message_type: 消息类型
        """
        if not self._sessions:
            return

        # 序列化消息
        serialized = serialize_message(message, format='json')
        data = json.loads(serialized.decode('utf-8'))
        data['type'] = message_type

        # 发送到所有订阅的客户端
        for session in self._sessions.values():
            # 检查订阅
            if message_type not in session.subscriptions and 'all' not in session.subscriptions:
                continue

            try:
                await session.websocket.send_json(data)
                self._stats['messages_sent'] += 1
            except Exception as e:
                self._logger.warning(f"发送消息失败: {e}")

    async def publish_plots(self, plots: list) -> None:
        """发布点迹数据"""
        from radar.protocol.messages import PlotDataMessage

        msg = PlotDataMessage(plots=plots)
        await self.broadcast(msg, 'plot_update')

    async def publish_tracks(self, tracks: list) -> None:
        """发布航迹数据"""
        from radar.protocol.messages import TrackDataMessage

        msg = TrackDataMessage(tracks=tracks)
        await self.broadcast(msg, 'track_update')

    async def publish_beam(self, beam_info: Any) -> None:
        """发布波束数据"""
        from radar.protocol.messages import BeamDataMessage

        msg = BeamDataMessage(beam_status=beam_info)
        await self.broadcast(msg, 'beam_update')

    def set_command_handler(self, handler: Callable) -> None:
        """设置指令处理器"""
        self._command_handler = handler

    def set_data_queues(self, plot_queue: asyncio.Queue,
                       track_queue: asyncio.Queue,
                       beam_queue: asyncio.Queue) -> None:
        """设置数据队列"""
        self._plot_queue = plot_queue
        self._track_queue = track_queue
        self._beam_queue = beam_queue

    async def start(self) -> None:
        """启动网络服务"""
        import uvicorn

        self._logger.info(f"启动网络服务器: {self._host}:{self._port}")

        config = uvicorn.Config(
            app=self._app,
            host=self._host,
            port=self._port,
            log_level="info"
        )

        server = uvicorn.Server(config)
        await server.serve()

    async def stop(self) -> None:
        """停止网络服务"""
        self._logger.info("停止网络服务器...")

        # 关闭所有连接
        for session in list(self._sessions.values()):
            await session.websocket.close()

        self._sessions.clear()
        self._websocket_clients.clear()

    def get_app(self) -> FastAPI:
        """获取FastAPI应用"""
        return self._app

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self._stats,
            'sessions': len(self._sessions),
        }


__all__ = [
    "ClientSession",
    "NetworkManager",
]
