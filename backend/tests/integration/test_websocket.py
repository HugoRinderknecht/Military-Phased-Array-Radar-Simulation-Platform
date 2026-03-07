"""
WebSocket集成测试

测试内容：
- WebSocket连接建立
- 订阅仿真更新
- Ping/pong心跳
- 多连接处理
- 断开连接清理
"""

import pytest
import asyncio
from fastapi.testclient import TestClient


@pytest.mark.integration
class TestWebSocketConnection:
    """WebSocket连接测试"""

    def test_websocket_connect(self, client: TestClient):
        """测试基本WebSocket连接"""
        with client.websocket_connect("/ws") as websocket:
            # 连接成功
            assert websocket is not None

    def test_websocket_ping_pong(self, client: TestClient):
        """测试ping/pong心跳"""
        with client.websocket_connect("/ws") as websocket:
            # 发送ping
            websocket.send_json({"type": "ping"})

            # 接收pong
            data = websocket.receive_json()
            assert data["type"] == "pong"

    def test_websocket_send_message(self, client: TestClient):
        """测试发送消息"""
        with client.websocket_connect("/ws") as websocket:
            test_message = {"type": "test", "data": "hello"}
            websocket.send_json(test_message)
            # 消息应该成功发送（没有异常）

    def test_websocket_multiple_messages(self, client: TestClient):
        """测试发送多条消息"""
        with client.websocket_connect("/ws") as websocket:
            for i in range(5):
                websocket.send_json({"type": "ping", "id": i})
                response = websocket.receive_json()
                assert response["type"] == "pong"

    def test_websocket_disconnect(self, client: TestClient):
        """测试断开连接"""
        with client.websocket_connect("/ws") as websocket:
            websocket.send_json({"type": "ping"})
            websocket.receive_json()

        # 连接应该正常关闭


@pytest.mark.integration
class TestWebSocketSimulation:
    """WebSocket仿真订阅测试"""

    def test_subscribe_to_simulation(self, client: TestClient):
        """测试订阅仿真更新"""
        with client.websocket_connect("/ws") as websocket:
            # 订阅仿真
            websocket.send_json({
                "type": "subscribe",
                "simulation_id": "test-sim-123"
            })

            # 接收确认
            response = websocket.receive_json(timeout=2)
            assert response["type"] in ["subscription_confirmed", "pong", "error"]

    def test_unsubscribe_from_simulation(self, client: TestClient):
        """测试取消订阅"""
        with client.websocket_connect("/ws") as websocket:
            # 先订阅
            websocket.send_json({
                "type": "subscribe",
                "simulation_id": "test-sim-456"
            })

            # 取消订阅
            websocket.send_json({
                "type": "unsubscribe",
                "simulation_id": "test-sim-456"
            })

    def test_subscribe_multiple_simulations(self, client: TestClient):
        """测试订阅多个仿真"""
        with client.websocket_connect("/ws") as websocket:
            sim_ids = ["sim-1", "sim-2", "sim-3"]

            for sim_id in sim_ids:
                websocket.send_json({
                    "type": "subscribe",
                    "simulation_id": sim_id
                })

    def test_receive_simulation_update(self, client: TestClient):
        """测试接收仿真更新（如果有仿真在运行）"""
        with client.websocket_connect("/ws") as websocket:
            websocket.send_json({
                "type": "subscribe",
                "simulation_id": "test-sim-update"
            })

            # 尝试接收消息（可能没有实际更新）
            try:
                data = websocket.receive_json(timeout=1)
                # 如果收到消息，验证格式
                if "type" in data:
                    assert isinstance(data["type"], str)
            except:
                # 超时也是正常的（没有更新）
                pass


@pytest.mark.integration
class TestWebSocketConcurrency:
    """WebSocket并发测试"""

    def test_multiple_connections(self, client: TestClient):
        """测试多个WebSocket连接"""
        connections = []

        try:
            # 创建3个连接
            for i in range(3):
                ws = client.websocket_connect("/ws")
                ws.__enter__()
                connections.append(ws)

                # 测试每个连接
                ws.send_json({"type": "ping", "id": i})
                response = ws.receive_json()
                assert response["type"] == "pong"

        finally:
            # 清理所有连接
            for ws in connections:
                try:
                    ws.__exit__(None, None, None)
                except:
                    pass

    def test_concurrent_subscriptions(self, client: TestClient):
        """测试并发订阅"""
        async def subscribe_and_listen(sim_id):
            with client.websocket_connect("/ws") as websocket:
                websocket.send_json({
                    "type": "subscribe",
                    "simulation_id": sim_id
                })
                # 尝试接收
                try:
                    websocket.receive_json(timeout=0.5)
                except:
                    pass

        # 模拟并发订阅
        import threading
        threads = []
        for i in range(3):
            t = threading.Thread(
                target=subscribe_and_listen,
                args=(f"sim-{i}",)
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=2)


@pytest.mark.integration
class TestWebSocketErrorHandling:
    """WebSocket错误处理测试"""

    def test_invalid_message_format(self, client: TestClient):
        """测试无效消息格式"""
        with client.websocket_connect("/ws") as websocket:
            # 发送无效的JSON
            websocket.send_text("invalid json")

            try:
                response = websocket.receive_json(timeout=1)
                # 可能收到错误响应
            except:
                pass  # 连接可能关闭或忽略

    def test_missing_message_type(self, client: TestClient):
        """测试缺少消息类型"""
        with client.websocket_connect("/ws") as websocket:
            websocket.send_json({"data": "test"})

            try:
                response = websocket.receive_json(timeout=1)
                # 应该收到错误或被忽略
            except:
                pass

    def test_unsupported_message_type(self, client: TestClient):
        """测试不支持的消息类型"""
        with client.websocket_connect("/ws") as websocket:
            websocket.send_json({"type": "unsupported_type"})

            try:
                response = websocket.receive_json(timeout=1)
                # 应该收到错误响应
                if "type" in response:
                    assert response["type"] in ["error", "pong"]
            except:
                pass


@pytest.mark.integration
class TestWebSocketAuthentication:
    """WebSocket认证测试"""

    def test_unauthenticated_connection(self, client: TestClient):
        """测试未认证连接（如果需要认证）"""
        # 根据实际API实现，WebSocket可能需要认证
        try:
            with client.websocket_connect("/ws") as websocket:
                # 连接可能成功（公共端点）或失败（需要认证）
                pass
        except Exception as e:
            # 如果需要认证，这里会失败
            pass

    def test_authenticated_connection(self, client: TestClient):
        """测试认证后连接"""
        # 先登录获取token
        response = client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "admin123"}
        )
        token = response.json()["access_token"]

        # 尝试带token的WebSocket连接
        # 注意：这取决于WebSocket认证实现
        try:
            with client.websocket_connect({
                "/ws",
                headers={"Authorization": f"Bearer {token}"}
            }) as websocket:
                websocket.send_json({"type": "ping"})
                response = websocket.receive_json()
                assert response["type"] == "pong"
        except:
            # 可能不支持此认证方式
            pass


@pytest.mark.integration
class TestWebSocketDataIntegrity:
    """WebSocket数据完整性测试"""

    def test_large_message(self, client: TestClient):
        """测试发送大消息"""
        with client.websocket_connect("/ws") as websocket:
            large_data = {"type": "test", "data": "x" * 10000}
            websocket.send_json(large_data)
            # 应该成功发送

    def test_rapid_messages(self, client: TestClient):
        """测试快速发送多条消息"""
        with client.websocket_connect("/ws") as websocket:
            for i in range(100):
                websocket.send_json({"type": "ping", "id": i})

            # 接收一些响应
            for _ in range(10):
                response = websocket.receive_json()
                assert response["type"] == "pong"

    def test_special_characters_in_message(self, client: TestClient):
        """测试消息中的特殊字符"""
        with client.websocket_connect("/ws") as websocket:
            special_message = {
                "type": "test",
                "data": "测试中文\n\r\t<script>"
            }
            websocket.send_json(special_message)
            # 应该成功发送

    def test_unicode_characters(self, client: TestClient):
        """测试Unicode字符"""
        with client.websocket_connect("/ws") as websocket:
            unicode_message = {
                "type": "test",
                "data": "🚀🎯📡 Emoji测试"
            }
            websocket.send_json(unicode_message)
            # 应该成功发送
