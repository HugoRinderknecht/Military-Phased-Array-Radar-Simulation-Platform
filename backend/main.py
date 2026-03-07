"""
雷达仿真平台 - 主应用入口
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import json

from app.config import get_settings
from app.api import auth, radar, scene, simulation
from app.utils.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)


# WebSocket连接管理
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"广播消息失败: {e}")


manager = ConnectionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info(f"启动 {settings.app_name} v{settings.app_version}")
    logger.info(f"调试模式: {settings.debug}")

    # 初始化文件管理器（创建必要的目录和文件）
    from app.storage.file_manager import file_manager
    logger.info("文件管理器初始化完成")

    yield

    logger.info("应用关闭")


# 创建FastAPI应用
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="雷达系统仿真平台 - 完整实现信号处理和跟踪算法",
    lifespan=lifespan,
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 全局验证错误处理
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """处理验证错误，记录详细信息"""
    logger.error(f"422验证错误: {json.dumps(exc.errors(), indent=2, ensure_ascii=False)}")
    body = await request.body()
    logger.error(f"请求体: {body.decode('utf-8', errors='replace') if body else 'empty'}")
    logger.error(f"请求URL: {request.url}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )


# 健康检查
@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "app_name": settings.app_name,
        "version": settings.app_version,
    }


# 注册路由
app.include_router(auth.router, prefix="/api")
app.include_router(radar.router, prefix="/api")
app.include_router(scene.router, prefix="/api")
app.include_router(simulation.router, prefix="/api")


# WebSocket端点
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket端点用于实时推送仿真数据

    客户端连接后可订阅仿真更新
    """
    await manager.connect(websocket)
    logger.info(f"WebSocket连接建立: {websocket.client}")

    try:
        while True:
            # 接收客户端消息
            data = await websocket.receive_json()

            if data.get("type") == "subscribe":
                # 订阅仿真
                simulation_id = data.get("simulation_id")
                logger.info(f"客户端订阅仿真: {simulation_id}")

                # 发送确认
                await websocket.send_json({
                    "type": "subscription_confirmed",
                    "simulation_id": simulation_id
                })

            elif data.get("type") == "ping":
                # 心跳
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info(f"WebSocket连接断开: {websocket.client}")
    except Exception as e:
        logger.error(f"WebSocket错误: {e}")
        manager.disconnect(websocket)


# 全局异常处理
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"未处理的异常: {exc}", exc_info=True)
    return HTTPException(status_code=500, detail="内部服务器错误")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info" if settings.debug else "warning",
    )
