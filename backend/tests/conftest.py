"""
Pytest配置和共享Fixtures

提供测试所需的通用配置和fixtures
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import AsyncGenerator, Generator
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.config import settings
from app.storage.file_manager import FileManager


# ============ pytest配置 ============

def pytest_configure(config):
    """配置pytest标记"""
    config.addinivalue_line("markers", "unit: 单元测试")
    config.addinivalue_line("markers", "integration: 集成测试")
    config.addinivalue_line("markers", "e2e: 端到端测试")
    config.addinivalue_line("markers", "performance: 性能测试")
    config.addinivalue_line("markers", "slow: 慢速测试")


# ============ Event Loop Fixtures ============

@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """创建事件循环用于异步测试"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ============ 临时目录Fixtures ============

@pytest.fixture(scope="function")
def temp_data_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """创建临时数据目录用于测试"""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    users_dir = data_dir / "users"
    users_dir.mkdir()
    materials_file = data_dir / "materials.json"
    users_file = data_dir / "users.json"

    # 创建基础文件
    import json
    from app.storage.file_manager import DEFAULT_MATERIALS

    with open(materials_file, 'w', encoding='utf-8') as f:
        json.dump(DEFAULT_MATERIALS, f, ensure_ascii=False, indent=2)

    with open(users_file, 'w', encoding='utf-8') as f:
        json.dump({"users": []}, f, ensure_ascii=False, indent=2)

    yield data_dir

    # 清理由tmp_path自动处理


# ============ FileManager Fixtures ============

@pytest.fixture(scope="function")
def test_file_manager(temp_data_dir: Path) -> FileManager:
    """创建测试用的FileManager实例"""
    manager = FileManager(data_dir=str(temp_data_dir))
    return manager


@pytest.fixture(scope="function")
def file_manager_with_admin(test_file_manager: FileManager) -> FileManager:
    """创建包含管理员用户的FileManager"""
    from app.utils.auth import hash_password

    # 创建默认管理员用户
    admin_data = {
        "id": "admin-001",
        "username": "admin",
        "password": hash_password("admin123"),
        "email": "admin@example.com",
        "role": "admin",
        "created_at": "2024-01-01T00:00:00"
    }

    users_data = test_file_manager.load_users()
    users_data["users"].append(admin_data)
    test_file_manager.save_users(users_data)

    return test_file_manager


# ============ HTTP Client Fixtures ============

@pytest.fixture(scope="function")
async def client(temp_data_dir: Path) -> AsyncGenerator[AsyncClient, None]:
    """创建异步HTTP客户端用于测试"""
    # 临时修改settings使用测试数据目录
    original_data_dir = settings.data_dir
    settings.data_dir = str(temp_data_dir)

    # 确保管理员用户存在
    test_manager = FileManager(data_dir=str(temp_data_dir))
    from app.utils.auth import hash_password
    from app.storage.file_manager import DEFAULT_MATERIALS

    users_data = test_manager.load_users()
    if not any(u["username"] == "admin" for u in users_data["users"]):
        admin_data = {
            "id": "admin-001",
            "username": "admin",
            "password": hash_password("admin123"),
            "email": "admin@example.com",
            "role": "admin",
            "created_at": "2024-01-01T00:00:00"
        }
        users_data["users"].append(admin_data)
        test_manager.save_users(users_data)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as ac:
        yield ac

    # 恢复原始设置
    settings.data_dir = original_data_dir


# ============ 用户数据Fixtures ============

@pytest.fixture(scope="function")
def admin_user() -> dict:
    """管理员用户数据"""
    return {
        "id": "admin-001",
        "username": "admin",
        "password": "admin123",
        "email": "admin@example.com",
        "role": "admin"
    }


@pytest.fixture(scope="function")
def test_user_data() -> dict:
    """测试用户数据"""
    return {
        "id": "test-user-001",
        "username": "testuser",
        "password": "testpass123",
        "email": "test@example.com",
        "role": "user"
    }


@pytest.fixture(scope="function")
async def admin_headers(client: AsyncClient, admin_user: dict) -> dict:
    """获取管理员认证头"""
    response = await client.post(
        "/api/auth/login",
        json={"username": admin_user["username"], "password": admin_user["password"]}
    )
    assert response.status_code == 200
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(scope="function")
async def user_headers(client: AsyncClient, test_user_data: dict) -> dict:
    """获取普通用户认证头"""
    # 先注册用户（需要管理员权限）
    admin_response = await client.post(
        "/api/auth/login",
        json={"username": "admin", "password": "admin123"}
    )
    admin_token = admin_response.json()["access_token"]

    # 注册新用户
    register_response = await client.post(
        "/api/auth/register",
        json=test_user_data,
        headers={"Authorization": f"Bearer {admin_token}"}
    )

    # 登录新用户
    login_response = await client.post(
        "/api/auth/login",
        json={"username": test_user_data["username"], "password": test_user_data["password"]}
    )
    assert login_response.status_code == 200
    token = login_response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


# ============ 雷达模型Fixtures ============

@pytest.fixture(scope="function")
def valid_radar_data() -> dict:
    """有效的雷达模型数据"""
    return {
        "id": "test-radar-001",
        "name": "测试雷达",
        "description": "测试雷达模型",
        "transmitter": {
            "frequency": 10e9,
            "bandwidth": 20e6,
            "power": 10000.0,
            "gain": 30.0,
            "pulse_width": 1e-6,
            "prf": 1000,
            "waveform_type": "LFM"
        },
        "receiver": {
            "noise_figure": 3.0,
            "bandwidth": 20e6,
            "loss": 2.0
        },
        "antenna": {
            "type": "planar_array",
            "num_elements_h": 20,
            "num_elements_v": 10,
            "element_spacing_h": 0.015,
            "element_spacing_v": 0.015
        },
        "signal_processing": {
            "mti_enabled": True,
            "mtd_enabled": True,
            "cfar_enabled": True,
            "pulse_compression": True
        },
        "scan": {
            "type": "sector",
            "azimuth_start": -45,
            "azimuth_end": 45,
            "elevation_start": 0,
            "elevation_end": 30,
            "scan_rate": 60
        },
        "tracking": {
            "enabled": True,
            "filter_type": "EKF",
            "association_method": "JPDA"
        }
    }


@pytest.fixture(scope="function")
async def created_radar_id(client: AsyncClient, admin_headers: dict, valid_radar_data: dict) -> str:
    """创建雷达模型并返回ID"""
    response = await client.post(
        "/api/radars",
        json=valid_radar_data,
        headers=admin_headers
    )
    assert response.status_code in [200, 201]
    return valid_radar_data["id"]


# ============ 场景Fixtures ============

@pytest.fixture(scope="function")
def valid_scene_data(created_radar_id: str) -> dict:
    """有效的场景数据"""
    return {
        "id": "test-scene-001",
        "name": "测试场景",
        "description": "测试仿真场景",
        "radar_model_id": created_radar_id,
        "duration": 100,
        "time_step": 0.1,
        "environment": {
            "temperature": 20,
            "pressure": 101325,
            "humidity": 50
        },
        "targets": [
            {
                "id": "target-001",
                "name": "目标1",
                "type": "aircraft",
                "position": {"x": 10000, "y": 0, "z": 5000},
                "velocity": {"x": 200, "y": 0, "z": 0},
                "rcs": 10,
                "trajectory_type": "linear"
            }
        ],
        "clutter": {
            "enabled": True,
            "type": "land",
            "distribution": "weibull",
            "density": 0.1
        }
    }


@pytest.fixture(scope="function")
async def created_scene_id(client: AsyncClient, admin_headers: dict, valid_scene_data: dict) -> str:
    """创建场景并返回ID"""
    response = await client.post(
        "/api/scenes",
        json=valid_scene_data,
        headers=admin_headers
    )
    assert response.status_code in [200, 201]
    return valid_scene_data["id"]


# ============ 仿真Fixtures ============

@pytest.fixture(scope="function")
def valid_simulation_data(created_scene_id: str) -> dict:
    """有效的仿真数据"""
    return {
        "scene_id": created_scene_id,
        "save_results": True,
        "realtime": False
    }


# ============ 清理Fixtures ============

@pytest.fixture(scope="function")
async def cleanup_test_data(client: AsyncClient):
    """清理测试数据的fixture"""
    created_ids = {"radars": [], "scenes": [], "simulations": []}

    yield created_ids

    # 清理创建的测试数据
    headers = None
    for sim_id in created_ids["simulations"]:
        if headers:
            await client.delete(f"/api/sim/{sim_id}", headers=headers)

    for scene_id in created_ids["scenes"]:
        if headers:
            await client.delete(f"/api/scenes/{scene_id}", headers=headers)

    for radar_id in created_ids["radars"]:
        if headers:
            await client.delete(f"/api/radars/{radar_id}", headers=headers)
