"""
端到端用户工作流测试

测试完整的用户旅程：
1. 用户登录
2. 创建雷达模型
3. 创建场景
4. 启动仿真
5. 查看结果
"""

import pytest
import asyncio
from httpx import AsyncClient


@pytest.mark.asyncio
@pytest.mark.e2e
class TestUserCompleteWorkflow:
    """完整用户工作流测试"""

    async def test_new_user_complete_journey(self, client: AsyncClient):
        """测试新用户完整旅程"""

        # ========== 步骤1: 管理员登录 ==========
        login_response = await client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "admin123"}
        )
        assert login_response.status_code == 200
        admin_token = login_response.json()["access_token"]
        admin_headers = {"Authorization": f"Bearer {admin_token}"}

        # ========== 步骤2: 注册新用户 ==========
        new_user_data = {
            "id": "workflow-test-user",
            "username": "workflowuser",
            "password": "workflowpass123",
            "email": "workflow@example.com",
            "role": "user"
        }

        register_response = await client.post(
            "/api/auth/register",
            json=new_user_data,
            headers=admin_headers
        )
        assert register_response.status_code == 200

        # ========== 步骤3: 新用户登录 ==========
        user_login_response = await client.post(
            "/api/auth/login",
            json={"username": "workflowuser", "password": "workflowpass123"}
        )
        assert user_login_response.status_code == 200
        user_token = user_login_response.json()["access_token"]
        user_headers = {"Authorization": f"Bearer {user_token}"}

        # 验证能获取用户信息
        me_response = await client.get("/api/auth/me", headers=user_headers)
        assert me_response.status_code == 200
        user_info = me_response.json()
        assert user_info["username"] == "workflowuser"

        # ========== 步骤4: 创建雷达模型 ==========
        radar_data = {
            "id": "workflow-radar-1",
            "name": "工作流测试雷达",
            "description": "用于端到端测试的雷达模型",
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
                "num_elements_v": 10
            },
            "signal_processing": {
                "mti_enabled": True,
                "mtd_enabled": True,
                "cfar_enabled": True
            }
        }

        radar_response = await client.post(
            "/api/radars",
            json=radar_data,
            headers=user_headers
        )
        assert radar_response.status_code in [200, 201]
        radar_id = radar_data["id"]

        # 验证雷达模型已创建
        get_radar_response = await client.get(
            f"/api/radars/{radar_id}",
            headers=user_headers
        )
        assert get_radar_response.status_code == 200
        radar = get_radar_response.json()
        assert radar["name"] == "工作流测试雷达"

        # ========== 步骤5: 创建场景 ==========
        scene_data = {
            "id": "workflow-scene-1",
            "name": "工作流测试场景",
            "description": "用于端到端测试的场景",
            "radar_model_id": radar_id,
            "duration": 50,
            "time_step": 0.1,
            "environment": {
                "temperature": 20,
                "pressure": 101325,
                "humidity": 50
            },
            "targets": [
                {
                    "id": "target-1",
                    "name": "测试目标",
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
                "density": 0.05
            }
        }

        scene_response = await client.post(
            "/api/scenes",
            json=scene_data,
            headers=user_headers
        )
        assert scene_response.status_code in [200, 201]
        scene_id = scene_data["id"]

        # 验证场景已创建
        get_scene_response = await client.get(
            f"/api/scenes/{scene_id}",
            headers=user_headers
        )
        assert get_scene_response.status_code == 200
        scene = get_scene_response.json()
        assert scene["name"] == "工作流测试场景"
        assert len(scene["targets"]) == 1

        # ========== 步骤6: 列出用户的资源 ==========
        # 列出雷达模型
        radars_list_response = await client.get(
            "/api/radars",
            headers=user_headers
        )
        assert radars_list_response.status_code == 200
        radars = radars_list_response.json()
        assert len(radars) >= 1
        assert any(r["id"] == radar_id for r in radars)

        # 列出场景
        scenes_list_response = await client.get(
            "/api/scenes",
            headers=user_headers
        )
        assert scenes_list_response.status_code == 200
        scenes = scenes_list_response.json()
        assert len(scenes) >= 1
        assert any(s["id"] == scene_id for s in scenes)

        # ========== 步骤7: 启动仿真 ==========
        sim_data = {
            "scene_id": scene_id,
            "save_results": True,
            "realtime": False
        }

        sim_response = await client.post(
            "/api/sim/start",
            json=sim_data,
            headers=user_headers
        )
        assert sim_response.status_code == 200
        sim_result = sim_response.json()
        assert "simulation_id" in sim_result
        simulation_id = sim_result["simulation_id"]

        # ========== 步骤8: 检查仿真状态 ==========
        # 等待一下让仿真开始
        await asyncio.sleep(0.5)

        status_response = await client.get(
            f"/api/sim/status?simulation_id={simulation_id}",
            headers=user_headers
        )
        assert status_response.status_code == 200
        status = status_response.json()
        assert status["simulation_id"] == simulation_id
        assert status["status"] in ["running", "completed", "pending"]

        # ========== 步骤9: 列出仿真 ==========
        list_sim_response = await client.get(
            "/api/sim/list",
            headers=user_headers
        )
        assert list_sim_response.status_code == 200
        simulations = list_sim_response.json()
        assert len(simulations) >= 1

        # ========== 步骤10: 清理（可选）==========
        # 删除仿真
        delete_sim_response = await client.delete(
            f"/api/sim/{simulation_id}",
            headers=user_headers
        )
        assert delete_sim_response.status_code in [200, 204]

        # 删除场景
        delete_scene_response = await client.delete(
            f"/api/scenes/{scene_id}",
            headers=user_headers
        )
        assert delete_scene_response.status_code in [200, 204]

        # 删除雷达模型
        delete_radar_response = await client.delete(
            f"/api/radars/{radar_id}",
            headers=user_headers
        )
        assert delete_radar_response.status_code in [200, 204]

        # 验证删除成功
        verify_radar_response = await client.get(
            f"/api/radars/{radar_id}",
            headers=user_headers
        )
        assert verify_radar_response.status_code == 404


@pytest.mark.asyncio
@pytest.mark.e2e
class TestAdminWorkflow:
    """管理员工作流测试"""

    async def test_admin_manages_users(self, client: AsyncClient):
        """测试管理员管理用户"""
        # 管理员登录
        login_response = await client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "admin123"}
        )
        admin_token = login_response.json()["access_token"]
        admin_headers = {"Authorization": f"Bearer {admin_token}"}

        # 创建用户
        user_data = {
            "id": "managed-user-1",
            "username": "manageduser",
            "password": "managedpass123",
            "email": "managed@example.com",
            "role": "user"
        }

        create_response = await client.post(
            "/api/auth/register",
            json=user_data,
            headers=admin_headers
        )
        assert create_response.status_code == 200

        # 列出用户
        users_response = await client.get(
            "/api/auth/users",
            headers=admin_headers
        )
        assert users_response.status_code == 200
        users = users_response.json()["users"]
        assert any(u["username"] == "manageduser" for u in users)

    async def test_admin_can_access_all_resources(self, client: AsyncClient):
        """测试管理员可以访问所有资源"""
        # 管理员登录
        login_response = await client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "admin123"}
        )
        admin_token = login_response.json()["access_token"]
        admin_headers = {"Authorization": f"Bearer {admin_token}"}

        # 创建用户的雷达模型
        user_radar = {
            "id": "admin-test-radar",
            "name": "管理员测试雷达",
            "transmitter": {"frequency": 10e9, "bandwidth": 20e6, "power": 10000.0}
        }
        await client.post(
            "/api/radars",
            json=user_radar,
            headers=admin_headers
        )

        # 管理员应该能列出所有资源
        response = await client.get("/api/radars", headers=admin_headers)
        assert response.status_code == 200
        radars = response.json()
        assert len(radars) >= 1


@pytest.mark.asyncio
@pytest.mark.e2e
class TestErrorRecoveryWorkflow:
    """错误恢复工作流测试"""

    async def test_recover_from_invalid_radar_creation(self, client: AsyncClient):
        """测试从无效雷达创建中恢复"""
        # 登录
        login_response = await client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "admin123"}
        )
        admin_headers = {"Authorization": f"Bearer {login_response.json()['access_token']}"}

        # 尝试创建无效雷达
        invalid_radar = {
            "id": "invalid-radar",
            "name": "无效雷达"
            # 缺少transmitter
        }

        invalid_response = await client.post(
            "/api/radars",
            json=invalid_radar,
            headers=admin_headers
        )
        assert invalid_response.status_code == 422

        # 立即创建有效雷达
        valid_radar = {
            "id": "valid-radar-after-error",
            "name": "有效雷达",
            "transmitter": {"frequency": 10e9, "bandwidth": 20e6, "power": 10000.0}
        }

        valid_response = await client.post(
            "/api/radars",
            json=valid_radar,
            headers=admin_headers
        )
        assert valid_response.status_code in [200, 201]

    async def test_recover_from_simulation_failure(self, client: AsyncClient):
        """测试从仿真失败中恢复"""
        # 登录
        login_response = await client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "admin123"}
        )
        admin_headers = {"Authorization": f"Bearer {login_response.json()['access_token']}"}

        # 创建雷达和场景
        radar = {
            "id": "sim-test-radar",
            "name": "仿真测试雷达",
            "transmitter": {"frequency": 10e9, "bandwidth": 20e6, "power": 10000.0}
        }
        await client.post("/api/radars", json=radar, headers=admin_headers)

        scene = {
            "id": "sim-test-scene",
            "name": "仿真测试场景",
            "radar_model_id": "sim-test-radar",
            "duration": 10,
            "targets": [{
                "id": "t1",
                "type": "aircraft",
                "position": {"x": 5000, "y": 0, "z": 1000},
                "velocity": {"x": 100, "y": 0, "z": 0},
                "rcs": 5
            }]
        }
        await client.post("/api/scenes", json=scene, headers=admin_headers)

        # 尝试使用不存在的场景启动仿真
        invalid_sim_response = await client.post(
            "/api/sim/start",
            json={"scene_id": "nonexistent-scene"},
            headers=admin_headers
        )
        assert invalid_sim_response.status_code in [400, 404]

        # 使用有效场景启动仿真
        valid_sim_response = await client.post(
            "/api/sim/start",
            json={"scene_id": "sim-test-scene"},
            headers=admin_headers
        )
        assert valid_sim_response.status_code == 200
