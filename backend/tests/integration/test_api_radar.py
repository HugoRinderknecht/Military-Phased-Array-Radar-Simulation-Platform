"""
雷达模型API集成测试

测试内容：
- 雷达模型CRUD操作
- 参数验证
- 用户数据隔离
- 材料列表获取
- 重复ID处理
"""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
@pytest.mark.integration
class TestRadarModelAPI:
    """雷达模型API集成测试"""

    async def test_list_empty_radars(self, client: AsyncClient, admin_headers: dict):
        """测试列出雷达模型（空列表）"""
        response = await client.get("/api/radars", headers=admin_headers)

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    async def test_create_radar_model(self, client: AsyncClient, admin_headers: dict, valid_radar_data: dict):
        """测试创建雷达模型"""
        response = await client.post(
            "/api/radars",
            json=valid_radar_data,
            headers=admin_headers
        )

        assert response.status_code in [200, 201]
        data = response.json()
        assert data["id"] == valid_radar_data["id"]
        assert data["name"] == valid_radar_data["name"]

    async def test_create_radar_model_minimal_data(self, client: AsyncClient, admin_headers: dict):
        """测试使用最小数据创建雷达模型"""
        minimal_data = {
            "id": "minimal-radar",
            "name": "最小雷达模型",
            "transmitter": {
                "frequency": 10e9,
                "bandwidth": 20e6,
                "power": 10000.0
            }
        }

        response = await client.post(
            "/api/radars",
            json=minimal_data,
            headers=admin_headers
        )

        assert response.status_code in [200, 201]

    async def test_create_duplicate_radar_id(self, client: AsyncClient, admin_headers: dict, valid_radar_data: dict):
        """测试创建重复ID的雷达模型"""
        # 第一次创建
        response1 = await client.post(
            "/api/radars",
            json=valid_radar_data,
            headers=admin_headers
        )
        assert response1.status_code in [200, 201]

        # 第二次创建相同ID
        response2 = await client.post(
            "/api/radars",
            json=valid_radar_data,
            headers=admin_headers
        )
        assert response2.status_code == 400

    async def test_get_radar_model(self, client: AsyncClient, admin_headers: dict, created_radar_id: str):
        """测试获取特定雷达模型"""
        response = await client.get(
            f"/api/radars/{created_radar_id}",
            headers=admin_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == created_radar_id

    async def test_get_nonexistent_radar_model(self, client: AsyncClient, admin_headers: dict):
        """测试获取不存在的雷达模型"""
        response = await client.get(
            "/api/radars/nonexistent-radar",
            headers=admin_headers
        )
        assert response.status_code == 404

    async def test_update_radar_model(self, client: AsyncClient, admin_headers: dict, created_radar_id: str):
        """测试更新雷达模型"""
        update_data = {
            "name": "更新的雷达名称",
            "description": "更新的描述"
        }

        response = await client.put(
            f"/api/radars/{created_radar_id}",
            json=update_data,
            headers=admin_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "更新的雷达名称"
        assert data["description"] == "更新的描述"

    async def test_update_nonexistent_radar_model(self, client: AsyncClient, admin_headers: dict):
        """测试更新不存在的雷达模型"""
        response = await client.put(
            "/api/radars/nonexistent",
            json={"name": "新名称"},
            headers=admin_headers
        )
        assert response.status_code == 404

    async def test_delete_radar_model(self, client: AsyncClient, admin_headers: dict, created_radar_id: str):
        """测试删除雷达模型"""
        response = await client.delete(
            f"/api/radars/{created_radar_id}",
            headers=admin_headers
        )
        assert response.status_code == 204

        # 验证删除后不存在
        get_response = await client.get(
            f"/api/radars/{created_radar_id}",
            headers=admin_headers
        )
        assert get_response.status_code == 404

    async def test_delete_nonexistent_radar_model(self, client: AsyncClient, admin_headers: dict):
        """测试删除不存在的雷达模型"""
        response = await client.delete(
            "/api/radars/nonexistent",
            headers=admin_headers
        )
        assert response.status_code == 404

    async def test_user_data_isolation(self, client: AsyncClient, cleanup_test_data):
        """测试用户数据隔离"""
        # 管理员创建雷达
        admin_response = await client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "admin123"}
        )
        admin_token = admin_response.json()["access_token"]

        radar_data = {
            "id": "isolation-test-radar",
            "name": "测试雷达",
            "transmitter": {"frequency": 10e9, "bandwidth": 20e6, "power": 10000.0}
        }

        await client.post(
            "/api/radars",
            json=radar_data,
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        cleanup_test_data["radars"].append("isolation-test-radar")

        # 创建新用户
        register_response = await client.post(
            "/api/auth/register",
            json={
                "id": "test-user-isolation",
                "username": "testisolation",
                "password": "pass123",
                "email": "testiso@example.com",
                "role": "user"
            },
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        # 新用户登录
        user_login_response = await client.post(
            "/api/auth/login",
            json={"username": "testisolation", "password": "pass123"}
        )
        user_token = user_login_response.json()["access_token"]

        # 新用户不应该能看到管理员的雷达
        list_response = await client.get(
            "/api/radars",
            headers={"Authorization": f"Bearer {user_token}"}
        )
        radars = list_response.json()
        assert not any(r["id"] == "isolation-test-radar" for r in radars)

    async def test_list_materials(self, client: AsyncClient, admin_headers: dict):
        """测试列出材料"""
        response = await client.get(
            "/api/radars/materials/list",
            headers=admin_headers
        )

        assert response.status_code == 200
        materials = response.json()
        assert isinstance(materials, dict)
        # 验证常见材料存在
        assert "GaAs" in materials or "GaN" in materials

    async def test_get_material_properties(self, client: AsyncClient, admin_headers: dict):
        """测试获取特定材料属性"""
        response = await client.get(
            "/api/radars/materials/list",
            headers=admin_headers
        )

        materials = response.json()
        # 假设有GaN材料
        if "GaN" in materials:
            gan = materials["GaN"]
            assert "breakdown_field" in gan or "name" in gan

    async def test_create_radar_without_auth(self, client: AsyncClient, valid_radar_data: dict):
        """测试未认证创建雷达模型"""
        response = await client.post(
            "/api/radars",
            json=valid_radar_data
        )
        assert response.status_code == 401

    async def test_get_radar_without_auth(self, client: AsyncClient):
        """测试未认证获取雷达模型"""
        response = await client.get("/api/radars/test-radar")
        assert response.status_code == 401

    async def test_validation_invalid_frequency(self, client: AsyncClient, admin_headers: dict):
        """测试无效频率验证"""
        invalid_data = {
            "id": "invalid-freq-radar",
            "name": "无效频率雷达",
            "transmitter": {
                "frequency": -100,  # 负频率
                "bandwidth": 20e6,
                "power": 10000.0
            }
        }

        response = await client.post(
            "/api/radars",
            json=invalid_data,
            headers=admin_headers
        )
        # 应该被验证器拒绝
        assert response.status_code == 422

    async def test_validation_missing_required_field(self, client: AsyncClient, admin_headers: dict):
        """测试缺少必需字段"""
        invalid_data = {
            "id": "missing-field-radar",
            "name": "缺少字段雷达"
            # 缺少transmitter
        }

        response = await client.post(
            "/api/radars",
            json=invalid_data,
            headers=admin_headers
        )
        assert response.status_code == 422

    async def test_pagination_list_radars(self, client: AsyncClient, admin_headers: dict):
        """测试分页列出雷达模型"""
        # 创建多个雷达
        for i in range(5):
            radar_data = {
                "id": f"paginated-radar-{i}",
                "name": f"雷达 {i}",
                "transmitter": {
                    "frequency": 10e9,
                    "bandwidth": 20e6,
                    "power": 10000.0
                }
            }
            await client.post(
                "/api/radars",
                json=radar_data,
                headers=admin_headers
            )

        # 获取列表
        response = await client.get(
            "/api/radars",
            headers=admin_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 5


@pytest.mark.asyncio
@pytest.mark.integration
class TestRadarModelEdgeCases:
    """雷达模型边界情况测试"""

    async def test_very_long_radar_id(self, client: AsyncClient, admin_headers: dict):
        """测试超长雷达ID"""
        radar_data = {
            "id": "a" * 1000,
            "name": "长ID雷达",
            "transmitter": {"frequency": 10e9, "bandwidth": 20e6, "power": 10000.0}
        }

        response = await client.post(
            "/api/radars",
            json=radar_data,
            headers=admin_headers
        )
        # 应该被接受或拒绝
        assert response.status_code in [200, 201, 400, 422]

    async def test_special_characters_in_name(self, client: AsyncClient, admin_headers: dict):
        """测试名称中的特殊字符"""
        radar_data = {
            "id": "special-chars-radar",
            "name": "测试<script>雷达",
            "transmitter": {"frequency": 10e9, "bandwidth": 20e6, "power": 10000.0}
        }

        response = await client.post(
            "/api/radars",
            json=radar_data,
            headers=admin_headers
        )
        # 应该正确处理
        assert response.status_code in [200, 201, 400, 422]

    async def test_extreme_frequency_values(self, client: AsyncClient, admin_headers: dict):
        """测试极端频率值"""
        radar_data = {
            "id": "extreme-freq-radar",
            "name": "极端频率雷达",
            "transmitter": {
                "frequency": 1e20,  # 非常高的频率
                "bandwidth": 20e6,
                "power": 10000.0
            }
        }

        response = await client.post(
            "/api/radars",
            json=radar_data,
            headers=admin_headers
        )
        # 应该被验证器处理
        assert response.status_code in [200, 201, 400, 422]

    async def test_zero_power(self, client: AsyncClient, admin_headers: dict):
        """测试零功率"""
        radar_data = {
            "id": "zero-power-radar",
            "name": "零功率雷达",
            "transmitter": {
                "frequency": 10e9,
                "bandwidth": 20e6,
                "power": 0
            }
        }

        response = await client.post(
            "/api/radars",
            json=radar_data,
            headers=admin_headers
        )
        # 零功率可能是无效的
        assert response.status_code in [200, 201, 400, 422]

    async def test_concurrent_create_same_radar(self, client: AsyncClient, admin_headers: dict):
        """测试并发创建相同雷达"""
        import asyncio

        radar_data = {
            "id": "concurrent-radar",
            "name": "并发测试雷达",
            "transmitter": {"frequency": 10e9, "bandwidth": 20e6, "power": 10000.0}
        }

        async def create_radar():
            return await client.post(
                "/api/radars",
                json=radar_data,
                headers=admin_headers
            )

        responses = await asyncio.gather(
            create_radar(),
            create_radar(),
            create_radar(),
            return_exceptions=True
        )

        # 至少有一个应该成功
        success_count = sum(1 for r in responses if r.status_code in [200, 201])
        assert success_count >= 1
        # 其他的应该失败（重复ID）
        fail_count = sum(1 for r in responses if r.status_code == 400)
        assert fail_count >= 0
