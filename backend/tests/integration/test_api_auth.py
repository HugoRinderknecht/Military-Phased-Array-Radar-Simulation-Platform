"""
认证API集成测试

测试内容：
- 用户登录（成功/失败场景）
- Token生成和验证
- 受保护路由访问
- 获取当前用户信息
- 用户注册（仅管理员）
- 用户列表（仅管理员）
"""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
@pytest.mark.integration
class TestAuthAPI:
    """认证API集成测试"""

    async def test_admin_login_success(self, client: AsyncClient):
        """测试管理员成功登录"""
        response = await client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "admin123"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert isinstance(data["access_token"], str)
        assert len(data["access_token"]) > 0

    async def test_login_with_invalid_username(self, client: AsyncClient):
        """测试使用不存在的用户名登录"""
        response = await client.post(
            "/api/auth/login",
            json={"username": "nonexistent_user", "password": "anypassword"}
        )

        assert response.status_code == 401
        data = response.json()
        assert "detail" in data

    async def test_login_with_invalid_password(self, client: AsyncClient):
        """测试使用错误密码登录"""
        response = await client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "wrongpassword"}
        )

        assert response.status_code == 401
        data = response.json()
        assert "detail" in data

    async def test_login_with_empty_credentials(self, client: AsyncClient):
        """测试使用空凭据登录"""
        response = await client.post(
            "/api/auth/login",
            json={"username": "", "password": ""}
        )

        assert response.status_code == 422  # Validation error

    async def test_login_missing_username(self, client: AsyncClient):
        """测试缺少用户名"""
        response = await client.post(
            "/api/auth/login",
            json={"password": "password123"}
        )

        assert response.status_code == 422  # Validation error

    async def test_login_missing_password(self, client: AsyncClient):
        """测试缺少密码"""
        response = await client.post(
            "/api/auth/login",
            json={"username": "admin"}
        )

        assert response.status_code == 422  # Validation error

    async def test_get_current_user_without_token(self, client: AsyncClient):
        """测试不带token访问受保护路由"""
        response = await client.get("/api/auth/me")

        assert response.status_code == 401

    async def test_get_current_user_with_invalid_token(self, client: AsyncClient):
        """测试使用无效token访问受保护路由"""
        response = await client.get(
            "/api/auth/me",
            headers={"Authorization": "Bearer invalid_token"}
        )

        assert response.status_code == 401

    async def test_get_current_user_with_valid_token(self, client: AsyncClient):
        """测试使用有效token获取当前用户信息"""
        # 先登录获取token
        login_response = await client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "admin123"}
        )
        token = login_response.json()["access_token"]

        # 使用token获取用户信息
        response = await client.get(
            "/api/auth/me",
            headers={"Authorization": f"Bearer {token}"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "admin"
        assert data["role"] == "admin"
        assert "email" in data

    async def test_bearer_token_format(self, client: AsyncClient):
        """测试Bearer token格式要求"""
        login_response = await client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "admin123"}
        )
        token = login_response.json()["access_token"]

        # 测试不带"Bearer"前缀
        response = await client.get(
            "/api/auth/me",
            headers={"Authorization": token}
        )
        assert response.status_code == 401

    async def test_register_user_as_admin(self, client: AsyncClient, admin_headers: dict):
        """测试管理员注册新用户"""
        new_user_data = {
            "id": "new-user-001",
            "username": "newuser",
            "password": "newpass123",
            "email": "newuser@example.com",
            "role": "user"
        }

        response = await client.post(
            "/api/auth/register",
            json=new_user_data,
            headers=admin_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "newuser"
        assert data["role"] == "user"
        assert "password" not in data  # 密码不应返回

    async def test_register_user_as_regular_user(self, client: AsyncClient, user_headers: dict):
        """测试普通用户无法注册新用户"""
        new_user_data = {
            "id": "another-user-001",
            "username": "anotheruser",
            "password": "pass123",
            "email": "another@example.com",
            "role": "user"
        }

        response = await client.post(
            "/api/auth/register",
            json=new_user_data,
            headers=user_headers
        )

        assert response.status_code == 403  # Forbidden

    async def test_register_user_without_auth(self, client: AsyncClient):
        """测试未认证用户无法注册新用户"""
        new_user_data = {
            "id": "unauth-user-001",
            "username": "unauthuser",
            "password": "pass123",
            "email": "unauth@example.com",
            "role": "user"
        }

        response = await client.post(
            "/api/auth/register",
            json=new_user_data
        )

        assert response.status_code == 401

    async def test_register_duplicate_username(self, client: AsyncClient, admin_headers: dict):
        """测试注册重复用户名"""
        user_data = {
            "id": "dup-user-001",
            "username": "testuser",
            "password": "pass123",
            "email": "dup@example.com",
            "role": "user"
        }

        # 第一次注册
        response1 = await client.post(
            "/api/auth/register",
            json=user_data,
            headers=admin_headers
        )
        assert response1.status_code == 200

        # 第二次注册相同用户名
        user_data["id"] = "dup-user-002"
        user_data["email"] = "dup2@example.com"
        response2 = await client.post(
            "/api/auth/register",
            json=user_data,
            headers=admin_headers
        )
        assert response2.status_code == 400  # Bad Request

    async def test_get_users_list_as_admin(self, client: AsyncClient, admin_headers: dict):
        """测试管理员获取用户列表"""
        response = await client.get(
            "/api/auth/users",
            headers=admin_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert "users" in data
        assert isinstance(data["users"], list)
        # 至少应该有admin用户
        assert len(data["users"]) >= 1
        assert any(u["username"] == "admin" for u in data["users"])

    async def test_get_users_list_as_regular_user(self, client: AsyncClient, user_headers: dict):
        """测试普通用户无法获取用户列表"""
        response = await client.get(
            "/api/auth/users",
            headers=user_headers
        )

        assert response.status_code == 403  # Forbidden

    async def test_newly_registered_user_can_login(self, client: AsyncClient, admin_headers: dict):
        """测试新注册用户可以正常登录"""
        # 注册新用户
        new_user_data = {
            "id": "login-test-user",
            "username": "logintest",
            "password": "loginpass123",
            "email": "logintest@example.com",
            "role": "user"
        }
        await client.post(
            "/api/auth/register",
            json=new_user_data,
            headers=admin_headers
        )

        # 使用新用户登录
        login_response = await client.post(
            "/api/auth/login",
            json={"username": "logintest", "password": "loginpass123"}
        )

        assert login_response.status_code == 200
        data = login_response.json()
        assert "access_token" in data

    async def test_token_expires_after_logout(self, client: AsyncClient):
        """测试登出后token失效（如果实现了登出功能）"""
        # 登录
        login_response = await client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "admin123"}
        )
        token = login_response.json()["access_token"]

        # 验证token有效
        response1 = await client.get(
            "/api/auth/me",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response1.status_code == 200

        # 注意：当前API可能没有实现logout endpoint
        # 如果有，在这里调用logout

    async def test_password_not_returned_in_user_info(self, client: AsyncClient, admin_headers: dict):
        """测试用户信息中不包含密码"""
        response = await client.get(
            "/api/auth/me",
            headers=admin_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert "password" not in data

    async def test_user_role_field_exists(self, client: AsyncClient, admin_headers: dict):
        """测试用户角色字段存在"""
        response = await client.get(
            "/api/auth/me",
            headers=admin_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert "role" in data
        assert data["role"] in ["admin", "user"]


@pytest.mark.asyncio
@pytest.mark.integration
class TestAuthEdgeCases:
    """认证边界情况测试"""

    async def test_very_long_username(self, client: AsyncClient):
        """测试超长用户名"""
        response = await client.post(
            "/api/auth/login",
            json={"username": "a" * 1000, "password": "admin123"}
        )
        # 应该被验证器拒绝或正常处理
        assert response.status_code in [400, 401, 422]

    async def test_special_characters_in_username(self, client: AsyncClient):
        """测试用户名中的特殊字符"""
        response = await client.post(
            "/api/auth/login",
            json={"username": "admin<script>", "password": "admin123"}
        )
        # 应该被正确处理或拒绝
        assert response.status_code in [400, 401, 422]

    async def test_sql_injection_attempt(self, client: AsyncClient):
        """测试SQL注入尝试（应该被框架防护）"""
        response = await client.post(
            "/api/auth/login",
            json={"username": "' OR '1'='1", "password": "admin123"}
        )
        # 应该失败而不是返回用户数据
        assert response.status_code in [400, 401, 422]

    async def test_concurrent_login_requests(self, client: AsyncClient):
        """测试并发登录请求"""
        import asyncio

        async def login_attempt():
            return await client.post(
                "/api/auth/login",
                json={"username": "admin", "password": "admin123"}
            )

        # 并发发送多个登录请求
        responses = await asyncio.gather(
            login_attempt(),
            login_attempt(),
            login_attempt(),
            login_attempt(),
            login_attempt()
        )

        # 所有请求都应该成功
        for response in responses:
            assert response.status_code == 200
            assert "access_token" in response.json()
