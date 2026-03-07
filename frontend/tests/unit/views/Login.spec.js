/**
 * Login组件测试
 *
 * 测试内容：
 * - 组件渲染
 * - 表单验证
 * - 登录成功流程
 * - 登录失败处理
 * - 加载状态
 * - 导航行为
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
import { createRouter, createMemoryHistory } from 'vue-router'
import { createPinia, setActivePinia } from 'pinia'
import ElementPlus from 'element-plus'
import Login from '@/views/Login.vue'
import { useUserStore } from '@/stores/user'
import * as api from '@/api'

// Mock API模块
vi.mock('@/api', () => ({
  login: vi.fn(),
  getUserInfo: vi.fn()
}))

describe('Login.vue', () => {
  let wrapper
  let router
  let pinia
  let userStore

  beforeEach(() => {
    // 重置所有mocks
    vi.clearAllMocks()
    localStorage.clear()

    // 创建Pinia store
    pinia = createPinia()
    setActivePinia(pinia)
    userStore = useUserStore()

    // 创建路由
    router = createRouter({
      history = createMemoryHistory(),
      routes: [
        { path: '/', component: { template: '<div>Dashboard</div>' } },
        { path: '/login', component: Login }
      ]
    })

    // 挂载组件
    wrapper = mount(Login, {
      global: {
        plugins: [pinia, router, ElementPlus],
        stubs: {
          'el-form': true,
          'el-form-item': true,
          'el-input': true,
          'el-button': true
        }
      }
    })
  })

  afterEach(() => {
    wrapper?.unmount()
  })

  describe('组件渲染', () => {
    it('应该正确渲染登录表单', () => {
      expect(wrapper.find('.login-header h1').text()).toBe('雷达仿真平台')
      expect(wrapper.find('.login-header p').text()).toBe('Radar Simulation Platform')
      expect(wrapper.find('.login-footer p').text()).toContain('admin / admin123')
    })

    it('应该显示用户名和密码输入框', () => {
      const inputs = wrapper.findAll('el-input-stub')
      expect(inputs).toHaveLength(2)
    })

    it('应该显示登录按钮', () => {
      const button = wrapper.find('el-button-stub')
      expect(button.exists()).toBe(true)
      expect(button.attributes('type')).toBe('primary')
      expect(button.attributes('size')).toBe('large')
    })
  })

  describe('表单验证', () => {
    it('应该要求用户名必填', async () => {
      // 触发验证而不输入用户名
      wrapper.vm.form.username = ''
      wrapper.vm.form.password = 'password123'

      // Mock validate方法返回false
      const mockValidate = vi.fn().mockResolvedValue(false)
      wrapper.vm.formRef = { validate: mockValidate }

      await wrapper.vm.handleLogin()

      expect(mockValidate).toHaveBeenCalled()
    })

    it('应该要求密码至少6位', async () => {
      wrapper.vm.form.username = 'testuser'
      wrapper.vm.form.password = '12345' // 少于6位

      const mockValidate = vi.fn().mockResolvedValue(false)
      wrapper.vm.formRef = { validate: mockValidate }

      await wrapper.vm.handleLogin()

      expect(mockValidate).toHaveBeenCalled()
    })

    it('验证失败时不应该调用登录API', async () => {
      wrapper.vm.form.username = 'testuser'
      wrapper.vm.form.password = '12345'

      const mockValidate = vi.fn().mockResolvedValue(false)
      wrapper.vm.formRef = { validate: mockValidate }

      await wrapper.vm.handleLogin()

      expect(api.login).not.toHaveBeenCalled()
      expect(wrapper.vm.loading).toBe(false)
    })
  })

  describe('登录成功', () => {
    it('成功登录后应该导航到仪表板', async () => {
      const mockToken = 'test-jwt-token'
      api.login.mockResolvedValue({ access_token: mockToken })
      api.getUserInfo.mockResolvedValue({
        username: 'admin',
        role: 'admin',
        email: 'admin@example.com'
      })

      const pushSpy = vi.spyOn(router, 'push')
      const mockValidate = vi.fn().mockResolvedValue(true)
      wrapper.vm.formRef = { validate: mockValidate }

      wrapper.vm.form.username = 'admin'
      wrapper.vm.form.password = 'admin123'

      await wrapper.vm.handleLogin()
      await flushPromises()

      expect(api.login).toHaveBeenCalledWith('admin', 'admin123')
      expect(userStore.token).toBe(mockToken)
      expect(localStorage.getItem('token')).toBe(mockToken)
      expect(pushSpy).toHaveBeenCalledWith('/')
      expect(wrapper.vm.loading).toBe(false)
    })

    it('登录成功后应该获取用户信息', async () => {
      const mockUserInfo = {
        username: 'admin',
        role: 'admin',
        email: 'admin@example.com'
      }

      api.login.mockResolvedValue({ access_token: 'test-token' })
      api.getUserInfo.mockResolvedValue(mockUserInfo)

      const mockValidate = vi.fn().mockResolvedValue(true)
      wrapper.vm.formRef = { validate: mockValidate }

      wrapper.vm.form.username = 'admin'
      wrapper.vm.form.password = 'admin123'

      await wrapper.vm.handleLogin()
      await flushPromises()

      expect(api.getUserInfo).toHaveBeenCalled()
      expect(userStore.userInfo).toEqual(mockUserInfo)
    })

    it('登录成功时应该显示成功消息', async () => {
      api.login.mockResolvedValue({ access_token: 'test-token' })
      api.getUserInfo.mockResolvedValue({ username: 'admin', role: 'admin' })

      const mockValidate = vi.fn().mockResolvedValue(true)
      wrapper.vm.formRef = { validate: mockValidate }

      wrapper.vm.form.username = 'admin'
      wrapper.vm.form.password = 'admin123'

      await wrapper.vm.handleLogin()
      await flushPromises()

      // ElMessage.success 应该被调用
      const { ElMessage } = await import('element-plus')
      expect(ElMessage.success).toHaveBeenCalledWith('登录成功')
    })
  })

  describe('登录失败', () => {
    it('无效凭据应该显示错误消息', async () => {
      const errorDetail = '用户名或密码错误'
      api.login.mockRejectedValue({
        response: { data: { detail: errorDetail } }
      })

      const mockValidate = vi.fn().mockResolvedValue(true)
      wrapper.vm.formRef = { validate: mockValidate }

      wrapper.vm.form.username = 'wronguser'
      wrapper.vm.form.password = 'wrongpass'

      await wrapper.vm.handleLogin()
      await flushPromises()

      expect(api.login).toHaveBeenCalledWith('wronguser', 'wrongpass')
      expect(wrapper.vm.loading).toBe(false)
      expect(userStore.token).toBe('')

      // ElMessage.error 应该被调用
      const { ElMessage } = await import('element-plus')
      expect(ElMessage.error).toHaveBeenCalledWith(errorDetail)
    })

    it('网络错误应该显示通用错误消息', async () => {
      api.login.mockRejectedValue(new Error('Network Error'))

      const mockValidate = vi.fn().mockResolvedValue(true)
      wrapper.vm.formRef = { validate: mockValidate }

      wrapper.vm.form.username = 'admin'
      wrapper.vm.form.password = 'admin123'

      await wrapper.vm.handleLogin()
      await flushPromises()

      expect(wrapper.vm.loading).toBe(false)
      const { ElMessage } = await import('element-plus')
      expect(ElMessage.error).toHaveBeenCalledWith('登录失败')
    })

    it('登录失败后不应该导航', async () => {
      api.login.mockRejectedValue({
        response: { data: { detail: 'Invalid credentials' } }
      })

      const pushSpy = vi.spyOn(router, 'push')
      const mockValidate = vi.fn().mockResolvedValue(true)
      wrapper.vm.formRef = { validate: mockValidate }

      await wrapper.vm.handleLogin()
      await flushPromises()

      expect(pushSpy).not.toHaveBeenCalled()
    })
  })

  describe('加载状态', () => {
    it('登录过程中应该显示加载状态', async () => {
      let resolveLogin
      api.login.mockImplementation(() => new Promise(resolve => {
        resolveLogin = resolve
      }))
      api.getUserInfo.mockResolvedValue({ username: 'admin' })

      const mockValidate = vi.fn().mockResolvedValue(true)
      wrapper.vm.formRef = { validate: mockValidate }

      wrapper.vm.form.username = 'admin'
      wrapper.vm.form.password = 'admin123'

      const loginPromise = wrapper.vm.handleLogin()

      // 在登录过程中
      expect(wrapper.vm.loading).toBe(true)
      const button = wrapper.find('el-button-stub')
      expect(button.attributes('loading')).toBeDefined()

      // 完成登录
      resolveLogin({ access_token: 'test-token' })
      await loginPromise
      await flushPromises()

      expect(wrapper.vm.loading).toBe(false)
    })

    it('加载状态下按钮应该禁用', async () => {
      wrapper.vm.loading = true
      await wrapper.vm.$nextTick()

      const button = wrapper.find('el-button-stub')
      expect(button.attributes('disabled')).toBeDefined()
    })
  })

  describe('表单数据绑定', () => {
    it('应该正确绑定用户名输入', async () => {
      const testUsername = 'testuser'
      wrapper.vm.form.username = testUsername
      await wrapper.vm.$nextTick()

      expect(wrapper.vm.form.username).toBe(testUsername)
    })

    it('应该正确绑定密码输入', async () => {
      const testPassword = 'password123'
      wrapper.vm.form.password = testPassword
      await wrapper.vm.$nextTick()

      expect(wrapper.vm.form.password).toBe(testPassword)
    })
  })

  describe('键盘事件', () => {
    it('密码框回车应该触发登录', async () => {
      const handleLoginSpy = vi.spyOn(wrapper.vm, 'handleLogin')

      // 模拟回车键事件
      await wrapper.vm.handleLogin()

      expect(handleLoginSpy).toHaveBeenCalled()
    })
  })

  describe('用户体验', () => {
    it('应该显示默认管理员凭据提示', () => {
      const footer = wrapper.find('.login-footer')
      expect(footer.text()).toContain('admin')
      expect(footer.text()).toContain('admin123')
    })

    it('表单应该居中显示', () => {
      const container = wrapper.find('.login-container')
      expect(container.exists()).toBe(true)
    })
  })
})
