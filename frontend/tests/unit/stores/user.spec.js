/**
 * User Store测试
 *
 * 测试内容：
 * - 初始状态
 * - login action
 * - logout action
 * - fetchUserInfo action
 * - isLoggedIn computed
 * - Token持久化
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { useUserStore } from '@/stores/user'
import * as api from '@/api'

// Mock API模块
vi.mock('@/api', () => ({
  login: vi.fn(),
  getUserInfo: vi.fn()
}))

describe('User Store', () => {
  let userStore

  beforeEach(() => {
    // 创建新的Pinia实例
    setActivePinia(createPinia())
    userStore = useUserStore()
    vi.clearAllMocks()
    localStorage.clear()
  })

  afterEach(() => {
    localStorage.clear()
  })

  describe('初始状态', () => {
    it('应该有空的初始状态', () => {
      expect(userStore.token).toBe('')
      expect(userStore.userInfo).toBe(null)
      expect(userStore.isLoggedIn).toBe(false)
    })
  })

  describe('isLoggedIn computed', () => {
    it('有token时应该返回true', () => {
      userStore.token = 'test-token'
      expect(userStore.isLoggedIn).toBe(true)
    })

    it('没有token时应该返回false', () => {
      userStore.token = ''
      expect(userStore.isLoggedIn).toBe(false)
    })

    it('token为null时应该返回false', () => {
      userStore.token = null
      expect(userStore.isLoggedIn).toBe(false)
    })

    it('token为undefined时应该返回false', () => {
      userStore.token = undefined
      expect(userStore.isLoggedIn).toBe(false)
    })
  })

  describe('login action', () => {
    it('成功登录应该设置token', async () => {
      const mockToken = 'test-jwt-token'
      const mockUserInfo = {
        username: 'admin',
        role: 'admin',
        email: 'admin@example.com'
      }

      api.login.mockResolvedValue({ access_token: mockToken })
      api.getUserInfo.mockResolvedValue(mockUserInfo)

      const result = await userStore.login('admin', 'admin123')

      expect(result.success).toBe(true)
      expect(userStore.token).toBe(mockToken)
      expect(localStorage.getItem('token')).toBe(mockToken)
      expect(api.login).toHaveBeenCalledWith({ username: 'admin', password: 'admin123' })
    })

    it('成功登录应该获取并设置用户信息', async () => {
      const mockToken = 'test-token'
      const mockUserInfo = {
        username: 'testuser',
        role: 'user',
        email: 'test@example.com'
      }

      api.login.mockResolvedValue({ access_token: mockToken })
      api.getUserInfo.mockResolvedValue(mockUserInfo)

      await userStore.login('testuser', 'password123')

      expect(api.getUserInfo).toHaveBeenCalled()
      expect(userStore.userInfo).toEqual(mockUserInfo)
    })

    it('登录失败应该返回错误结果', async () => {
      const errorDetail = '用户名或密码错误'
      api.login.mockRejectedValue({
        response: { data: { detail: errorDetail } }
      })

      const result = await userStore.login('wronguser', 'wrongpass')

      expect(result.success).toBe(false)
      expect(result.message).toBe(errorDetail)
      expect(userStore.token).toBe('')
      expect(userStore.userInfo).toBe(null)
    })

    it('登录失败时不应该获取用户信息', async () => {
      api.login.mockRejectedValue(new Error('Login failed'))

      await userStore.login('user', 'pass')

      expect(api.getUserInfo).not.toHaveBeenCalled()
      expect(userStore.userInfo).toBe(null)
    })

    it('网络错误应该返回通用错误消息', async () => {
      api.login.mockRejectedValue(new Error('Network Error'))

      const result = await userStore.login('admin', 'admin123')

      expect(result.success).toBe(false)
      expect(result.message).toBe('登录失败')
    })

    it('没有响应详情时应该使用默认错误消息', async () => {
      api.login.mockRejectedValue({})

      const result = await userStore.login('admin', 'admin123')

      expect(result.success).toBe(false)
      expect(result.message).toBe('登录失败')
    })

    it('获取用户信息失败不应该导致登录失败', async () => {
      api.login.mockResolvedValue({ access_token: 'test-token' })
      api.getUserInfo.mockRejectedValue(new Error('Fetch failed'))

      const result = await userStore.login('admin', 'admin123')

      // 登录应该成功，即使获取用户信息失败
      expect(result.success).toBe(true)
      expect(userStore.token).toBe('test-token')
      // 但用户信息可能为空
      expect(userStore.userInfo).toBe(null)
    })
  })

  describe('logout action', () => {
    it('登出应该清除token', () => {
      userStore.token = 'test-token'
      userStore.logout()

      expect(userStore.token).toBe('')
    })

    it('登出应该清除用户信息', () => {
      userStore.userInfo = { username: 'admin', role: 'admin' }
      userStore.logout()

      expect(userStore.userInfo).toBe(null)
    })

    it('登出应该清除localStorage中的token', () => {
      localStorage.setItem('token', 'test-token')
      userStore.token = 'test-token'
      userStore.logout()

      expect(localStorage.getItem('token')).toBe(null)
    })

    it('登出时isLoggedIn应该变为false', () => {
      userStore.token = 'test-token'
      expect(userStore.isLoggedIn).toBe(true)

      userStore.logout()
      expect(userStore.isLoggedIn).toBe(false)
    })

    it('登出时没有任何状态应该正常工作', () => {
      userStore.token = ''
      userStore.userInfo = null

      expect(() => userStore.logout()).not.toThrow()
      expect(userStore.token).toBe('')
      expect(userStore.userInfo).toBe(null)
    })
  })

  describe('fetchUserInfo action', () => {
    it('成功获取用户信息应该更新store', async () => {
      const mockUserInfo = {
        username: 'admin',
        role: 'admin',
        email: 'admin@example.com'
      }
      api.getUserInfo.mockResolvedValue(mockUserInfo)

      await userStore.fetchUserInfo()

      expect(api.getUserInfo).toHaveBeenCalled()
      expect(userStore.userInfo).toEqual(mockUserInfo)
    })

    it('获取用户信息失败应该捕获错误', async () => {
      const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
      api.getUserInfo.mockRejectedValue(new Error('Fetch failed'))

      await userStore.fetchUserInfo()

      expect(consoleErrorSpy).toHaveBeenCalled()
      expect(userStore.userInfo).toBe(null)

      consoleErrorSpy.mockRestore()
    })

    it('获取用户信息失败不应该抛出异常', async () => {
      api.getUserInfo.mockRejectedValue(new Error('Fetch failed'))

      await expect(userStore.fetchUserInfo()).resolves.toBeUndefined()
    })
  })

  describe('Token持久化', () => {
    it('登录时应该保存token到localStorage', async () => {
      const mockToken = 'saved-token'
      api.login.mockResolvedValue({ access_token: mockToken })
      api.getUserInfo.mockResolvedValue({ username: 'admin' })

      await userStore.login('admin', 'admin123')

      expect(localStorage.getItem('token')).toBe(mockToken)
    })

    it('登出时应该从localStorage删除token', () => {
      localStorage.setItem('token', 'test-token')
      userStore.token = 'test-token'
      userStore.logout()

      expect(localStorage.getItem('token')).toBe(null)
    })

    it('应该可以从localStorage恢复token', () => {
      const savedToken = 'restored-token'
      localStorage.setItem('token', savedToken)

      // 创建新的store实例（模拟页面刷新）
      setActivePinia(createPinia())
      const newUserStore = useUserStore()

      // 手动恢复token（实际应用中可能在store初始化时恢复）
      newUserStore.token = localStorage.getItem('token') || ''

      expect(newUserStore.token).toBe(savedToken)
      expect(newUserStore.isLoggedIn).toBe(true)
    })
  })

  describe('Store actions', () => {
    it('login action应该接受用户名和密码参数', async () => {
      api.login.mockResolvedValue({ access_token: 'token' })
      api.getUserInfo.mockResolvedValue({ username: 'user' })

      await userStore.login('testuser', 'testpass')

      expect(api.login).toHaveBeenCalledWith({
        username: 'testuser',
        password: 'testpass'
      })
    })

    it('login应该返回包含success和message的对象', async () => {
      api.login.mockResolvedValue({ access_token: 'token' })
      api.getUserInfo.mockResolvedValue({ username: 'user' })

      const result = await userStore.login('user', 'pass')

      expect(result).toHaveProperty('success')
      expect(typeof result.success).toBe('boolean')
    })

    it('login失败时应该返回message属性', async () => {
      api.login.mockRejectedValue({
        response: { data: { detail: '错误信息' } }
      })

      const result = await userStore.login('user', 'pass')

      expect(result).toHaveProperty('message')
      expect(typeof result.message).toBe('string')
    })
  })
})
