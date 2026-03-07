import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { login as loginApi, getUserInfo } from '@/api'

export const useUserStore = defineStore('user', () => {
  const token = ref('')
  const userInfo = ref(null)

  const isLoggedIn = computed(() => !!token.value)

  async function login(username, password) {
    try {
      const response = await loginApi({ username, password })
      token.value = response.access_token
      localStorage.setItem('token', token.value)

      // 获取用户信息
      await fetchUserInfo()

      return { success: true }
    } catch (error) {
      return {
        success: false,
        message: error.response?.data?.detail || '登录失败'
      }
    }
  }

  async function fetchUserInfo() {
    try {
      const response = await getUserInfo()
      userInfo.value = response
    } catch (error) {
      console.error('获取用户信息失败:', error)
    }
  }

  function logout() {
    token.value = ''
    userInfo.value = null
    localStorage.removeItem('token')
  }

  return {
    token,
    userInfo,
    isLoggedIn,
    login,
    fetchUserInfo,
    logout
  }
})
