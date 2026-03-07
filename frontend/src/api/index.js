import axios from 'axios'
import { ElMessage } from 'element-plus'
import router from '@/router'

const api = axios.create({
  baseURL: '/api',
  timeout: 30000
})

// 请求拦截器
api.interceptors.request.use(
  config => {
    const token = localStorage.getItem('token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    // 确保Content-Type正确
    config.headers['Content-Type'] = 'application/json'
    console.log('API请求:', config.method?.toUpperCase(), config.url, config.data)
    return config
  },
  error => {
    return Promise.reject(error)
  }
)

// 响应拦截器
api.interceptors.response.use(
  response => response.data,
  error => {
    if (error.response) {
      switch (error.response.status) {
        case 401:
          ElMessage.error('登录已过期，请重新登录')
          localStorage.removeItem('token')
          router.push('/login')
          break
        case 403:
          ElMessage.error('没有权限访问')
          break
        case 404:
          ElMessage.error('请求的资源不存在')
          break
        case 500:
          ElMessage.error('服务器错误')
          break
        default:
          ElMessage.error(error.response.data?.detail || '请求失败')
      }
    } else {
      ElMessage.error('网络错误')
    }
    return Promise.reject(error)
  }
)

// 认证相关API
export const login = (data) => api.post('/auth/login', data)
export const register = (data) => api.post('/auth/register', data)
export const getUserInfo = () => api.get('/auth/me')
export const getUsers = () => api.get('/auth/users')

// 雷达模型相关API
export const listRadars = () => api.get('/radars')
export const getRadar = (id) => api.get(`/radars/${id}`)
export const createRadar = (data) => api.post('/radars', data)
export const updateRadar = (id, data) => api.put(`/radars/${id}`, data)
export const deleteRadar = (id) => api.delete(`/radars/${id}`)
export const getMaterials = () => api.get('/radars/materials/list')

// 场景相关API
export const listScenes = () => api.get('/scenes')
export const getScene = (id) => api.get(`/scenes/${id}`)
export const createScene = (data) => api.post('/scenes', data)
export const updateScene = (id, data) => api.put(`/scenes/${id}`, data)
export const deleteScene = (id) => api.delete(`/scenes/${id}`)

// 仿真相关API
export const startSimulation = (data) => api.post('/sim/start', data)
export const controlSimulation = (data) => api.post('/sim/control', data)
export const getSimulationStatus = (id) => api.get(`/sim/status?simulation_id=${id}`)
export const listSimulations = () => api.get('/sim/list')
export const deleteSimulation = (id) => api.delete(`/sim/${id}`)

export default api
