/**
 * 测试环境设置文件
 * 在所有测试运行前执行
 */

import { vi } from 'vitest'

// ============ Mock localStorage ============

const localStorageMock = {
  getItem: vi.fn((key) => {
    const store = localStorageMock._store
    return store[key] || null
  }),
  setItem: vi.fn((key, value) => {
    const store = localStorageMock._store
    store[key] = String(value)
  }),
  removeItem: vi.fn((key) => {
    const store = localStorageMock._store
    delete store[key]
  }),
  clear: vi.fn(() => {
    const store = localStorageMock._store
    Object.keys(store).forEach(key => delete store[key])
  }),
  get length() {
    const store = localStorageMock._store
    return Object.keys(store).length
  },
  key: vi.fn((index) => {
    const store = localStorageMock._store
    const keys = Object.keys(store)
    return keys[index] || null
  }),
  _store: {}
}

global.localStorage = localStorageMock

// ============ Mock window.matchMedia ============

Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: vi.fn().mockImplementation(query => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })),
})

// ============ Mock ResizeObserver ============

global.ResizeObserver = vi.fn().mockImplementation(() => ({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn(),
}))

// ============ Mock IntersectionObserver ============

global.IntersectionObserver = vi.fn().mockImplementation(() => ({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn(),
}))

// ============ Mock Element Plus ============

vi.mock('element-plus', async () => {
  const actual = await vi.importActual('element-plus')
  return {
    ...actual,
    ElMessage: {
      success: vi.fn(),
      error: vi.fn(),
      warning: vi.fn(),
      info: vi.fn()
    },
    ElMessageBox: {
      confirm: vi.fn(() => Promise.resolve('confirm')),
      alert: vi.fn(() => Promise.resolve()),
      prompt: vi.fn(() => Promise.resolve({ value: 'test' }))
    },
    ElNotification: {
      success: vi.fn(),
      error: vi.fn(),
      warning: vi.fn(),
      info: vi.fn()
    }
  }
})

// ============ Mock ECharts ============

vi.mock('echarts', () => ({
  default: {
    init: vi.fn(() => ({
      setOption: vi.fn(),
      resize: vi.fn(),
      dispose: vi.fn(),
      on: vi.fn(),
      off: vi.fn(),
      getOption: vi.fn(() => ({}))
    }))
  }
}))

// ============ Mock Socket.io-client ============

vi.mock('socket.io-client', () => ({
  default: vi.fn(() => ({
    on: vi.fn(),
    emit: vi.fn(),
    disconnect: vi.fn(),
    connect: vi.fn(),
    connected: true
  }))
}))

// ============ 清理函数 ============

afterEach(() => {
  // 每个测试后清理localStorage
  localStorageMock.clear()
})
