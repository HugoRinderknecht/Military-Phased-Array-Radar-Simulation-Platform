import { defineConfig } from 'vitest/config'
import vue from '@vitejs/plugin-vue'
import { fileURLToPath } from 'node:url'

export default defineConfig({
  plugins: [vue()],
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./setup-tests.js'],
    include: ['tests/unit/**/*.{test,spec}.{js,mjs}', 'src/**/__tests__/**/*.{test,spec}.{js,mjs}'],
    exclude: ['node_modules', 'dist', 'tests/e2e'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      include: ['src/**/*.{js,vue}'],
      exclude: [
        'node_modules/',
        'tests/',
        '**/*.d.ts',
        '**/*.config.*',
        'src/main.js',
        '**/mockData.js'
      ]
    },
    mockReset: true,
    restoreMocks: true
  },
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url))
    }
  }
})
