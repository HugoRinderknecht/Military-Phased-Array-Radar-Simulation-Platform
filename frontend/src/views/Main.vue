<template>
  <div class="main-container">
    <el-container>
      <!-- 侧边栏 -->
      <el-aside width="250px">
        <div class="logo">
          <h2>雷达仿真平台</h2>
        </div>

        <el-menu
          :default-active="activeMenu"
          router
          class="menu"
        >
          <el-menu-item index="/dashboard">
            <el-icon><DataBoard /></el-icon>
            <span>仪表盘</span>
          </el-menu-item>

          <el-menu-item index="/radar">
            <el-icon><Setting /></el-icon>
            <span>雷达模型</span>
          </el-menu-item>

          <el-menu-item index="/scene">
            <el-icon><Document /></el-icon>
            <span>场景管理</span>
          </el-menu-item>

          <el-menu-item index="/simulation">
            <el-icon><VideoPlay /></el-icon>
            <span>仿真控制</span>
          </el-menu-item>

          <el-menu-item index="/results">
            <el-icon><Files /></el-icon>
            <span>仿真结果</span>
          </el-menu-item>
        </el-menu>
      </el-aside>

      <!-- 主内容区 -->
      <el-container>
        <!-- 顶部栏 -->
        <el-header>
          <div class="header-left">
            <span class="page-title">{{ pageTitle }}</span>
          </div>

          <div class="header-right">
            <el-dropdown @command="handleCommand">
              <span class="user-info">
                <el-icon><User /></el-icon>
                {{ userStore.userInfo?.username || '用户' }}
                <el-icon><ArrowDown /></el-icon>
              </span>
              <template #dropdown>
                <el-dropdown-menu>
                  <el-dropdown-item command="profile">个人信息</el-dropdown-item>
                  <el-dropdown-item command="logout" divided>退出登录</el-dropdown-item>
                </el-dropdown-menu>
              </template>
            </el-dropdown>
          </div>
        </el-header>

        <!-- 内容区 -->
        <el-main>
          <router-view />
        </el-main>
      </el-container>
    </el-container>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { useUserStore } from '@/stores/user'

const route = useRoute()
const router = useRouter()
const userStore = useUserStore()

const activeMenu = computed(() => route.path)

const pageTitleMap = {
  '/dashboard': '仪表盘',
  '/radar': '雷达模型管理',
  '/scene': '场景管理',
  '/simulation': '仿真控制',
  '/results': '仿真结果'
}

const pageTitle = computed(() => {
  return pageTitleMap[route.path] || '雷达仿真平台'
})

async function handleCommand(command) {
  if (command === 'logout') {
    try {
      await ElMessageBox.confirm('确定要退出登录吗？', '提示', {
        type: 'warning'
      })

      userStore.logout()
      ElMessage.success('已退出登录')
      router.push('/login')
    } catch {
      // 取消操作
    }
  } else if (command === 'profile') {
    ElMessage.info('个人信息功能开发中')
  }
}
</script>

<style scoped>
.main-container {
  width: 100%;
  height: 100%;
}

.el-container {
  height: 100%;
}

.el-aside {
  background-color: #304156;
  display: flex;
  flex-direction: column;
}

.logo {
  height: 60px;
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: #263445;
}

.logo h2 {
  color: #fff;
  font-size: 18px;
  margin: 0;
}

.menu {
  border: none;
  background-color: transparent;
}

:deep(.el-menu-item) {
  color: #bfcbd9;
}

:deep(.el-menu-item:hover) {
  background-color: #263445 !important;
}

:deep(.el-menu-item.is-active) {
  color: #409eff !important;
  background-color: #263445 !important;
}

.el-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  border-bottom: 1px solid #e6e6e6;
  background-color: #fff;
}

.page-title {
  font-size: 18px;
  font-weight: 500;
  color: #333;
}

.user-info {
  display: flex;
  align-items: center;
  gap: 5px;
  cursor: pointer;
  padding: 5px 10px;
  border-radius: 4px;
  transition: background-color 0.3s;
}

.user-info:hover {
  background-color: #f5f5f5;
}

.el-main {
  background-color: #f0f2f5;
  padding: 20px;
  overflow-y: auto;
}
</style>
