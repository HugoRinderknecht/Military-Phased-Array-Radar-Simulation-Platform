<template>
  <div class="dashboard">
    <el-row :gutter="20">
      <el-col :span="6">
        <el-card class="stat-card">
          <div class="stat-content">
            <div class="stat-icon" style="background: #409eff;">
              <el-icon size="30"><Setting /></el-icon>
            </div>
            <div class="stat-info">
              <div class="stat-value">{{ radarCount }}</div>
              <div class="stat-label">雷达模型</div>
            </div>
          </div>
        </el-card>
      </el-col>

      <el-col :span="6">
        <el-card class="stat-card">
          <div class="stat-content">
            <div class="stat-icon" style="background: #67c23a;">
              <el-icon size="30"><Document /></el-icon>
            </div>
            <div class="stat-info">
              <div class="stat-value">{{ sceneCount }}</div>
              <div class="stat-label">场景配置</div>
            </div>
          </div>
        </el-card>
      </el-col>

      <el-col :span="6">
        <el-card class="stat-card">
          <div class="stat-content">
            <div class="stat-icon" style="background: #e6a23c;">
              <el-icon size="30"><VideoPlay /></el-icon>
            </div>
            <div class="stat-info">
              <div class="stat-value">{{ simulationCount }}</div>
              <div class="stat-label">运行仿真</div>
            </div>
          </div>
        </el-card>
      </el-col>

      <el-col :span="6">
        <el-card class="stat-card">
          <div class="stat-content">
            <div class="stat-icon" style="background: #f56c6c;">
              <el-icon size="30"><Files /></el-icon>
            </div>
            <div class="stat-info">
              <div class="stat-value">{{ resultCount }}</div>
              <div class="stat-label">仿真结果</div>
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <el-row :gutter="20" style="margin-top: 20px;">
      <el-col :span="12">
        <el-card>
          <template #header>
            <div class="card-header">
              <span>快速操作</span>
            </div>
          </template>

          <div class="quick-actions">
            <el-button type="primary" @click="$router.push('/radar')">
              <el-icon><Plus /></el-icon>
              创建雷达模型
            </el-button>

            <el-button type="success" @click="$router.push('/scene')">
              <el-icon><Plus /></el-icon>
              创建场景
            </el-button>

            <el-button type="warning" @click="$router.push('/simulation')">
              <el-icon><VideoPlay /></el-icon>
              启动仿真
            </el-button>
          </div>
        </el-card>
      </el-col>

      <el-col :span="12">
        <el-card>
          <template #header>
            <div class="card-header">
              <span>系统信息</span>
            </div>
          </template>

          <el-descriptions :column="1" border>
            <el-descriptions-item label="系统版本">1.0.0</el-descriptions-item>
            <el-descriptions-item label="用户名">{{ userStore.userInfo?.username }}</el-descriptions-item>
            <el-descriptions-item label="用户角色">{{ userStore.userInfo?.role }}</el-descriptions-item>
            <el-descriptions-item label="存储路径">./data</el-descriptions-item>
          </el-descriptions>
        </el-card>
      </el-col>
    </el-row>

    <el-row style="margin-top: 20px;">
      <el-col :span="24">
        <el-card>
          <template #header>
            <div class="card-header">
              <span>最近仿真</span>
              <el-button type="text" @click="$router.push('/results')">查看全部</el-button>
            </div>
          </template>

          <el-empty v-if="recentSimulations.length === 0" description="暂无仿真记录" />

          <el-table v-else :data="recentSimulations" stripe>
            <el-table-column prop="simulation_id" label="仿真ID" width="200" />
            <el-table-column prop="status" label="状态" width="100">
              <template #default="{ row }">
                <el-tag v-if="row.status === 'completed'" type="success">已完成</el-tag>
                <el-tag v-else-if="row.status === 'running'" type="warning">运行中</el-tag>
                <el-tag v-else-if="row.status === 'error'" type="danger">错误</el-tag>
                <el-tag v-else type="info">{{ row.status }}</el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="progress" label="进度" width="200">
              <template #default="{ row }">
                <el-progress :percentage="row.progress" />
              </template>
            </el-table-column>
            <el-table-column prop="started_at" label="开始时间" />
          </el-table>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { listRadars, listScenes, listSimulations } from '@/api'
import { useUserStore } from '@/stores/user'

const userStore = useUserStore()

const radarCount = ref(0)
const sceneCount = ref(0)
const simulationCount = ref(0)
const resultCount = ref(0)
const recentSimulations = ref([])

async function loadData() {
  try {
    const [radars, scenes, simulations] = await Promise.all([
      listRadars(),
      listScenes(),
      listSimulations()
    ])

    radarCount.value = radars.length
    sceneCount.value = scenes.length
    simulationCount.value = simulations.filter(s => s.status === 'running').length
    resultCount.value = simulations.filter(s => s.status === 'completed').length

    // 最近的5条仿真记录
    recentSimulations.value = simulations.slice(0, 5)
  } catch (error) {
    console.error('加载数据失败:', error)
  }
}

onMounted(() => {
  loadData()
})
</script>

<style scoped>
.dashboard {
  height: 100%;
  overflow-y: auto;
}

.stat-card {
  cursor: pointer;
  transition: transform 0.3s, box-shadow 0.3s;
}

.stat-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
}

.stat-content {
  display: flex;
  align-items: center;
  gap: 20px;
}

.stat-icon {
  width: 60px;
  height: 60px;
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
}

.stat-info {
  flex: 1;
}

.stat-value {
  font-size: 28px;
  font-weight: bold;
  color: #333;
}

.stat-label {
  font-size: 14px;
  color: #999;
  margin-top: 5px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.quick-actions {
  display: flex;
  flex-direction: column;
  gap: 15px;
}

.quick-actions .el-button {
  width: 100%;
  justify-content: flex-start;
}
</style>
