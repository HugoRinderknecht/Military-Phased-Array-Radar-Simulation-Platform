<template>
  <div class="simulation-page">
    <el-row :gutter="20">
      <el-col :span="24">
        <el-card>
          <template #header>
            <div class="card-header">
              <span>仿真控制</span>
            </div>
          </template>

          <el-form :model="simForm" label-width="120px">
            <el-row :gutter="20">
              <el-col :span="8">
                <el-form-item label="选择场景">
                  <el-select v-model="simForm.sceneId" placeholder="请选择场景" style="width: 100%;">
                    <el-option
                      v-for="scene in scenes"
                      :key="scene.id"
                      :label="scene.name"
                      :value="scene.id"
                    />
                  </el-select>
                </el-form-item>
              </el-col>

              <el-col :span="8">
                <el-form-item label="保存结果">
                  <el-switch v-model="simForm.saveResults" />
                </el-form-item>
              </el-col>

              <el-col :span="8">
                <el-form-item>
                  <el-button
                    type="primary"
                    :disabled="!simForm.sceneId || running"
                    :loading="starting"
                    @click="startSimulation"
                  >
                    <el-icon><VideoPlay /></el-icon>
                    启动仿真
                  </el-button>
                  <el-button
                    :disabled="!running"
                    @click="pauseSimulation"
                  >
                    <el-icon><VideoPause /></el-icon>
                    {{ paused ? '继续' : '暂停' }}
                  </el-button>
                  <el-button
                    type="danger"
                    :disabled="!running"
                    @click="stopSimulation"
                  >
                    <el-icon><VideoStop /></el-icon>
                    停止
                  </el-button>
                </el-form-item>
              </el-col>
            </el-row>
          </el-form>
        </el-card>
      </el-col>
    </el-row>

    <el-row :gutter="20" style="margin-top: 20px;">
      <el-col :span="16">
        <el-card>
          <template #header>
            <div class="card-header">
              <span>PPI 显示</span>
              <el-switch v-model="showTracks" active-text="显示航迹" inactive-text="仅点迹" style="margin-left: 20px" />
            </div>
          </template>

          <div ref="ppiChart" class="ppi-display" v-loading="loadingPPI"></div>
        </el-card>
      </el-col>

      <el-col :span="8">
        <el-card>
          <template #header>
            <div class="card-header">
              <span>仿真状态</span>
            </div>
          </template>

          <el-descriptions :column="1" border v-if="currentSimulation">
            <el-descriptions-item label="状态">
              <el-tag v-if="currentSimulation.status === 'running'" type="warning">运行中</el-tag>
              <el-tag v-else-if="currentSimulation.status === 'completed'" type="success">已完成</el-tag>
              <el-tag v-else-if="currentSimulation.status === 'paused'" type="info">已暂停</el-tag>
              <el-tag v-else type="danger">错误</el-tag>
            </el-descriptions-item>
            <el-descriptions-item label="进度">
              {{ currentSimulation.progress.toFixed(1) }}%
            </el-descriptions-item>
            <el-descriptions-item label="当前时间">
              {{ currentSimulation.current_time.toFixed(2) }}s
            </el-descriptions-item>
            <el-descriptions-item label="点迹数">
              {{ currentSimulation.plots_count }}
            </el-descriptions-item>
            <el-descriptions-item label="航迹数">
              {{ currentSimulation.tracks_count }}
            </el-descriptions-item>
          </el-descriptions>

          <el-progress
            v-if="currentSimulation"
            :percentage="currentSimulation.progress"
            :status="currentSimulation.status === 'completed' ? 'success' : undefined"
            style="margin-top: 20px;"
          />
        </el-card>

        <el-card style="margin-top: 20px;">
          <template #header>
            <div class="card-header">
              <span>航迹列表</span>
            </div>
          </template>

          <el-table :data="tracks" size="small" max-height="300">
            <el-table-column prop="track_id" label="航迹ID" width="100" />
            <el-table-column prop="x" label="X (km)" width="70">
              <template #default="{ row }">
                {{ (row.x / 1000).toFixed(1) }}
              </template>
            </el-table-column>
            <el-table-column prop="y" label="Y (km)" width="70">
              <template #default="{ row }">
                {{ (row.y / 1000).toFixed(1) }}
              </template>
            </el-table-column>
            <el-table-column prop="z" label="高度 (km)" width="70">
              <template #default="{ row }">
                {{ (row.z / 1000).toFixed(1) }}
              </template>
            </el-table-column>
            <el-table-column label="速度" width="70">
              <template #default="{ row }">
                {{ Math.sqrt(row.vx**2 + row.vy**2 + row.vz**2).toFixed(0) }}
              </template>
            </el-table-column>
          </el-table>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, nextTick } from 'vue'
import { ElMessage } from 'element-plus'
import { listScenes, startSimulation as startSimApi, getSimulationStatus, controlSimulation } from '@/api'
import * as echarts from 'echarts'

const scenes = ref([])
const running = ref(false)
const paused = ref(false)
const starting = ref(false)
const loadingPPI = ref(false)
const currentSimulation = ref(null)
const tracks = ref([])
const plots = ref([])
const showTracks = ref(true)

const simForm = ref({
  sceneId: '',
  saveResults: true,
  enableClutter: true
})

let statusCheckInterval = null
let currentSimulationId = null
let ppiChartInstance = null
const ppiChart = ref(null)

async function loadScenes() {
  try {
    scenes.value = await listScenes()

    // 如果URL中有sceneId参数，自动选中
    const urlParams = new URLSearchParams(window.location.search)
    const sceneId = urlParams.get('sceneId')
    if (sceneId) {
      simForm.value.sceneId = sceneId
    }
  } catch (error) {
    ElMessage.error('加载场景失败')
  }
}

async function startSimulation() {
  starting.value = true

  try {
    const result = await startSimApi({
      scene_id: simForm.value.sceneId,
      save_results: simForm.value.saveResults,
      enable_clutter: simForm.value.enableClutter
    })

    currentSimulationId = result.simulation_id
    running.value = true
    paused.value = false

    ElMessage.success('仿真已启动')

    // 初始化PPI图表
    await nextTick()
    initPPIChart()

    // 开始轮询状态
    startStatusCheck()
  } catch (error) {
    ElMessage.error('启动仿真失败')
  } finally {
    starting.value = false
  }
}

async function pauseSimulation() {
  if (!currentSimulationId) return

  try {
    const command = paused.value ? 'resume' : 'pause'
    await controlSimulation({
      simulation_id: currentSimulationId,
      command: command
    })
    paused.value = !paused.value
    ElMessage.success(paused.value ? '已暂停' : '已继续')
  } catch (error) {
    ElMessage.error('控制仿真失败')
  }
}

async function stopSimulation() {
  running.value = false
  paused.value = false

  if (statusCheckInterval) {
    clearInterval(statusCheckInterval)
    statusCheckInterval = null
  }

  ElMessage.info('仿真已停止')
}

function initPPIChart() {
  if (!ppiChart.value) return

  if (ppiChartInstance) {
    ppiChartInstance.dispose()
  }

  ppiChartInstance = echarts.init(ppiChart.value)

  const option = {
    backgroundColor: '#000',
    polar: {
      radius: [0, '90%']
    },
    angleAxis: {
      type: 'value',
      startAngle: 90,
      min: -180,
      max: 180,
      axisLine: { lineStyle: { color: '#0f0' } },
      axisLabel: { color: '#0f0', formatter: '{value}°' },
      splitLine: { lineStyle: { color: '#0a3d0a', type: 'dashed' } }
    },
    radiusAxis: {
      type: 'value',
      name: '距离(km)',
      nameLocation: 'middle',
      nameGap: 25,
      nameTextStyle: { color: '#0f0' },
      axisLine: { lineStyle: { color: '#0f0' } },
      axisLabel: { color: '#0f0', formatter: '{value}' },
      splitLine: { lineStyle: { color: '#0a3d0a' } }
    },
    series: [
      {
        type: 'scatter',
        coordinateSystem: 'polar',
        symbolSize: 6,
        data: [],
        itemStyle: { color: '#00ff00' },
        name: '点迹'
      },
      {
        type: 'scatter',
        coordinateSystem: 'polar',
        symbolSize: 12,
        data: [],
        itemStyle: { color: '#ffff00', borderColor: '#ff0000', borderWidth: 2 },
        name: '航迹'
      }
    ],
    tooltip: {
      formatter: (params) => {
        if (params.seriesIndex === 0) {
          return `点迹<br/>方位: ${params.value[0].toFixed(1)}°<br/>距离: ${(params.value[1] / 1000).toFixed(1)}km`
        } else {
          return `航迹 ${params.data.trackId}<br/>方位: ${params.value[0].toFixed(1)}°<br/>距离: ${(params.value[1] / 1000).toFixed(1)}km`
        }
      }
    },
    legend: {
      data: ['点迹', '航迹'],
      textStyle: { color: '#0f0' },
      bottom: 10
    }
  }

  ppiChartInstance.setOption(option)
}

function updatePPIChart() {
  if (!ppiChartInstance || !currentSimulation.value) return

  // 生成模拟点迹数据（基于仿真状态）
  const plotsData = []
  const plotsCount = currentSimulation.value.plots_count || 0

  for (let i = 0; i < plotsCount; i++) {
    const angle = Math.random() * 360 - 180
    const range = 5000 + Math.random() * 45000
    plotsData.push([angle, range])
  }

  // 生成模拟航迹数据
  const tracksData = []
  const tracksCount = currentSimulation.value.tracks_count || 0

  for (let i = 0; i < tracksCount; i++) {
    const angle = Math.random() * 360 - 180
    const range = 10000 + Math.random() * 30000
    tracksData.push({
      value: [angle, range],
      trackId: `T-${i + 1}`
    })
  }

  ppiChartInstance.setOption({
    series: [
      { data: plotsData },
      { data: tracksData }
    ]
  })
}

function startStatusCheck() {
  if (statusCheckInterval) {
    clearInterval(statusCheckInterval)
  }

  statusCheckInterval = setInterval(async () => {
    if (!currentSimulationId) return

    try {
      const status = await getSimulationStatus(currentSimulationId)
      currentSimulation.value = status

      // 更新PPI显示
      updatePPIChart()

      // 更新航迹列表
      if (status.tracks_count > 0 && tracks.value.length !== status.tracks_count) {
        generateMockTracks(status.tracks_count)
      }

      if (status.status === 'completed' || status.status === 'error') {
        running.value = false
        paused.value = false
        clearInterval(statusCheckInterval)
        statusCheckInterval = null

        if (status.status === 'completed') {
          ElMessage.success('仿真完成')
        } else {
          ElMessage.error('仿真出错: ' + status.error_message)
        }
      }
    } catch (error) {
      console.error('获取仿真状态失败:', error)
    }
  }, 1000)
}

function generateMockTracks(count) {
  tracks.value = []
  for (let i = 0; i < count; i++) {
    tracks.value.push({
      track_id: `T-${i + 1}`,
      x: Math.random() * 100000 - 50000,
      y: Math.random() * 100000 - 50000,
      z: Math.random() * 15000 + 1000,
      vx: (Math.random() - 0.5) * 400,
      vy: (Math.random() - 0.5) * 400,
      vz: (Math.random() - 0.5) * 50
    })
  }
}

onMounted(() => {
  loadScenes()

  // 监听窗口大小变化
  window.addEventListener('resize', () => {
    if (ppiChartInstance) {
      ppiChartInstance.resize()
    }
  })
})

onUnmounted(() => {
  if (statusCheckInterval) {
    clearInterval(statusCheckInterval)
  }
  if (ppiChartInstance) {
    ppiChartInstance.dispose()
  }
})
</script>

<style scoped>
.simulation-page {
  height: 100%;
  overflow-y: auto;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.ppi-display {
  width: 100%;
  height: 450px;
  background: #000;
  border-radius: 4px;
}
</style>
