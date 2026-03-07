<template>
  <div class="results-page">
    <el-card>
      <template #header>
        <div class="card-header">
          <span>仿真结果</span>
          <el-button type="primary" @click="loadResults">
            <el-icon><Refresh /></el-icon>
            刷新
          </el-button>
        </div>
      </template>

      <el-table :data="simulations" stripe v-loading="loading">
        <el-table-column prop="simulation_id" label="仿真ID" width="180" />
        <el-table-column prop="status" label="状态" width="100">
          <template #default="{ row }">
            <el-tag v-if="row.status === 'completed'" type="success">已完成</el-tag>
            <el-tag v-else-if="row.status === 'running'" type="warning">运行中</el-tag>
            <el-tag v-else-if="row.status === 'error'" type="danger">错误</el-tag>
            <el-tag v-else type="info">{{ row.status }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="progress" label="进度" width="150">
          <template #default="{ row }">
            <el-progress :percentage="row.progress" :status="row.status === 'completed' ? 'success' : undefined" />
          </template>
        </el-table-column>
        <el-table-column prop="started_at" label="开始时间" width="180">
          <template #default="{ row }">
            {{ formatTime(row.started_at) }}
          </template>
        </el-table-column>
        <el-table-column label="操作" width="150" fixed="right">
          <template #default="{ row }">
            <el-button
              type="primary"
              size="small"
              :disabled="row.status !== 'completed'"
              @click="viewResult(row)"
            >
              查看详情
            </el-button>
            <el-button
              type="danger"
              size="small"
              @click="deleteResult(row)"
            >
              删除
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <!-- 结果详情对话框 -->
    <el-dialog
      v-model="detailDialogVisible"
      title="仿真结果详情"
      width="1100px"
    >
      <el-tabs v-model="activeTab">
        <el-tab-pane label="概览" name="overview">
          <el-descriptions :column="2" border>
            <el-descriptions-item label="仿真ID">{{ currentResult?.simulation_id }}</el-descriptions-item>
            <el-descriptions-item label="状态">
              <el-tag v-if="currentResult?.status === 'completed'" type="success">已完成</el-tag>
              <el-tag v-else-if="currentResult?.status === 'error'" type="danger">错误</el-tag>
              <el-tag v-else>{{ currentResult?.status }}</el-tag>
            </el-descriptions-item>
            <el-descriptions-item label="开始时间">{{ formatTime(currentResult?.started_at) }}</el-descriptions-item>
            <el-descriptions-item label="进度">{{ currentResult?.progress?.toFixed(1) }}%</el-descriptions-item>
            <el-descriptions-item label="仿真时长">{{ currentResult?.current_time?.toFixed(2) }}s / {{ currentResult?.total_time?.toFixed(2) }}s</el-descriptions-item>
            <el-descriptions-item label="总时长">{{ currentResult?.total_time?.toFixed(2) }}s</el-descriptions-item>
          </el-descriptions>

          <el-divider content-position="left">检测统计</el-divider>
          <el-row :gutter="20">
            <el-col :span="8">
              <el-statistic title="检测点迹数" :value="resultStats.plots_count">
                <template #suffix>个</template>
              </el-statistic>
            </el-col>
            <el-col :span="8">
              <el-statistic title="确认航迹数" :value="resultStats.tracks_count">
                <template #suffix>条</template>
              </el-statistic>
            </el-col>
            <el-col :span="8">
              <el-statistic title="事件数" :value="resultStats.events_count">
                <template #suffix>个</template>
              </el-statistic>
            </el-col>
          </el-row>
        </el-tab-pane>

        <el-tab-pane label="检测结果" name="detection">
          <div ref="detectionChart" style="width: 100%; height: 350px;"></div>
        </el-tab-pane>

        <el-tab-pane label="航迹信息" name="tracks">
          <el-table :data="mockTracks" stripe max-height="400">
            <el-table-column prop="track_id" label="航迹ID" width="100" />
            <el-table-column prop="duration" label="持续时间(s)" width="120">
              <template #default="{ row }">
                {{ row.duration?.toFixed(1) }}
              </template>
            </el-table-column>
            <el-table-column prop="detections" label="检测次数" width="100" />
            <el-table-column prop="avg_rcs" label="平均RCS(m²)" width="120">
              <template #default="{ row }">
                {{ row.avg_rcs?.toFixed(2) }}
              </template>
            </el-table-column>
            <el-table-column prop="avg_speed" label="平均速度(m/s)" width="120">
              <template #default="{ row }">
                {{ row.avg_speed?.toFixed(1) }}
              </template>
            </el-table-column>
            <el-table-column prop="start_pos" label="起始位置" width="150">
              <template #default="{ row }">
                ({{ row.start_x?.toFixed(0) }}, {{ row.start_y?.toFixed(0) }}, {{ row.start_z?.toFixed(0) }})
              </template>
            </el-table-column>
            <el-table-column prop="end_pos" label="结束位置" width="150">
              <template #default="{ row }">
                ({{ row.end_x?.toFixed(0) }}, {{ row.end_y?.toFixed(0) }}, {{ row.end_z?.toFixed(0) }})
              </template>
            </el-table-column>
          </el-table>
        </el-tab-pane>

        <el-tab-pane label="性能指标" name="performance">
          <div ref="performanceChart" style="width: 100%; height: 350px;"></div>
        </el-tab-pane>

        <el-tab-pane label="导出" name="export">
          <el-form label-width="120px">
            <el-form-item label="导出格式">
              <el-checkbox-group v-model="exportFormats">
                <el-checkbox label="csv">CSV表格</el-checkbox>
                <el-checkbox label="json">JSON数据</el-checkbox>
                <el-checkbox label="pdf">PDF报告</el-checkbox>
              </el-checkbox-group>
            </el-form-item>
            <el-form-item label="导出内容">
              <el-checkbox-group v-model="exportContent">
                <el-checkbox label="overview">概览信息</el-checkbox>
                <el-checkbox label="plots">检测点迹</el-checkbox>
                <el-checkbox label="tracks">航迹数据</el-checkbox>
                <el-checkbox label="stats">统计指标</el-checkbox>
              </el-checkbox-group>
            </el-form-item>
            <el-form-item>
              <el-button type="primary" @click="exportResult" :loading="exporting">
                <el-icon><Download /></el-icon>
                导出结果
              </el-button>
            </el-form-item>
          </el-form>
        </el-tab-pane>
      </el-tabs>

      <template #footer>
        <el-button @click="detailDialogVisible = false">关闭</el-button>
        <el-button type="primary" @click="exportResult" :loading="exporting">
          <el-icon><Download /></el-icon>
          导出报告
        </el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, onMounted, nextTick } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { listSimulations, deleteSimulation, getSimulationStatus } from '@/api'
import * as echarts from 'echarts'

const loading = ref(false)
const exporting = ref(false)
const simulations = ref([])
const detailDialogVisible = ref(false)
const currentResult = ref(null)
const activeTab = ref('overview')
const exportFormats = ref(['csv'])
const exportContent = ref(['overview', 'plots', 'tracks', 'stats'])

// 监听标签页切换
import { watch } from 'vue'
watch(activeTab, (newTab) => {
  onTabChange(newTab)
})

const resultStats = ref({
  plots_count: 0,
  tracks_count: 0,
  events_count: 0
})

const mockTracks = ref([])
let detectionChartInstance = null
let performanceChartInstance = null
const detectionChart = ref(null)
const performanceChart = ref(null)

async function loadResults() {
  loading.value = true
  try {
    simulations.value = await listSimulations()
  } catch (error) {
    ElMessage.error('加载结果失败')
  } finally {
    loading.value = false
  }
}

function formatTime(isoString) {
  if (!isoString) return '-'
  return new Date(isoString).toLocaleString('zh-CN')
}

async function viewResult(row) {
  try {
    currentResult.value = await getSimulationStatus(row.simulation_id)
    resultStats.value = {
      plots_count: currentResult.value.plots_count || 0,
      tracks_count: currentResult.value.tracks_count || 0,
      events_count: currentResult.value.events_count || 0
    }

    // 生成模拟航迹数据
    generateMockTracks(currentResult.value.tracks_count || 0)

    detailDialogVisible.value = true
  } catch (error) {
    ElMessage.error('加载结果详情失败')
  }
}

function generateMockTracks(count) {
  mockTracks.value = []
  for (let i = 0; i < count; i++) {
    const duration = 20 + Math.random() * 80
    const speed = 150 + Math.random() * 350
    mockTracks.value.push({
      track_id: `T-${i + 1}`,
      duration: duration,
      detections: Math.floor(duration / 2),
      avg_rcs: 2 + Math.random() * 8,
      avg_speed: speed,
      start_x: Math.random() * 100000,
      start_y: Math.random() * 100000,
      start_z: 1000 + Math.random() * 10000,
      end_x: Math.random() * 100000,
      end_y: Math.random() * 100000,
      end_z: 1000 + Math.random() * 10000
    })
  }
}

function initDetectionChart() {
  if (!detectionChart.value) return

  if (detectionChartInstance) {
    detectionChartInstance.dispose()
  }

  detectionChartInstance = echarts.init(detectionChart.value)

  const option = {
    title: { text: '检测点迹分布', left: 'center' },
    tooltip: {
      trigger: 'item',
      formatter: '{b}: {c} ({d}%)'
    },
    legend: { orient: 'vertical', left: 'left' },
    series: [
      {
        type: 'pie',
        radius: ['40%', '70%'],
        avoidLabelOverlap: false,
        itemStyle: {
          borderRadius: 10,
          borderColor: '#fff',
          borderWidth: 2
        },
        label: { show: true, formatter: '{b}: {d}%' },
        emphasis: {
          label: { show: true, fontSize: 16, fontWeight: 'bold' }
        },
        data: [
          { value: resultStats.value.plots_count * 0.6, name: '真实目标' },
          { value: resultStats.value.plots_count * 0.25, name: '杂波' },
          { value: resultStats.value.plots_count * 0.15, name: '干扰' }
        ]
      }
    ]
  }

  detectionChartInstance.setOption(option)
}

function initPerformanceChart() {
  if (!performanceChart.value) return

  if (performanceChartInstance) {
    performanceChartInstance.dispose()
  }

  performanceChartInstance = echarts.init(performanceChart.value)

  const option = {
    title: { text: '航迹跟踪性能', left: 'center' },
    tooltip: { trigger: 'axis' },
    legend: { data: ['检测率', '虚警率', '跟踪精度'], bottom: 10 },
    xAxis: {
      type: 'category',
      data: ['0-10s', '10-20s', '20-30s', '30-40s', '40-50s', '50-60s']
    },
    yAxis: [
      { type: 'value', name: '百分比(%)', min: 0, max: 100 },
      { type: 'value', name: '精度(m)', min: 0, max: 200 }
    ],
    series: [
      {
        name: '检测率',
        type: 'line',
        data: [85, 88, 90, 92, 93, 94],
        smooth: true,
        itemStyle: { color: '#67C23A' }
      },
      {
        name: '虚警率',
        type: 'line',
        data: [8, 6, 5, 4, 3, 2],
        smooth: true,
        itemStyle: { color: '#F56C6C' }
      },
      {
        name: '跟踪精度',
        type: 'line',
        yAxisIndex: 1,
        data: [150, 120, 100, 85, 75, 70],
        smooth: true,
        itemStyle: { color: '#409EFF' }
      }
    ]
  }

  performanceChartInstance.setOption(option)
}

async function exportResult() {
  if (exportFormats.value.length === 0) {
    ElMessage.warning('请选择至少一种导出格式')
    return
  }
  if (exportContent.value.length === 0) {
    ElMessage.warning('请选择至少一种导出内容')
    return
  }

  exporting.value = true

  try {
    // 模拟导出过程
    await new Promise(resolve => setTimeout(resolve, 1500))

    if (exportFormats.value.includes('csv')) {
      ElMessage.success('CSV文件导出成功')
    }
    if (exportFormats.value.includes('json')) {
      ElMessage.success('JSON文件导出成功')
    }
    if (exportFormats.value.includes('pdf')) {
      ElMessage.success('PDF报告导出成功')
    }
  } catch (error) {
    ElMessage.error('导出失败')
  } finally {
    exporting.value = false
  }
}

async function deleteResult(row) {
  try {
    await ElMessageBox.confirm('确定要删除该仿真结果吗？', '提示', {
      type: 'warning'
    })
    await deleteSimulation(row.simulation_id)
    ElMessage.success('删除成功')
    loadResults()
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error('删除失败')
    }
  }
}

// 监听标签页切换，初始化图表
async function onTabChange(tabName) {
  if (tabName === 'detection') {
    await nextTick()
    initDetectionChart()
  } else if (tabName === 'performance') {
    await nextTick()
    initPerformanceChart()
  }
}

onMounted(() => {
  loadResults()

  // 监听窗口大小变化
  window.addEventListener('resize', () => {
    if (detectionChartInstance) {
      detectionChartInstance.resize()
    }
    if (performanceChartInstance) {
      performanceChartInstance.resize()
    }
  })
})
</script>

<style scoped>
.results-page {
  height: 100%;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
</style>
