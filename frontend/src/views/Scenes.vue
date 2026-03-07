<template>
  <div class="scenes-page">
    <el-card>
      <template #header>
        <div class="card-header">
          <span>场景管理</span>
          <el-button type="primary" @click="showCreateDialog">
            <el-icon><Plus /></el-icon>
            创建场景
          </el-button>
        </div>
      </template>

      <el-table :data="scenes" stripe v-loading="loading">
        <el-table-column prop="id" label="ID" width="150" />
        <el-table-column prop="name" label="场景名称" />
        <el-table-column prop="description" label="描述" />
        <el-table-column prop="radar_model_id" label="关联雷达" width="150" />
        <el-table-column prop="targets" label="目标数量" width="100">
          <template #default="{ row }">
            {{ row.targets?.length || 0 }}
          </template>
        </el-table-column>
        <el-table-column prop="updated_at" label="更新时间" width="200" />
        <el-table-column label="操作" width="250" fixed="right">
          <template #default="{ row }">
            <el-button type="primary" size="small" @click="editScene(row)">编辑</el-button>
            <el-button type="success" size="small" @click="startSimulation(row)">仿真</el-button>
            <el-button type="danger" size="small" @click="deleteScene(row)">删除</el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <!-- 创建/编辑场景对话框 -->
    <el-dialog
      v-model="dialogVisible"
      :title="isEdit ? '编辑场景' : '创建场景'"
      width="900px"
    >
      <el-form :model="form" label-width="140px">
        <el-divider content-position="left">基本信息</el-divider>
        <el-form-item label="场景ID">
          <el-input v-model="form.id" :disabled="isEdit" placeholder="自动生成" />
        </el-form-item>
        <el-form-item label="场景名称" required>
          <el-input v-model="form.name" placeholder="请输入场景名称" />
        </el-form-item>
        <el-form-item label="场景描述">
          <el-input v-model="form.description" type="textarea" :rows="2" placeholder="请输入场景描述" />
        </el-form-item>
        <el-form-item label="关联雷达模型" required>
          <el-select v-model="form.radar_model_id" placeholder="请选择雷达模型" style="width: 100%">
            <el-option
              v-for="radar in radars"
              :key="radar.id"
              :label="`${radar.name} (${radar.type})`"
              :value="radar.id"
            />
          </el-select>
        </el-form-item>

        <el-divider content-position="left">仿真参数</el-divider>
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item label="仿真时长(s)" required>
              <el-input-number v-model="form.duration" :min="1" :max="1000" style="width: 100%" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="时间步长(s)" required>
              <el-input-number v-model="form.time_step" :min="0.01" :max="1" :step="0.01" style="width: 100%" />
            </el-form-item>
          </el-col>
        </el-row>

        <el-divider content-position="left">
          目标列表
          <el-button type="primary" size="small" @click="addTarget" style="margin-left: 10px">
            <el-icon><Plus /></el-icon> 添加目标
          </el-button>
        </el-divider>
        <el-empty v-if="form.targets.length === 0" description="暂无目标，请添加" />
        <div v-for="(target, index) in form.targets" :key="index" class="target-item">
          <el-card>
            <template #header>
              <div class="target-header">
                <span>目标 {{ index + 1 }}: {{ target.name }}</span>
                <el-button type="danger" size="small" @click="removeTarget(index)">删除</el-button>
              </div>
            </template>
            <el-row :gutter="20">
              <el-col :span="12">
                <el-form-item label="目标ID">
                  <el-input v-model="target.id" placeholder="如: target-1" />
                </el-form-item>
              </el-col>
              <el-col :span="12">
                <el-form-item label="目标名称">
                  <el-input v-model="target.name" placeholder="如: 敌机-1" />
                </el-form-item>
              </el-col>
            </el-row>
            <el-row :gutter="20">
              <el-col :span="12">
                <el-form-item label="目标类型">
                  <el-select v-model="target.type" style="width: 100%">
                    <el-option label="战斗机" value="fighter" />
                    <el-option label="轰炸机" value="bomber" />
                    <el-option label="直升机" value="helicopter" />
                    <el-option label="无人机" value="uav" />
                    <el-option label="导弹" value="missile" />
                    <el-option label="舰船" value="ship" />
                    <el-option label="民航" value="civil" />
                  </el-select>
                </el-form-item>
              </el-col>
              <el-col :span="12">
                <el-form-item label="RCS模型">
                  <el-select v-model="target.rcs_model" style="width: 100%">
                    <el-option label="Swerling I" value="swerling1" />
                    <el-option label="Swerling II" value="swerling2" />
                    <el-option label="Swerling III" value="swerling3" />
                    <el-option label="Swerling IV" value="swerling4" />
                    <el-option label="常数" value="constant" />
                  </el-select>
                </el-form-item>
              </el-col>
            </el-row>
            <el-row :gutter="20">
              <el-col :span="12">
                <el-form-item label="平均RCS(m²)" required>
                  <el-input-number v-model="target.rcs_mean" :min="0.01" :max="100" :step="0.1" style="width: 100%" />
                </el-form-item>
              </el-col>
              <el-col :span="12">
                <el-form-item label="IFF识别码">
                  <el-input v-model="target.iff_code" placeholder="可选" />
                </el-form-item>
              </el-col>
            </el-row>
          </el-card>
        </div>
      </el-form>

      <template #footer>
        <el-button @click="dialogVisible = false">取消</el-button>
        <el-button type="primary" @click="saveScene" :loading="saving">确定</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { useRouter } from 'vue-router'
import { listScenes, createScene, updateScene, deleteScene as deleteSceneApi, listRadars } from '@/api'

const router = useRouter()
const loading = ref(false)
const saving = ref(false)
const scenes = ref([])
const radars = ref([])
const dialogVisible = ref(false)
const isEdit = ref(false)

const form = ref({
  id: '',
  name: '',
  description: '',
  radar_model_id: '',
  targets: [],
  environment: {
    noise_power: 1e-12,
    clutter: {
      type: 'ground',
      pdf: 'rayleigh',
      psd: 'gaussian',
      strength: 0.1,
      correlation_time: 0.01
    },
    interferences: [],
    atmosphere: {
      temperature: 15,
      pressure: 101325,
      humidity: 50,
      rain_rate: 0
    }
  },
  duration: 100,
  time_step: 0.1
})

async function loadScenes() {
  loading.value = true
  try {
    scenes.value = await listScenes()
  } catch (error) {
    ElMessage.error('加载场景失败')
  } finally {
    loading.value = false
  }
}

async function loadRadars() {
  try {
    radars.value = await listRadars()
  } catch (error) {
    console.error('加载雷达模型失败:', error)
  }
}

function showCreateDialog() {
  isEdit.value = false
  form.value = {
    id: `scene-${Date.now()}`,
    name: '',
    description: '',
    radar_model_id: '',
    targets: [],
    environment: {
      noise_power: 1e-12,
      clutter: {
        type: 'ground',
        pdf: 'rayleigh',
        psd: 'gaussian',
        strength: 0.1,
        correlation_time: 0.01
      },
      interferences: [],
      atmosphere: {
        temperature: 15,
        pressure: 101325,
        humidity: 50,
        rain_rate: 0
      }
    },
    duration: 100,
    time_step: 0.1
  }
  dialogVisible.value = true
}

function addTarget() {
  form.value.targets.push({
    id: `target-${form.value.targets.length + 1}`,
    name: `目标-${form.value.targets.length + 1}`,
    type: 'fighter',
    rcs_model: 'swerling1',
    rcs_mean: 5,
    iff_code: '',
    trajectory: [
      { time: 0, x: 100000, y: 0, z: 5000, vx: -200, vy: 0, vz: 0 },
      { time: 10, x: 98000, y: 0, z: 5000, vx: -200, vy: 0, vz: 0 }
    ],
    altitude: 5000,
    speed: 200,
    heading: 180
  })
}

function removeTarget(index) {
  form.value.targets.splice(index, 1)
}

async function saveScene() {
  if (!form.value.name) {
    ElMessage.warning('请输入场景名称')
    return
  }
  if (!form.value.radar_model_id) {
    ElMessage.warning('请选择关联的雷达模型')
    return
  }
  if (form.value.targets.length === 0) {
    ElMessage.warning('请至少添加一个目标')
    return
  }

  saving.value = true
  try {
    if (isEdit.value) {
      await updateScene(form.value.id, form.value)
      ElMessage.success('更新成功')
    } else {
      await createScene(form.value)
      ElMessage.success('创建成功')
    }
    dialogVisible.value = false
    loadScenes()
  } catch (error) {
    console.error('保存失败:', error)
    ElMessage.error(error.response?.data?.detail || '保存失败，请检查输入数据')
  } finally {
    saving.value = false
  }
}

function editScene(row) {
  isEdit.value = true
  form.value = JSON.parse(JSON.stringify(row))
  dialogVisible.value = true
}

function startSimulation(row) {
  router.push({ path: '/simulation', query: { sceneId: row.id } })
}

async function deleteScene(row) {
  try {
    await ElMessageBox.confirm('确定要删除该场景吗？', '提示', {
      type: 'warning'
    })
    await deleteSceneApi(row.id)
    ElMessage.success('删除成功')
    loadScenes()
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error('删除失败')
    }
  }
}

onMounted(() => {
  loadScenes()
  loadRadars()
})
</script>

<style scoped>
.scenes-page {
  height: 100%;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.target-item {
  margin-bottom: 15px;
}

.target-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.target-item .el-card {
  border: 1px solid #dcdfe6;
}

.target-item .el-form-item {
  margin-bottom: 10px;
}
</style>
