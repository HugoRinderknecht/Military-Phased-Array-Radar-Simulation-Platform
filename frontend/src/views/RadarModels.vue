<template>
  <div class="radar-models-page">
    <el-card>
      <template #header>
        <div class="card-header">
          <span>雷达模型</span>
          <el-button type="primary" @click="showCreateDialog">
            <el-icon><Plus /></el-icon>
            创建模型
          </el-button>
        </div>
      </template>

      <el-table :data="radars" stripe v-loading="loading">
        <el-table-column prop="id" label="ID" width="150" />
        <el-table-column prop="name" label="名称" />
        <el-table-column prop="type" label="类型" width="100">
          <template #default="{ row }">
            <el-tag>{{ row.type }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="material" label="材料" width="100" />
        <el-table-column prop="updated_at" label="更新时间" width="200" />
        <el-table-column label="操作" width="200" fixed="right">
          <template #default="{ row }">
            <el-button type="primary" size="small" @click="viewRadar(row)">查看</el-button>
            <el-button type="danger" size="small" @click="deleteRadar(row)">删除</el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <!-- 创建/编辑对话框 -->
    <el-dialog
      v-model="dialogVisible"
      :title="isEdit ? '编辑雷达模型' : '创建雷达模型'"
      width="900px"
    >
      <el-form :model="form" label-width="140px">
        <el-divider content-position="left">基本信息</el-divider>
        <el-form-item label="模型ID">
          <el-input v-model="form.id" :disabled="isEdit" placeholder="自动生成或手动输入" />
        </el-form-item>
        <el-form-item label="模型名称" required>
          <el-input v-model="form.name" placeholder="请输入雷达模型名称" />
        </el-form-item>
        <el-form-item label="雷达类型" required>
          <el-select v-model="form.type" style="width: 100%">
            <el-option label="AESA (有源相控阵)" value="AESA" />
            <el-option label="PESA (无源相控阵)" value="PESA" />
            <el-option label="Mechanical (机械扫描)" value="Mechanical" />
          </el-select>
        </el-form-item>
        <el-form-item label="材料" required>
          <el-select v-model="form.material" style="width: 100%">
            <el-option label="GaN (氮化镓)" value="GaN" />
            <el-option label="GaAs (砷化镓)" value="GaAs" />
            <el-option label="Ga2O3 (氧化镓)" value="Ga2O3" />
          </el-select>
        </el-form-item>

        <el-divider content-position="left">天线参数</el-divider>
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item label="水平阵元数" required>
              <el-input-number v-model="form.antenna.num_h" :min="10" :max="5000" style="width: 100%" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="垂直阵元数" required>
              <el-input-number v-model="form.antenna.num_v" :min="10" :max="1000" style="width: 100%" />
            </el-form-item>
          </el-col>
        </el-row>
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item label="水平间距(λ)" required>
              <el-input-number v-model="form.antenna.d_h" :min="0.1" :max="0.5" :step="0.01" style="width: 100%" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="垂直间距(λ)" required>
              <el-input-number v-model="form.antenna.d_v" :min="0.1" :max="0.5" :step="0.01" style="width: 100%" />
            </el-form-item>
          </el-col>
        </el-row>
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item label="排列方式">
              <el-select v-model="form.antenna.arrangement" style="width: 100%">
                <el-option label="矩形 (rect)" value="rect" />
                <el-option label="三角 (triangular)" value="triangular" />
                <el-option label="圆形 (circular)" value="circular" />
              </el-select>
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="加权函数">
              <el-select v-model="form.antenna.taper" style="width: 100%">
                <el-option label="均匀 (uniform)" value="uniform" />
                <el-option label="Taylor" value="taylor" />
                <el-option label="Hamming" value="hamming" />
                <el-option label="Hanning" value="hanning" />
                <el-option label="Blackman" value="blackman" />
              </el-select>
            </el-form-item>
          </el-col>
        </el-row>

        <el-divider content-position="left">发射机参数</el-divider>
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item label="发射功率(W)" required>
              <el-input-number v-model="form.transmitter.power" :min="100" :max="1000000" style="width: 100%" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="工作频率(Hz)" required>
              <el-input-number v-model="form.transmitter.frequency" :min="1e9" :max="40e9" :step="1e9" style="width: 100%" />
            </el-form-item>
          </el-col>
        </el-row>
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item label="脉冲宽度(μs)" required>
              <el-input-number v-model="form.transmitter.pulse_width" :min="0.1" :max="100" :step="0.1" style="width: 100%" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="信号带宽(Hz)" required>
              <el-input-number v-model="form.transmitter.bandwidth" :min="1e6" :max="100e6" :step="1e6" style="width: 100%" />
            </el-form-item>
          </el-col>
        </el-row>
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item label="PRF(Hz)" required>
              <el-input-number v-model="form.transmitter.prf" :min="100" :max="20000" style="width: 100%" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="波形类型">
              <el-select v-model="form.transmitter.waveform" style="width: 100%">
                <el-option label="脉冲 (pulse)" value="pulse" />
                <el-option label="LFM (线性调频)" value="lfm" />
                <el-option label="Barker码" value="barker" />
                <el-option label="M序列" value="mseq" />
                <el-option label="Frank码" value="frank" />
              </el-select>
            </el-form-item>
          </el-col>
        </el-row>

        <el-divider content-position="left">接收机参数</el-divider>
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item label="噪声系数(dB)" required>
              <el-input-number v-model="form.receiver.noise_figure" :min="0" :max="10" :step="0.1" style="width: 100%" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="接收带宽(Hz)" required>
              <el-input-number v-model="form.receiver.bandwidth" :min="1e6" :max="100e6" :step="1e6" style="width: 100%" />
            </el-form-item>
          </el-col>
        </el-row>
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item label="噪声温度(K)">
              <el-input-number v-model="form.receiver.temperature" :min="0" :max="1000" style="width: 100%" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="接收增益(dB)">
              <el-input-number v-model="form.receiver.gain" :min="-20" :max="60" :step="0.1" style="width: 100%" />
            </el-form-item>
          </el-col>
        </el-row>

        <el-divider content-position="left">扫描参数</el-divider>
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item label="方位角范围" required>
              <el-slider v-model="form.scan.az_range" :min="-180" :max="180" range style="width: 90%; display: inline-block;" />
              <span style="margin-left: 10px">{{ form.scan.az_range[0] }}° ~ {{ form.scan.az_range[1] }}°</span>
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="俯仰角范围" required>
              <el-slider v-model="form.scan.el_range" :min="-90" :max="90" range style="width: 90%; display: inline-block;" />
              <span style="margin-left: 10px">{{ form.scan.el_range[0] }}° ~ {{ form.scan.el_range[1] }}°</span>
            </el-form-item>
          </el-col>
        </el-row>
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item label="最大距离(m)" required>
              <el-input-number v-model="form.scan.range_max" :min="1000" :max="500000" :step="1000" style="width: 100%" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="扫描周期(s)" required>
              <el-input-number v-model="form.scan.scan_period" :min="0.1" :max="10" :step="0.1" style="width: 100%" />
            </el-form-item>
          </el-col>
        </el-row>
        <el-form-item label="扫描模式">
          <el-select v-model="form.scan.beam_pattern" style="width: 100%">
            <el-option label="交错扫描 (staggered)" value="staggered" />
            <el-option label="光栅扫描 (raster)" value="raster" />
            <el-option label="螺旋扫描 (spiral)" value="spiral" />
          </el-select>
        </el-form-item>

        <el-divider content-position="left">跟踪参数</el-divider>
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item label="滤波器类型">
              <el-select v-model="form.tracking.filter" style="width: 100%">
                <el-option label="KF (卡尔曼滤波)" value="KF" />
                <el-option label="EKF (扩展卡尔曼)" value="EKF" />
                <el-option label="UKF (无迹卡尔曼)" value="UKF" />
              </el-select>
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="关联算法">
              <el-select v-model="form.tracking.association" style="width: 100%">
                <el-option label="NN (最近邻)" value="NN" />
                <el-option label="GNN (全局最近邻)" value="GNN" />
                <el-option label="PDA (概率数据关联)" value="PDA" />
                <el-option label="JPDA (联合PDA)" value="JPDA" />
              </el-select>
            </el-form-item>
          </el-col>
        </el-row>
      </el-form>

      <template #footer>
        <el-button @click="dialogVisible = false">取消</el-button>
        <el-button type="primary" @click="saveRadar" :loading="saving">确定</el-button>
      </template>
    </el-dialog>

    <!-- 雷达详情对话框 -->
    <el-dialog
      v-model="detailDialogVisible"
      title="雷达模型详情"
      width="900px"
    >
      <el-descriptions :column="2" border v-if="currentRadar">
        <el-descriptions-item label="模型ID">{{ currentRadar.id }}</el-descriptions-item>
        <el-descriptions-item label="模型名称">{{ currentRadar.name }}</el-descriptions-item>
        <el-descriptions-item label="雷达类型">
          <el-tag>{{ currentRadar.type }}</el-tag>
        </el-descriptions-item>
        <el-descriptions-item label="材料">
          <el-tag>{{ currentRadar.material }}</el-tag>
        </el-descriptions-item>
        <el-descriptions-item label="创建人">{{ currentRadar.created_by }}</el-descriptions-item>
        <el-descriptions-item label="创建时间">{{ formatTime(currentRadar.created_at) }}</el-descriptions-item>
      </el-descriptions>

      <el-divider content-position="left">天线参数</el-divider>
      <el-descriptions :column="2" border v-if="currentRadar?.antenna">
        <el-descriptions-item label="水平阵元数">{{ currentRadar.antenna.num_h }}</el-descriptions-item>
        <el-descriptions-item label="垂直阵元数">{{ currentRadar.antenna.num_v }}</el-descriptions-item>
        <el-descriptions-item label="水平间距">{{ currentRadar.antenna.d_h }}λ</el-descriptions-item>
        <el-descriptions-item label="垂直间距">{{ currentRadar.antenna.d_v }}λ</el-descriptions-item>
        <el-descriptions-item label="排列方式">{{ currentRadar.antenna.arrangement }}</el-descriptions-item>
        <el-descriptions-item label="加权函数">{{ currentRadar.antenna.taper }}</el-descriptions-item>
      </el-descriptions>

      <el-divider content-position="left">发射机参数</el-divider>
      <el-descriptions :column="2" border v-if="currentRadar?.transmitter">
        <el-descriptions-item label="发射功率">{{ currentRadar.transmitter.power }} W</el-descriptions-item>
        <el-descriptions-item label="工作频率">{{ (currentRadar.transmitter.frequency / 1e9).toFixed(2) }} GHz</el-descriptions-item>
        <el-descriptions-item label="脉冲宽度">{{ (currentRadar.transmitter.pulse_width * 1e6).toFixed(2) }} μs</el-descriptions-item>
        <el-descriptions-item label="信号带宽">{{ (currentRadar.transmitter.bandwidth / 1e6).toFixed(2) }} MHz</el-descriptions-item>
        <el-descriptions-item label="PRF">{{ currentRadar.transmitter.prf }} Hz</el-descriptions-item>
        <el-descriptions-item label="波形类型">{{ currentRadar.transmitter.waveform }}</el-descriptions-item>
      </el-descriptions>

      <el-divider content-position="left">接收机参数</el-divider>
      <el-descriptions :column="2" border v-if="currentRadar?.receiver">
        <el-descriptions-item label="噪声系数">{{ currentRadar.receiver.noise_figure }} dB</el-descriptions-item>
        <el-descriptions-item label="接收带宽">{{ (currentRadar.receiver.bandwidth / 1e6).toFixed(2) }} MHz</el-descriptions-item>
        <el-descriptions-item label="噪声温度">{{ currentRadar.receiver.temperature }} K</el-descriptions-item>
        <el-descriptions-item label="接收增益">{{ currentRadar.receiver.gain }} dB</el-descriptions-item>
      </el-descriptions>

      <el-divider content-position="left">扫描参数</el-divider>
      <el-descriptions :column="2" border v-if="currentRadar?.scan">
        <el-descriptions-item label="方位角范围">{{ currentRadar.scan.az_min }}° ~ {{ currentRadar.scan.az_max }}°</el-descriptions-item>
        <el-descriptions-item label="俯仰角范围">{{ currentRadar.scan.el_min }}° ~ {{ currentRadar.scan.el_max }}°</el-descriptions-item>
        <el-descriptions-item label="最大距离">{{ (currentRadar.scan.range_max / 1000).toFixed(1) }} km</el-descriptions-item>
        <el-descriptions-item label="扫描周期">{{ currentRadar.scan.scan_period }} s</el-descriptions-item>
        <el-descriptions-item label="扫描模式" :span="2">{{ currentRadar.scan.beam_pattern }}</el-descriptions-item>
      </el-descriptions>

      <el-divider content-position="left">跟踪参数</el-divider>
      <el-descriptions :column="2" border v-if="currentRadar?.tracking">
        <el-descriptions-item label="滤波器类型">{{ currentRadar.tracking.filter }}</el-descriptions-item>
        <el-descriptions-item label="关联算法">{{ currentRadar.tracking.association }}</el-descriptions-item>
        <el-descriptions-item label="M/N起始准则">{{ currentRadar.tracking.m_n_start.join('/') }}</el-descriptions-item>
        <el-descriptions-item label="航迹丢失帧数">{{ currentRadar.tracking.track_lost }}</el-descriptions-item>
      </el-descriptions>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { listRadars, createRadar, updateRadar, deleteRadar as deleteRadarApi, getRadar } from '@/api'

const loading = ref(false)
const saving = ref(false)
const radars = ref([])
const dialogVisible = ref(false)
const detailDialogVisible = ref(false)
const isEdit = ref(false)
const currentRadar = ref(null)

const form = ref({
  id: '',
  name: '',
  type: 'AESA',
  material: 'GaN',
  antenna: {
    num_h: 1000,
    num_v: 200,
    d_h: 0.5,
    d_v: 0.5,
    arrangement: 'rect',
    taper: 'hamming'
  },
  transmitter: {
    power: 20000,
    frequency: 10e9,
    waveform: 'lfm',
    pulse_width: 1e-6,
    bandwidth: 10e6,
    prf: 10000
  },
  receiver: {
    noise_figure: 3,
    bandwidth: 10e6,
    temperature: 290,
    gain: 0
  },
  scan: {
    az_min: -60,
    az_max: 60,
    el_min: -10,
    el_max: 10,
    range_max: 200000,
    beam_pattern: 'staggered',
    scan_period: 2.0,
    az_range: [-60, 60],
    el_range: [-10, 10]
  },
  tracking: {
    filter: 'EKF',
    association: 'JPDA',
    m_n_start: [3, 4],
    track_lost: 3,
    gate_probability: 0.99,
    process_noise: 1.0,
    measurement_noise: 10.0
  }
})

async function loadRadars() {
  loading.value = true
  try {
    radars.value = await listRadars()
  } catch (error) {
    ElMessage.error('加载雷达模型失败')
  } finally {
    loading.value = false
  }
}

function showCreateDialog() {
  isEdit.value = false
  form.value = {
    id: `radar-${Date.now()}`,
    name: '',
    type: 'AESA',
    material: 'GaN',
    antenna: {
      num_h: 1000,
      num_v: 200,
      d_h: 0.5,
      d_v: 0.5,
      arrangement: 'rect',
      taper: 'hamming'
    },
    transmitter: {
      power: 20000,
      frequency: 10e9,
      waveform: 'lfm',
      pulse_width: 1e-6,
      bandwidth: 10e6,
      prf: 10000
    },
    receiver: {
      noise_figure: 3,
      bandwidth: 10e6,
      temperature: 290,
      gain: 0
    },
    scan: {
      az_min: -60,
      az_max: 60,
      el_min: -10,
      el_max: 10,
      range_max: 200000,
      beam_pattern: 'staggered',
      scan_period: 2.0,
      az_range: [-60, 60],
      el_range: [-10, 10]
    },
    tracking: {
      filter: 'EKF',
      association: 'JPDA',
      m_n_start: [3, 4],
      track_lost: 3,
      gate_probability: 0.99,
      process_noise: 1.0,
      measurement_noise: 10.0
    }
  }
  dialogVisible.value = true
}

async function viewRadar(row) {
  try {
    currentRadar.value = await getRadar(row.id)
    detailDialogVisible.value = true
  } catch (error) {
    ElMessage.error('加载雷达详情失败')
  }
}

function formatTime(isoString) {
  if (!isoString) return '-'
  return new Date(isoString).toLocaleString('zh-CN')
}

async function saveRadar() {
  if (!form.value.name) {
    ElMessage.warning('请输入模型名称')
    return
  }

  saving.value = true
  try {
    // 准备提交数据，将az_range和el_range转换为az_min/max和el_min/max
    const submitData = {
      ...form.value,
      scan: {
        ...form.value.scan,
        az_min: form.value.scan.az_range[0],
        az_max: form.value.scan.az_range[1],
        el_min: form.value.scan.el_range[0],
        el_max: form.value.scan.el_range[1]
      }
    }
    delete submitData.scan.az_range
    delete submitData.scan.el_range

    // 单位转换：pulse_width从μs转为s
    submitData.transmitter.pulse_width = submitData.transmitter.pulse_width * 1e-6

    console.log('发送的数据:', JSON.stringify(submitData, null, 2))

    if (isEdit.value) {
      await updateRadar(form.value.id, submitData)
      ElMessage.success('更新成功')
    } else {
      await createRadar(submitData)
      ElMessage.success('创建成功')
    }
    dialogVisible.value = false
    loadRadars()
  } catch (error) {
    console.error('保存失败:', error)
    console.error('错误详情:', error.response?.data)
    const detail = error.response?.data?.detail
    if (Array.isArray(detail)) {
      console.log('验证错误列表:', detail)
      ElMessage.error('验证错误: ' + detail.map(e => `${e.loc?.join('.')}: ${e.msg}`).join('; '))
    } else {
      ElMessage.error(detail || '保存失败，请检查输入数据')
    }
  } finally {
    saving.value = false
  }
}

async function deleteRadar(row) {
  try {
    await ElMessageBox.confirm('确定要删除该雷达模型吗？', '提示', {
      type: 'warning'
    })
    await deleteRadarApi(row.id)
    ElMessage.success('删除成功')
    loadRadars()
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error('删除失败')
    }
  }
}

onMounted(() => {
  loadRadars()
})
</script>

<style scoped>
.radar-models-page {
  height: 100%;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
</style>
