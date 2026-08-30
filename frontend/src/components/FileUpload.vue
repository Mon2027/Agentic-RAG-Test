<template>
  <div class="space-y-3">
    <div>
      <div class="mb-2 flex items-center justify-between">
        <label class="text-sm font-medium text-blue-950">上传研报</label>
        <span class="text-xs text-blue-700/55">PDF</span>
      </div>
      <div
        class="group cursor-pointer rounded-md border border-dashed border-blue-300 bg-blue-50/70 px-3 py-3 transition-colors hover:border-blue-500 hover:bg-blue-50"
        :class="{ 'border-blue-500 bg-blue-50': isDraggingReport }"
        @click="reportInput?.click()"
        @dragover.prevent="isDraggingReport = true"
        @dragleave.prevent="isDraggingReport = false"
        @drop.prevent="handleReportDrop"
      >
        <input
          ref="reportInput"
          type="file"
          accept=".pdf"
          class="hidden"
          @change="handleReportSelect"
        />
        <div class="flex items-center gap-3">
          <div class="flex h-9 w-9 items-center justify-center rounded border border-blue-200 bg-white text-sm font-semibold text-blue-700">
            PDF
          </div>
          <div class="min-w-0 text-left">
            <p class="text-sm font-medium text-blue-950">拖入或选择研报文件</p>
            <p class="text-xs text-blue-700/65">上传后将在后台解析并建立索引</p>
          </div>
        </div>
      </div>
    </div>

    <div>
      <div class="mb-2 flex items-center justify-between">
        <label class="text-sm font-medium text-emerald-950">上传数据文件</label>
        <span class="text-xs text-emerald-700/55">CSV/XLSX</span>
      </div>
      <div
        class="group cursor-pointer rounded-md border border-dashed border-emerald-300 bg-emerald-50/70 px-3 py-3 transition-colors hover:border-emerald-500 hover:bg-emerald-50"
        :class="{ 'border-emerald-500 bg-emerald-50': isDraggingData }"
        @click="dataInput?.click()"
        @dragover.prevent="isDraggingData = true"
        @dragleave.prevent="isDraggingData = false"
        @drop.prevent="handleDataDrop"
      >
        <input
          ref="dataInput"
          type="file"
          accept=".csv,.xlsx,.xls"
          class="hidden"
          @change="handleDataSelect"
        />
        <div class="flex items-center gap-3">
          <div class="flex h-9 w-9 items-center justify-center rounded border border-emerald-200 bg-white text-xs font-semibold text-emerald-700">
            DATA
          </div>
          <div class="min-w-0 text-left">
            <p class="text-sm font-medium text-emerald-950">拖入或选择表格文件</p>
            <p class="text-xs text-emerald-700/65">支持统计分析与图表生成</p>
          </div>
        </div>
      </div>
    </div>

    <div v-if="uploading" class="rounded border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-700">
      <span class="loading-dots">正在上传</span>
    </div>
    <div
      v-if="uploadMessage"
      class="rounded border px-3 py-2 text-sm"
      :class="uploadError ? 'border-red-200 bg-red-50 text-red-700' : 'border-emerald-200 bg-emerald-50 text-emerald-700'"
    >
      {{ uploadMessage }}
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import api from '@/services/api'

const reportInput = ref<HTMLInputElement | null>(null)
const dataInput = ref<HTMLInputElement | null>(null)
const isDraggingReport = ref(false)
const isDraggingData = ref(false)
const uploading = ref(false)
const uploadMessage = ref('')
const uploadError = ref(false)

const emit = defineEmits<{
  uploaded: []
}>()

const clearMessage = () => {
  setTimeout(() => {
    uploadMessage.value = ''
    uploadError.value = false
  }, 4000)
}

const setUploadError = (message: string) => {
  uploadError.value = true
  uploadMessage.value = message
  clearMessage()
}

const resetInputs = () => {
  if (reportInput.value) reportInput.value.value = ''
  if (dataInput.value) dataInput.value.value = ''
}

const uploadReport = async (file: File) => {
  uploading.value = true
  uploadMessage.value = ''
  uploadError.value = false

  try {
    const response = await api.uploadReport(file)
    uploadMessage.value = `研报 "${response.file_name}" 已上传，正在后台解析。`
    emit('uploaded')
  } catch (error) {
    uploadError.value = true
    uploadMessage.value = `上传失败: ${error instanceof Error ? error.message : '未知错误'}`
  } finally {
    uploading.value = false
    resetInputs()
    clearMessage()
  }
}

const uploadData = async (file: File) => {
  uploading.value = true
  uploadMessage.value = ''
  uploadError.value = false

  try {
    const response = await api.uploadData(file)
    uploadMessage.value = `数据文件 "${response.file_name}" 已可用于分析。`
    emit('uploaded')
  } catch (error) {
    uploadError.value = true
    uploadMessage.value = `上传失败: ${error instanceof Error ? error.message : '未知错误'}`
  } finally {
    uploading.value = false
    resetInputs()
    clearMessage()
  }
}

const handleReportSelect = (event: Event) => {
  const target = event.target as HTMLInputElement
  const file = target.files?.[0]
  if (file && file.name.toLowerCase().endsWith('.pdf')) {
    uploadReport(file)
  } else if (file) {
    setUploadError('请选择 PDF 格式的研报文件。')
  }
}

const handleDataSelect = (event: Event) => {
  const target = event.target as HTMLInputElement
  const file = target.files?.[0]
  if (file && isSupportedDataFile(file)) {
    uploadData(file)
  } else if (file) {
    setUploadError('请选择 CSV、XLS 或 XLSX 格式的数据文件。')
  }
}

const handleReportDrop = (event: DragEvent) => {
  isDraggingReport.value = false
  const file = event.dataTransfer?.files[0]
  if (file && file.name.toLowerCase().endsWith('.pdf')) {
    uploadReport(file)
  } else if (file) {
    setUploadError('这里只能上传 PDF 研报。')
  }
}

const isSupportedDataFile = (file: File) => {
  const ext = file.name.toLowerCase().split('.').pop()
  return ['csv', 'xlsx', 'xls'].includes(ext || '')
}

const handleDataDrop = (event: DragEvent) => {
  isDraggingData.value = false
  const file = event.dataTransfer?.files[0]
  if (file && isSupportedDataFile(file)) {
    uploadData(file)
  } else if (file) {
    setUploadError('这里只能上传 CSV、XLS 或 XLSX 数据文件。')
  }
}
</script>

<script lang="ts">
export default {
  name: 'FileUpload',
}
</script>
