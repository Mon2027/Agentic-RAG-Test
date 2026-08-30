<template>
  <div class="space-y-3">
    <div class="flex items-center justify-between">
      <h3 class="text-sm font-semibold text-blue-950">已上传文件</h3>
      <button
        @click="loadDocuments"
        class="rounded border border-blue-200 bg-white px-2 py-1 text-xs text-blue-600 transition-colors hover:border-blue-300 hover:bg-blue-50 hover:text-blue-700"
      >
        刷新
      </button>
    </div>

    <div class="grid grid-cols-3 overflow-hidden rounded-md border border-blue-100 bg-[#dfeeff] p-0.5">
      <button
        v-for="option in filterOptions"
        :key="option.value"
        @click="activeFilter = option.value"
        class="rounded px-2 py-1.5 text-xs font-medium transition-colors"
        :class="activeFilter === option.value ? 'bg-white text-blue-950 shadow-sm' : 'text-blue-600 hover:text-blue-950'"
      >
        {{ option.label }}
        <span class="ml-1 text-slate-400">{{ option.count }}</span>
      </button>
    </div>

    <div v-if="loading" class="rounded-md border border-amber-200 bg-amber-50 px-3 py-5 text-center text-sm text-amber-700">
      <span class="loading-dots">正在加载</span>
    </div>

    <div v-else-if="documents.length === 0" class="rounded-md border border-dashed border-blue-300 bg-blue-50/70 px-3 py-6 text-center">
      <p class="text-sm font-medium text-blue-950">暂无文件</p>
      <p class="mt-1 text-xs text-blue-700/65">上传研报或数据文件后会显示在这里</p>
    </div>

    <div v-else-if="filteredDocuments.length === 0" class="rounded-md border border-dashed border-violet-300 bg-violet-50 px-3 py-6 text-center">
      <p class="text-sm font-medium text-indigo-950">当前筛选无文件</p>
      <p class="mt-1 text-xs text-indigo-700/60">切换筛选条件查看其他文件</p>
    </div>

    <div v-else class="space-y-2">
      <div
        v-for="doc in filteredDocuments"
        :key="doc.file_id"
        class="group rounded-md border border-blue-100 bg-[#fbfdff] p-3 transition-colors hover:border-blue-300"
      >
        <div class="flex items-start justify-between">
          <div class="flex-1 min-w-0">
            <div class="flex items-center gap-2">
              <span
                class="flex h-7 w-7 shrink-0 items-center justify-center rounded border text-[10px] font-semibold"
                :class="doc.file_type === 'pdf' ? 'border-blue-200 bg-blue-50 text-blue-700' : 'border-emerald-200 bg-emerald-50 text-emerald-700'"
              >
                {{ doc.file_type === 'pdf' ? 'PDF' : 'DATA' }}
              </span>
              <span class="truncate text-sm font-medium text-blue-950" :title="doc.file_name">
                {{ doc.file_name }}
              </span>
            </div>

            <div class="mt-2 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-blue-700/65">
              {{ formatFileSize(doc.file_size) }}
              <span v-if="doc.chunk_count">{{ doc.chunk_count }} 个片段</span>
              <span class="truncate" :title="doc.file_id">ID {{ shortId(doc.file_id) }}</span>
            </div>

            <div class="mt-2 flex items-center justify-between gap-2">
              <span
                class="inline-flex items-center rounded border px-2 py-0.5 text-xs font-medium"
                :class="statusClass(doc.status)"
              >
                {{ statusText(doc.status) }}
              </span>
              <span v-if="doc.message" class="min-w-0 truncate text-xs text-blue-700/50" :title="doc.message">
                {{ doc.message }}
              </span>
            </div>
          </div>

          <div class="ml-2 flex shrink-0 flex-col gap-1 opacity-100 sm:opacity-0 sm:transition-opacity sm:group-hover:opacity-100">
            <button
              v-if="doc.file_type === 'pdf'"
              @click="reindexDocument(doc.file_id)"
              :disabled="isBusy(doc)"
              class="rounded border border-transparent px-2 py-1 text-xs text-blue-700/60 transition-all hover:border-blue-200 hover:bg-blue-50 hover:text-blue-700 disabled:cursor-not-allowed disabled:opacity-40"
              title="重新清洗、分块并重建索引"
            >
              {{ reindexingId === doc.file_id ? '重建中' : '重建' }}
            </button>
            <button
              @click="deleteDocument(doc.file_id)"
              :disabled="isBusy(doc)"
              class="rounded border border-transparent px-2 py-1 text-xs text-blue-700/45 transition-all hover:border-red-200 hover:bg-red-50 hover:text-red-600 disabled:cursor-not-allowed disabled:opacity-40"
              title="删除文件"
            >
              删除
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, onMounted } from 'vue'
import api from '@/services/api'
import type { Document, DocumentStats } from '@/services/types'

type DocumentFilter = 'all' | 'reports' | 'data'

const documents = ref<Document[]>([])
const loading = ref(false)
const activeFilter = ref<DocumentFilter>('all')
const reindexingId = ref<string | null>(null)

const emit = defineEmits<{
  'stats-change': [stats: DocumentStats]
}>()

const buildStats = (items: Document[]): DocumentStats => {
  const reports = items.filter(doc => doc.file_type === 'pdf')
  const dataFiles = items.filter(doc => doc.file_type !== 'pdf')

  return {
    total: items.length,
    reports: reports.length,
    dataFiles: dataFiles.length,
    readyReports: reports.filter(doc => doc.status === 'completed').length,
    processing: items.filter(doc => doc.status === 'pending' || doc.status === 'processing').length,
    failed: items.filter(doc => doc.status === 'failed').length,
  }
}

const emitStats = () => {
  emit('stats-change', buildStats(documents.value))
}

const stats = computed(() => buildStats(documents.value))

const filterOptions = computed(() => [
  { value: 'all' as const, label: '全部', count: stats.value.total },
  { value: 'reports' as const, label: '研报', count: stats.value.reports },
  { value: 'data' as const, label: '数据', count: stats.value.dataFiles },
])

const filteredDocuments = computed(() => {
  if (activeFilter.value === 'reports') {
    return documents.value.filter(doc => doc.file_type === 'pdf')
  }
  if (activeFilter.value === 'data') {
    return documents.value.filter(doc => doc.file_type !== 'pdf')
  }
  return documents.value
})

const loadDocuments = async () => {
  loading.value = true
  try {
    documents.value = await api.listDocuments()
    emitStats()
  } catch (error) {
    console.error('Failed to load documents:', error)
  } finally {
    loading.value = false
  }
}

const deleteDocument = async (fileId: string) => {
  if (!confirm('确定要删除这个文件吗？')) return

  try {
    await api.deleteDocument(fileId)
    documents.value = documents.value.filter(d => d.file_id !== fileId)
    emitStats()
  } catch (error) {
    console.error('Failed to delete document:', error)
    alert('删除失败')
  }
}

const reindexDocument = async (fileId: string) => {
  if (reindexingId.value) return

  reindexingId.value = fileId
  try {
    await api.reindexDocument(fileId)
    documents.value = documents.value.map(doc => (
      doc.file_id === fileId
        ? { ...doc, status: 'processing', message: '正在重新索引' }
        : doc
    ))
    emitStats()
    await loadDocuments()
  } catch (error) {
    console.error('Failed to reindex document:', error)
    alert('重建索引失败')
  } finally {
    reindexingId.value = null
  }
}

const isBusy = (doc: Document): boolean => (
  reindexingId.value === doc.file_id || doc.status === 'pending' || doc.status === 'processing'
)

const formatFileSize = (bytes: number): string => {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

const shortId = (fileId: string): string => fileId.slice(0, 8)

const statusText = (status: Document['status']): string => {
  const labels: Record<Document['status'], string> = {
    pending: '待处理',
    processing: '处理中',
    completed: '已完成',
    failed: '失败',
  }
  return labels[status]
}

const statusClass = (status: Document['status']): string => {
  const classes: Record<Document['status'], string> = {
    pending: 'border-amber-200 bg-amber-50 text-amber-700',
    processing: 'border-blue-200 bg-blue-50 text-blue-700',
    completed: 'border-emerald-200 bg-emerald-50 text-emerald-700',
    failed: 'border-red-200 bg-red-50 text-red-700',
  }
  return classes[status]
}

onMounted(() => {
  loadDocuments()
})

// Expose for parent component
defineExpose({
  loadDocuments,
})
</script>

<script lang="ts">
export default {
  name: 'DocumentList',
}
</script>
