<template>
  <div class="h-screen overflow-hidden bg-[#edf5ff] text-slate-900">
    <header class="h-16 border-b border-blue-500 bg-[#4f8fcf] text-white shadow-sm">
      <div class="flex h-full items-center justify-between px-5">
        <div class="min-w-0">
          <h1 class="truncate text-base font-semibold text-white">研报分析工作台</h1>
          <p class="text-xs text-blue-50/85">DeepAgents / Agentic RAG</p>
        </div>

        <button
          @click="mobilePanelOpen = true"
          class="rounded border border-white/20 bg-white/10 px-3 py-2 text-xs font-medium text-white lg:hidden"
        >
          资料 {{ documentStats.total }}
        </button>

        <dl class="hidden items-center gap-2 md:flex">
          <div class="min-w-24 border-l border-white/15 px-4">
            <dt class="text-xs text-blue-50/80">研报</dt>
            <dd class="text-sm font-semibold text-white">{{ documentStats.reports }}</dd>
          </div>
          <div class="min-w-24 border-l border-white/15 px-4">
            <dt class="text-xs text-blue-50/80">数据文件</dt>
            <dd class="text-sm font-semibold text-white">{{ documentStats.dataFiles }}</dd>
          </div>
          <div class="min-w-24 border-l border-white/15 px-4">
            <dt class="text-xs text-blue-50/80">处理中</dt>
            <dd class="text-sm font-semibold text-amber-100">{{ documentStats.processing }}</dd>
          </div>
        </dl>
      </div>
    </header>

    <div class="grid h-[calc(100vh-4rem)] grid-cols-1 lg:grid-cols-[22rem_minmax(0,1fr)] xl:grid-cols-[22rem_minmax(0,1fr)_18rem]">
      <aside class="hidden min-h-0 border-r border-blue-100 bg-[#f5f9ff] lg:flex lg:flex-col">
        <section class="border-b border-blue-100 bg-[#f0f7ff] p-4">
          <div class="mb-3 flex items-center justify-between">
            <h2 class="text-sm font-semibold text-blue-950">资料区</h2>
            <span class="rounded border border-blue-200 bg-blue-50 px-2 py-0.5 text-xs text-blue-700">
              {{ documentStats.total }} 个文件
            </span>
          </div>
          <FileUpload @uploaded="refreshDocuments" />
        </section>

        <section class="min-h-0 flex-1 overflow-y-auto p-4">
          <DocumentList
            ref="documentListRef"
            @stats-change="updateDocumentStats"
          />
        </section>
      </aside>

      <main class="min-h-0 bg-[#edf5ff]">
        <ChatPanel />
      </main>

      <aside class="hidden min-h-0 border-l border-violet-100 bg-[#f7f5ff] xl:flex xl:flex-col">
        <section class="border-b border-violet-100 bg-[#f2f0ff] p-4">
          <div class="flex items-center justify-between gap-2">
            <h2 class="text-sm font-semibold text-indigo-950">历史对话</h2>
            <button
              class="rounded border border-violet-200 bg-white px-2 py-1 text-xs text-indigo-700 transition-colors hover:border-violet-300 hover:bg-violet-50"
              @click="loadChatHistory"
            >
              刷新
            </button>
          </div>
          <p class="mt-2 text-xs text-indigo-700/60">点击一条记录恢复上下文</p>
        </section>

        <section class="min-h-0 flex-1 overflow-y-auto p-3">
          <div v-if="chatHistory.length === 0" class="rounded-md border border-dashed border-violet-200 bg-white/60 px-3 py-6 text-center">
            <p class="text-sm font-medium text-indigo-950">暂无历史</p>
            <p class="mt-1 text-xs text-indigo-700/60">完成一次对话后会出现在这里</p>
          </div>

          <div v-else class="space-y-2">
            <button
              v-for="item in chatHistory"
              :key="item.id"
              type="button"
              class="w-full rounded-md border border-violet-100 bg-white px-3 py-2 text-left transition-colors hover:border-violet-300 hover:bg-violet-50"
              @click="loadConversation(item.id)"
            >
              <div class="line-clamp-2 text-sm font-medium text-indigo-950">
                {{ item.title }}
              </div>
              <div class="mt-2 flex items-center justify-between gap-2 text-xs text-indigo-700/55">
                <span>{{ formatHistoryTime(item.updatedAt) }}</span>
                <span>{{ item.messages.length }} 条</span>
              </div>
            </button>
          </div>
        </section>
      </aside>

      <div
        v-if="mobilePanelOpen"
        class="fixed inset-0 z-40 bg-blue-950/30 lg:hidden"
        @click.self="mobilePanelOpen = false"
      >
        <section class="absolute inset-x-0 bottom-0 max-h-[82vh] overflow-hidden rounded-t-lg border-t border-blue-100 bg-[#f5f9ff] shadow-xl">
          <div class="flex items-center justify-between border-b border-blue-100 bg-[#f0f7ff] px-4 py-3">
            <div>
              <h2 class="text-sm font-semibold text-blue-950">资料区</h2>
              <p class="text-xs text-blue-700/70">{{ documentStats.total }} 个文件</p>
            </div>
            <button
              class="rounded border border-blue-200 bg-white px-3 py-1.5 text-xs text-blue-700"
              @click="mobilePanelOpen = false"
            >
              关闭
            </button>
          </div>
          <div class="max-h-[calc(82vh-3.75rem)] overflow-y-auto p-4">
            <FileUpload @uploaded="refreshDocuments" />
            <div class="mt-4">
              <DocumentList
                ref="mobileDocumentListRef"
                @stats-change="updateDocumentStats"
              />
            </div>
          </div>
        </section>
      </div>

    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted, onUnmounted, ref } from 'vue'
import ChatPanel from '@/components/ChatPanel.vue'
import FileUpload from '@/components/FileUpload.vue'
import DocumentList from '@/components/DocumentList.vue'
import api from '@/services/api'
import type { Document, DocumentStats, Message } from '@/services/types'

type DocumentListExpose = {
  loadDocuments: () => Promise<void>
}

type ChatHistoryItem = {
  id: string
  title: string
  updatedAt: number
  messages: Message[]
  sessionId: string | null
}

const CHAT_HISTORY_STORAGE_KEY = 'report-analysis-chat-history'

const documentListRef = ref<DocumentListExpose | null>(null)
const mobileDocumentListRef = ref<DocumentListExpose | null>(null)
const mobilePanelOpen = ref(false)
const chatHistory = ref<ChatHistoryItem[]>([])
const documentStats = ref<DocumentStats>({
  total: 0,
  reports: 0,
  dataFiles: 0,
  readyReports: 0,
  processing: 0,
  failed: 0,
})

const refreshDocuments = () => {
  documentListRef.value?.loadDocuments()
  mobileDocumentListRef.value?.loadDocuments()
  loadDocumentStats()
}

const updateDocumentStats = (stats: DocumentStats) => {
  documentStats.value = stats
}

const buildDocumentStats = (items: Document[]): DocumentStats => {
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

const loadDocumentStats = async () => {
  try {
    documentStats.value = buildDocumentStats(await api.listDocuments())
  } catch (error) {
    console.error('Failed to load document stats:', error)
  }
}

const loadChatHistory = () => {
  try {
    const saved = localStorage.getItem(CHAT_HISTORY_STORAGE_KEY)
    if (!saved) {
      chatHistory.value = []
      return
    }
    const parsed = JSON.parse(saved) as ChatHistoryItem[]
    chatHistory.value = Array.isArray(parsed) ? parsed : []
  } catch (error) {
    console.error('Failed to load chat history:', error)
    chatHistory.value = []
  }
}

const loadConversation = (id: string) => {
  window.dispatchEvent(new CustomEvent('chat-history-load', { detail: id }))
}

const formatHistoryTime = (timestamp: number): string => (
  new Date(timestamp).toLocaleString('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  })
)

onMounted(() => {
  loadDocumentStats()
  loadChatHistory()
  window.addEventListener('chat-history-updated', loadChatHistory)
})

onUnmounted(() => {
  window.removeEventListener('chat-history-updated', loadChatHistory)
})
</script>
