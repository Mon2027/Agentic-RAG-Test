<template>
  <div class="flex h-full flex-col bg-[#f5f9ff]">
    <div class="border-b border-blue-100 bg-[#f0f7ff] px-3 py-3 sm:px-5">
      <div class="flex items-center justify-between gap-3">
        <div class="min-w-0">
          <h2 class="text-sm font-semibold text-blue-950">分析对话</h2>
          <p class="text-xs text-blue-700/65">研报问答、数据分析与联网补充统一入口</p>
        </div>
        <div class="flex shrink-0 items-center gap-2">
          <button
            type="button"
            class="rounded border border-blue-200 bg-white px-3 py-1.5 text-xs text-blue-700 transition-colors hover:border-blue-300 hover:bg-blue-50"
            @click="clearChat"
          >
            新对话
          </button>
          <span
            class="rounded border px-2 py-1 text-xs"
            :class="isLoading ? 'border-amber-200 bg-amber-50 text-amber-700' : 'border-emerald-200 bg-emerald-50 text-emerald-700'"
          >
            {{ isLoading ? '生成中' : '就绪' }}
          </span>
        </div>
      </div>
    </div>

    <div ref="messagesContainer" class="flex-1 space-y-4 overflow-y-auto bg-[#edf5ff] p-3 sm:p-5">
      <div v-if="messages.length === 0" class="mx-auto max-w-2xl py-8 text-center sm:py-12">
        <div class="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded border border-blue-200 bg-blue-50 text-sm font-semibold text-blue-700">
          RAG
        </div>
        <h2 class="text-lg font-semibold text-blue-950">开始一次研报分析</h2>
        <p class="mx-auto mt-2 max-w-md text-sm text-blue-700/65">
          上传 PDF 研报或 CSV/Excel 数据文件后，可以直接询问业务变化、财务指标、风险点，或要求生成图表。
        </p>
      </div>

      <div
        v-for="(message, index) in messages"
        :key="index"
        :class="[
          'flex',
          message.role === 'user' ? 'justify-end' : 'justify-start'
        ]"
      >
        <div
          :class="[
            'max-w-[96%] rounded-md border px-3 py-3 shadow-sm sm:max-w-[88%] sm:px-4',
            message.role === 'user'
              ? 'border-[#5b8bd9] bg-[#5b8bd9] text-white'
              : 'border-blue-100 bg-[#fbfdff] text-slate-800'
          ]"
        >
          <div
            v-if="message.toolName"
            class="mb-2 inline-flex items-center rounded border border-amber-200 bg-amber-50 px-2 py-1 text-xs text-amber-700"
          >
            <span class="loading-dots">{{ toolLabel(message.toolName) }}</span>
          </div>

          <div
            v-if="message.role === 'assistant'"
            class="markdown-content prose prose-sm max-w-none"
            v-html="renderMarkdown(displayContent(message.content))"
          />
          <div v-else class="whitespace-pre-wrap text-sm leading-6">{{ message.content }}</div>

          <div
            v-if="message.role === 'assistant' && extractChartUrls(message.content).length > 0"
            class="mt-4 grid gap-3"
          >
            <article
              v-for="chartUrl in extractChartUrls(message.content)"
              :key="chartUrl"
              class="overflow-hidden rounded-md border border-violet-100 bg-[#f8f6ff]"
            >
              <div class="flex items-center justify-between border-b border-violet-100 bg-[#f3f0ff] px-3 py-2">
                <div class="min-w-0">
                  <h3 class="text-xs font-semibold text-indigo-950">生成图表</h3>
                  <p class="truncate text-xs text-indigo-700/55">{{ chartFileName(chartUrl) }}</p>
                </div>
                <a
                  :href="chartUrl"
                  target="_blank"
                  rel="noopener noreferrer"
                  class="rounded border border-violet-200 bg-white px-2 py-1 text-xs text-indigo-700 transition-colors hover:border-violet-300 hover:bg-violet-50"
                >
                  打开
                </a>
              </div>
              <div class="bg-white p-2">
                <img
                  :src="chartUrl"
                  alt="生成的图表"
                  class="max-h-72 w-full rounded border border-slate-100 bg-white object-contain sm:max-h-96"
                  loading="lazy"
                />
              </div>
            </article>
          </div>

          <div
            v-if="message.role === 'assistant' && sourceRefs(message).length > 0"
            class="mt-4 border-t border-blue-100 pt-3"
          >
            <div class="mb-2 text-xs font-semibold text-blue-950">来源引用</div>
            <div class="flex flex-wrap gap-2">
              <span
                v-for="source in sourceRefs(message)"
                :key="source.key"
                class="inline-flex max-w-full items-center gap-1 rounded border border-blue-200 bg-blue-50 px-2 py-1 text-xs text-blue-800"
                :title="source.title"
              >
                <span class="font-medium">{{ source.kind }}</span>
                <span class="truncate">{{ source.label }}</span>
                <span v-if="source.pageLabel">{{ source.pageLabel }}</span>
              </span>
            </div>
          </div>
        </div>
      </div>

      <div v-if="isLoading" class="flex justify-start">
        <div class="rounded-md border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-700 shadow-sm">
          <span class="loading-dots">正在组织回答</span>
        </div>
      </div>
    </div>

    <div class="border-t border-blue-100 bg-[#f0f7ff] p-3 sm:p-4">
      <div class="flex flex-col gap-2 sm:flex-row">
        <input
          v-model="inputMessage"
          type="text"
          placeholder="输入您的问题..."
          class="min-w-0 flex-1 rounded-md border border-blue-200 bg-white px-4 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-100"
          :disabled="isLoading"
          @keyup.enter="sendMessage"
        />
        <button
          @click="sendMessage"
          :disabled="isLoading || !inputMessage.trim()"
          class="rounded-md bg-[#5b8bd9] px-5 py-2 text-sm font-medium text-white transition-colors hover:bg-[#4778c8] disabled:cursor-not-allowed disabled:opacity-50 sm:w-auto"
        >
          发送
        </button>
      </div>
      <div class="mt-2 text-xs text-blue-700/55">
        提示: 您可以询问研报内容、要求分析数据或进行联网搜索
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, nextTick, onMounted, watch } from 'vue'
import DOMPurify from 'dompurify'
import { marked } from 'marked'
import type { Message } from '@/services/types'
import api from '@/services/api'

type SourceRef = {
  key: string
  kind: string
  label: string
  pageLabel?: string
  title: string
}

type ChatHistoryItem = {
  id: string
  title: string
  updatedAt: number
  messages: Message[]
  sessionId: string | null
}

// State
const CHAT_STORAGE_KEY = 'report-analysis-chat-state'
const CHAT_HISTORY_STORAGE_KEY = 'report-analysis-chat-history'
const messages = ref<Message[]>([])
const inputMessage = ref('')
const isLoading = ref(false)
const sessionId = ref<string | null>(null)
const activeConversationId = ref<string | null>(null)
const messagesContainer = ref<HTMLElement | null>(null)

const cleanStoredMessages = (storedMessages: Message[]): Message[] => {
  const cleaned = storedMessages.filter(message => (
    message.role !== 'assistant' || message.content.trim().length > 0
  ))

  const lastMessage = cleaned[cleaned.length - 1]
  if (lastMessage?.role === 'user') {
    cleaned.push({
      role: 'assistant',
      content: '上次回答在生成过程中被中断了，请重新发送这个问题。',
    })
  }

  return cleaned
}

const chatSnapshot = (): Message[] => (
  messages.value
    .filter(message => message.role !== 'assistant' || message.content.trim().length > 0)
    .map(message => ({ ...message, toolName: undefined }))
)

const readChatHistory = (): ChatHistoryItem[] => {
  try {
    const saved = localStorage.getItem(CHAT_HISTORY_STORAGE_KEY)
    if (!saved) return []
    const parsed = JSON.parse(saved) as ChatHistoryItem[]
    return Array.isArray(parsed) ? parsed : []
  } catch (error) {
    console.error('Failed to read chat history:', error)
    return []
  }
}

const writeChatHistory = (history: ChatHistoryItem[]) => {
  localStorage.setItem(CHAT_HISTORY_STORAGE_KEY, JSON.stringify(history))
  window.dispatchEvent(new CustomEvent('chat-history-updated'))
}

const conversationTitle = (items: Message[]): string => {
  const firstUserMessage = items.find(message => message.role === 'user')?.content.trim()
  if (!firstUserMessage) return '新对话'
  return firstUserMessage.length > 24 ? `${firstUserMessage.slice(0, 24)}...` : firstUserMessage
}

const archiveCurrentChat = () => {
  const snapshot = chatSnapshot()
  if (snapshot.length === 0) return

  const id = activeConversationId.value || crypto.randomUUID()
  activeConversationId.value = id

  const item: ChatHistoryItem = {
    id,
    title: conversationTitle(snapshot),
    updatedAt: Date.now(),
    messages: snapshot,
    sessionId: sessionId.value,
  }

  const rest = readChatHistory().filter(historyItem => historyItem.id !== id)
  writeChatHistory([item, ...rest].slice(0, 30))
}

const saveChatState = () => {
  archiveCurrentChat()
  localStorage.setItem(
    CHAT_STORAGE_KEY,
    JSON.stringify({
      messages: messages.value,
      sessionId: sessionId.value,
      conversationId: activeConversationId.value,
    })
  )
}

// Configure marked
marked.setOptions({
  breaks: true,
  gfm: true,
})

// Render markdown
const renderMarkdown = (content: string): string => {
  const html = marked(content) as string
  return DOMPurify.sanitize(html)
}

const isSourceOnlyLine = (line: string): boolean => {
  const trimmed = line.trim()
  return (
    /^(?:📌\s*)?(?:检索来源|来源|信息来源)[：:]/.test(trimmed) ||
    /^🌐\s*来源[：:]/.test(trimmed) ||
    /^📊\s*来源[：:]/.test(trimmed) ||
    /^[-*]?\s*研报\s*《[^》]+》\s*第\d+(?:-\d+)?页/.test(trimmed) ||
    /^[-*]?\s*(?:研报\s*)?[^《\n]+\.pdf》?\s*第\d+(?:-\d+)?页/.test(trimmed)
  )
}

const displayContent = (content: string): string => {
  const cleaned = content
    .split('\n')
    .filter(line => !isSourceOnlyLine(line))
    .join('\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim()

  return cleaned || content
}

const extractChartUrls = (content: string): string[] => {
  const matches = content.matchAll(/\/static\/charts\/chart_[\w-]+\.png/g)
  return [...new Set([...matches].map(match => match[0]))]
}

const chartFileName = (url: string): string => url.split('/').pop() || url

const toolLabel = (toolName: string): string => {
  const labels: Record<string, string> = {
    task: '正在调用子代理',
    search_reports: '正在检索研报',
    check_rag_relevance: '正在判断相关性',
    web_search: '正在联网搜索',
    web_search_quick: '正在快速搜索',
    analyze_data: '正在分析数据',
    create_chart: '正在生成图表',
    read_csv_file: '正在读取数据',
    read_data_file: '正在读取数据',
  }
  return labels[toolName] || `正在使用 ${toolName}`
}

const normalizeReportLabel = (label: string): string => (
  label
    .trim()
    .replace(/^研报\s*/, '')
    .replace(/^《|》$/g, '')
    .replace(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_/i, '')
    .replace(/\.pdf$/i, '')
    .trim()
)

const sourceRefs = (message: Message): SourceRef[] => {
  const refs = new Map<string, SourceRef>()

  const addReportRef = (label: string, pageLabel?: string) => {
    const normalizedLabel = normalizeReportLabel(label)
    const key = `report-${normalizedLabel.toLowerCase()}-${pageLabel || ''}`
    refs.set(key, {
      key,
      kind: '研报',
      label: normalizedLabel,
      pageLabel,
      title: pageLabel ? `${normalizedLabel} ${pageLabel}` : normalizedLabel,
    })
  }

  for (const source of message.sources || []) {
    const pageLabel = source.page_number ? `第${source.page_number}页` : undefined
    addReportRef(source.file_name, pageLabel)
  }

  const reportMatches = message.content.matchAll(/研报\s*《([^》]+)》\s*(?:第(\d+)(?:-(\d+))?页)?/g)
  for (const match of reportMatches) {
    const pageLabel = match[2]
      ? `第${match[2]}${match[3] ? `-${match[3]}` : ''}页`
      : undefined
    addReportRef(match[1], pageLabel)
  }

  const listMatches = message.content.matchAll(/-\s+(?:研报《)?([^》\n]+?\.pdf)(?:》)?(?:\s*[（(]?第(\d+)(?:-(\d+))?页[^）)\n]*[）)]?)?/g)
  for (const match of listMatches) {
    const pageLabel = match[2]
      ? `第${match[2]}${match[3] ? `-${match[3]}` : ''}页`
      : undefined
    addReportRef(match[1], pageLabel)
  }

  if (/(?:🌐\s*)?来源[：:]\s*(?:联网搜索|网络搜索|外部搜索)/.test(message.content)) {
    refs.set('web-search', {
      key: 'web-search',
      kind: '联网',
      label: '外部搜索结果',
      title: '联网搜索',
    })
  }

  if (/数据分析|📊/.test(message.content)) {
    refs.set('data-analysis', {
      key: 'data-analysis',
      kind: '数据',
      label: '上传数据文件',
      title: '数据分析',
    })
  }

  return [...refs.values()]
}

// Scroll to bottom
const scrollToBottom = async () => {
  await nextTick()
  if (messagesContainer.value) {
    messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
  }
}

const clearChat = () => {
  archiveCurrentChat()
  messages.value = []
  sessionId.value = null
  activeConversationId.value = null
  localStorage.removeItem(CHAT_STORAGE_KEY)
}

// Send message
const sendMessage = async () => {
  const content = inputMessage.value.trim()
  if (!content || isLoading.value) return

  // Add user message
  messages.value.push({
    role: 'user',
    content,
  })

  inputMessage.value = ''
  isLoading.value = true
  await scrollToBottom()

  try {
    // Prepare request
    const request = {
      messages: messages.value.slice(-10), // Keep last 10 messages for context
      session_id: sessionId.value || undefined,
    }

    // Streaming response
    let assistantMessage: Message = {
      role: 'assistant',
      content: '',
    }
    messages.value.push(assistantMessage)

    for await (const event of api.chatStream(request)) {
      if (event.type === 'start') {
        sessionId.value = event.session_id || null
      } else if (event.type === 'token') {
        assistantMessage.content += event.content || ''
        await scrollToBottom()
      } else if (event.type === 'tool_start') {
        assistantMessage.toolName = event.tool
      } else if (event.type === 'tool_end') {
        assistantMessage.toolName = undefined
      } else if (event.type === 'end') {
        // Stream completed
        if (event.content && event.content.length >= assistantMessage.content.length) {
          assistantMessage.content = event.content
        }
        saveChatState()
      } else if (event.type === 'error') {
        assistantMessage.content = `错误: ${event.message}`
        saveChatState()
      }
    }

    saveChatState()
    await scrollToBottom()
  } catch (error) {
    console.error('Chat error:', error)
    messages.value.push({
      role: 'assistant',
      content: `抱歉，发生了错误: ${error instanceof Error ? error.message : '未知错误'}`,
    })
    saveChatState()
  } finally {
    isLoading.value = false
  }
}

onMounted(() => {
  try {
    const saved = localStorage.getItem(CHAT_STORAGE_KEY)
    if (saved) {
      const parsed = JSON.parse(saved) as { messages?: Message[]; sessionId?: string | null; conversationId?: string | null }
      messages.value = Array.isArray(parsed.messages) ? cleanStoredMessages(parsed.messages) : []
      sessionId.value = parsed.sessionId || null
      activeConversationId.value = parsed.conversationId || null
    }
  } catch (error) {
    console.error('Failed to restore chat state:', error)
  }
  scrollToBottom()

  window.addEventListener('chat-history-load', event => {
    const historyId = (event as CustomEvent<string>).detail
    const historyItem = readChatHistory().find(item => item.id === historyId)
    if (!historyItem) return

    messages.value = historyItem.messages
    sessionId.value = historyItem.sessionId
    activeConversationId.value = historyItem.id
    saveChatState()
    scrollToBottom()
  })
})

watch(
  [messages, sessionId],
  () => {
    saveChatState()
  },
  { deep: true }
)

defineExpose({
  clearChat,
})
</script>

<script lang="ts">
// Define component name
export default {
  name: 'ChatPanel',
}
</script>
