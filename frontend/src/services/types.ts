// API Types
export interface Message {
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp?: string
  toolName?: string
  sources?: Source[]
}

export interface ChatRequest {
  messages: Message[]
  session_id?: string
  context?: Record<string, unknown>
}

export interface ChatResponse {
  message: Message
  session_id: string
  sources?: Source[]
  charts?: string[]
}

export interface Source {
  file_id: string
  file_name: string
  page_number?: number
  content?: string
  score?: number
}

export interface UploadResponse {
  success: boolean
  file_id?: string
  file_name: string
  file_type: string
  message: string
  metadata?: Record<string, unknown>
}

export interface Document {
  file_id: string
  file_name: string
  file_type: string
  file_size: number
  status: 'pending' | 'processing' | 'completed' | 'failed'
  chunk_count?: number
  message?: string
}

export interface DocumentStats {
  total: number
  reports: number
  dataFiles: number
  readyReports: number
  processing: number
  failed: number
}

// SSE Event Types
export interface SSEEvent {
  type: 'start' | 'token' | 'tool_start' | 'tool_end' | 'end' | 'error'
  session_id?: string
  content?: string
  tool?: string
  message?: string
}
