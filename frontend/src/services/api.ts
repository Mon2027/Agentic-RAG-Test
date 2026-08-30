import type { ChatRequest, ChatResponse, UploadResponse, Document, SSEEvent } from './types'

const API_BASE = '/api'

// API client using fetch
class APIClient {
  private baseUrl: string

  constructor(baseUrl: string = API_BASE) {
    this.baseUrl = baseUrl
  }

  // Chat API
  async chat(request: ChatRequest): Promise<ChatResponse> {
    const response = await fetch(`${this.baseUrl}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    })

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || 'Chat request failed')
    }

    return response.json()
  }

  // Streaming chat using SSE
  async *chatStream(
    request: ChatRequest,
    onEvent?: (event: SSEEvent) => void
  ): AsyncGenerator<SSEEvent, void, unknown> {
    const response = await fetch(`${this.baseUrl}/chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    })

    if (!response.ok) {
      throw new Error('Stream request failed')
    }

    const reader = response.body?.getReader()
    if (!reader) {
      throw new Error('No response body')
    }

    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })

      // Process complete SSE events
      const lines = buffer.split('\n\n')
      buffer = lines.pop() || ''

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6)
          try {
            const event = JSON.parse(data) as SSEEvent
            onEvent?.(event)
            yield event
          } catch (e) {
            console.error('Failed to parse SSE event:', data)
          }
        }
      }
    }
  }

  // Upload report (PDF)
  async uploadReport(file: File): Promise<UploadResponse> {
    const formData = new FormData()
    formData.append('file', file)

    const response = await fetch(`${this.baseUrl}/upload/report`, {
      method: 'POST',
      body: formData,
    })

    if (!response.ok) {
      let errorMessage = 'Upload failed'
      try {
        const text = await response.text()
        if (text) {
          const error = JSON.parse(text)
          errorMessage = error.detail || errorMessage
        }
      } catch {
        errorMessage = `Upload failed with status ${response.status}`
      }
      throw new Error(errorMessage)
    }

    const text = await response.text()
    if (!text) {
      throw new Error('Server returned empty response')
    }
    return JSON.parse(text)
  }

  // Upload data file (CSV/Excel)
  async uploadData(file: File): Promise<UploadResponse> {
    const formData = new FormData()
    formData.append('file', file)

    const response = await fetch(`${this.baseUrl}/upload/data`, {
      method: 'POST',
      body: formData,
    })

    if (!response.ok) {
      let errorMessage = 'Upload failed'
      try {
        const text = await response.text()
        if (text) {
          const error = JSON.parse(text)
          errorMessage = error.detail || errorMessage
        }
      } catch {
        errorMessage = `Upload failed with status ${response.status}`
      }
      throw new Error(errorMessage)
    }

    const text = await response.text()
    if (!text) {
      throw new Error('Server returned empty response')
    }
    return JSON.parse(text)
  }

  // List documents
  async listDocuments(): Promise<Document[]> {
    const response = await fetch(`${this.baseUrl}/documents`)
    if (!response.ok) {
      throw new Error('Failed to list documents')
    }
    return response.json()
  }

  // Get document info
  async getDocument(fileId: string): Promise<Document> {
    const response = await fetch(`${this.baseUrl}/documents/${fileId}`)
    if (!response.ok) {
      throw new Error('Document not found')
    }
    return response.json()
  }

  // Delete document
  async deleteDocument(fileId: string): Promise<void> {
    const response = await fetch(`${this.baseUrl}/documents/${fileId}`, {
      method: 'DELETE',
    })
    if (!response.ok) {
      throw new Error('Failed to delete document')
    }
  }

  // Reindex PDF document
  async reindexDocument(fileId: string): Promise<UploadResponse> {
    const response = await fetch(`${this.baseUrl}/documents/${fileId}/reindex`, {
      method: 'POST',
    })

    if (!response.ok) {
      let errorMessage = 'Failed to reindex document'
      try {
        const error = await response.json()
        errorMessage = error.detail || errorMessage
      } catch {
        errorMessage = `Reindex failed with status ${response.status}`
      }
      throw new Error(errorMessage)
    }

    return response.json()
  }
}

// Export singleton instance
export const api = new APIClient()
export default api
