import type {
  AskResponse,
  AskStreamEvent,
  Project,
  ProjectDetail,
  RepoCandidate,
  RepoOption,
  RunRequest,
  SourceSnippet,
  RunStatus,
} from './types'

const BASE = import.meta.env.VITE_API_BASE_URL || ''

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, init)
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`
    try {
      const data = await response.json()
      message = data.error || data.detail || message
    } catch {
      // keep status text
    }
    throw new Error(message)
  }
  return (await response.json()) as T
}

export const listProjects = () => request<Project[]>('/api/projects')
export const listRepos = () => request<RepoOption[]>('/api/repos')
export const searchGithub = (q: string) =>
  request<RepoCandidate[]>(`/api/github/search?q=${encodeURIComponent(q)}`)
export const getProject = (id: string) => request<ProjectDetail>(`/api/projects/${id}`)

export const askProject = (
  id: string,
  question: string,
  sessionId?: string,
  mode: 'quick' | 'deep' = 'quick',
) =>
  request<AskResponse>(`/api/projects/${id}/ask`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, session_id: sessionId, mode }),
  })

export async function askProjectStream(
  id: string,
  question: string,
  sessionId: string | undefined,
  mode: 'quick' | 'deep',
  onEvent: (event: AskStreamEvent) => void,
): Promise<void> {
  const response = await fetch(`${BASE}/api/projects/${id}/ask/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, session_id: sessionId, mode }),
  })
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`
    try {
      const data = await response.json()
      message = data.error || data.detail || message
    } catch {
      // keep status text
    }
    throw new Error(message)
  }
  if (!response.body) throw new Error('streaming response has no body')
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  const consume = (block: string) => {
    for (const line of block.split(/\r?\n/)) {
      if (!line.startsWith('data:')) continue
      const raw = line.slice(5).trim()
      if (!raw) continue
      let event: AskStreamEvent
      try {
        event = JSON.parse(raw) as AskStreamEvent
      } catch {
        continue
      }
      onEvent(event)
    }
  }
  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n')
      let split = buffer.indexOf('\n\n')
      while (split >= 0) {
        consume(buffer.slice(0, split))
        buffer = buffer.slice(split + 2)
        split = buffer.indexOf('\n\n')
      }
    }
    buffer += decoder.decode()
    if (buffer.trim()) consume(buffer)
  } finally {
    reader.releaseLock()
  }
}

export const getSource = (id: string, path: string, start: number, end: number) =>
  request<SourceSnippet>(
    `/api/projects/${id}/source?path=${encodeURIComponent(path)}&start=${start}&end=${end}`,
  )

export const deleteProject = (id: string) =>
  request<{ deleted: boolean; id: string }>(`/api/projects/${id}`, { method: 'DELETE' })

export async function getPage(id: string, path: string): Promise<string> {
  const response = await fetch(
    `${BASE}/api/projects/${id}/page?path=${encodeURIComponent(path)}`,
  )
  if (!response.ok) {
    throw new Error(`failed to load page: ${response.status}`)
  }
  return response.text()
}

export const startRun = (payload: RunRequest) =>
  request<RunStatus>('/api/runs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })

export const getRun = (id: string) => request<RunStatus>(`/api/runs/${id}`)

export const cancelRun = (id: string) =>
  request<RunStatus>(`/api/runs/${id}/cancel`, { method: 'POST' })
