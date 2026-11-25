export interface UsageTokens {
  input?: number | null
  output?: number | null
  reasoning?: number | null
  total?: number | null
}

export interface Usage {
  tokens: UsageTokens
  cost_usd?: number | null
  cost_status?: string
  n_calls?: number | null
  llm_seconds?: number | null
  by_phase?: Record<string, Record<string, number>>
  generation_info?: Record<string, unknown> | null
}

export interface Project {
  id: string
  repo: string
  experiment: string
  system: string
  state_dir: string
  relative_path: string
  commit?: string | null
  language?: string | null
  pages: number
  updated_at?: string | null
  status: string
  graph_version?: string | null
  doc_version?: string | null
  usage: Usage
  source: string
}

export interface DocNode {
  id: string
  path: string
  title: string
  level: number
  doc_type: string
  component_id?: string
  status?: string
  section_count: number
  children: DocNode[]
}

export interface ComponentNode {
  id: string
  name: string
  path: string
  kind?: string
  depth?: number
  summary?: string
  member_count: number
  page_path?: string
  children: ComponentNode[]
}

export interface GraphSummary {
  nodes: number
  edges: number
  files: number
  node_types: Record<string, number>
  edge_types: Record<string, number>
  top_api_files: { path: string; api_count: number }[]
}

export interface ProjectDetail {
  project: Project
  doc_tree: DocNode
  pages: DocNode[]
  components: ComponentNode
  graph: GraphSummary
  usage: Usage
  manifest: Record<string, unknown>
}

export interface RepoOption {
  name: string
  full_name?: string
  url?: string | null
  path: string
  commit?: string | null
  language?: string | null
  source?: string
}

export interface RepoCandidate {
  full_name: string
  name?: string
  url?: string | null
  path?: string
  description?: string | null
  stars?: number
  language?: string | null
  source: 'github' | 'local'
}

export interface RunStatus {
  run_id: string
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled' | 'interrupted'
  source: string
  repo_path: string
  repo_name: string
  experiment: string
  state_dir: string
  log_path: string
  created_at: string
  updated_at: string
  remote_url?: string | null
  repo_commit?: string | null
  returncode?: number | null
  finished_at?: string | null
  project_id?: string | null
  error?: string | null
  progress: number
  stage: string
  logs?: string[]
}

export interface RunRequest {
  source: string
}

export interface Citation {
  number: number
  type: 'doc' | 'code' | 'component'
  label: string
  page?: string | null
  heading?: string | null
  path?: string | null
  start?: number | null
  end?: number | null
  component?: string | null
  component_id?: string | null
  id: string
}

export interface AskResponse {
  answer: string
  citations: Citation[]
  session_id: string
  mode?: 'quick' | 'deep'
  usage?: Record<string, unknown> | null
}

export type AskStreamEvent =
  | {
      type: 'start'
      session_id: string
      mode?: 'quick' | 'deep'
      citations: Citation[]
      usage?: Record<string, unknown> | null
    }
  | { type: 'delta'; content: string }
  | {
      type: 'done'
      answer: string
      citations: Citation[]
      session_id: string
      mode?: 'quick' | 'deep'
      usage?: Record<string, unknown> | null
    }
  | { type: 'error'; message: string }

export interface SourceSnippet {
  path: string
  start: number
  end: number
  content: string
}
