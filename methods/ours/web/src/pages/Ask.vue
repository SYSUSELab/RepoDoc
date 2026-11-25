<script setup lang="ts">
import { computed, nextTick, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { askProjectStream, getPage, getProject, getSource } from '../api'
import MarkdownContent from '../components/MarkdownContent.vue'
import ThemeToggle from '../components/ThemeToggle.vue'
import type { Citation, ProjectDetail } from '../types'

interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
  citations?: Citation[]
  streaming?: boolean
}

interface Evidence {
  type: 'doc' | 'code'
  title: string
  content: string
  path?: string
  start?: number
  end?: number
}

const route = useRoute()
const router = useRouter()
const projectId = computed(() => String(route.params.id || ''))
const detail = ref<ProjectDetail | null>(null)
const question = ref('')
const messages = ref<ChatMessage[]>([])
const sessionId = ref('')
const mode = ref<'quick' | 'deep'>('quick')
const loading = ref(false)
const error = ref('')
const mobilePane = ref<'chat' | 'evidence'>('chat')
const messagesEl = ref<HTMLElement | null>(null)

const panelSources = ref<Citation[]>([])
const panelView = ref<'sources' | 'source'>('sources')
const selectedCitation = ref<Citation | null>(null)
const evidence = ref<Evidence | null>(null)
const evidenceLoading = ref(false)
const isStreaming = computed(() => messages.value.some((message) => message.streaming))

const storageKey = computed(() => `repodoc-chat:${projectId.value}`)
const sessionKey = computed(() => `repodoc-chat-session:${projectId.value}`)
const examples = [
  'How does the request flow work?',
  'Where is the public API defined?',
  'Which components depend on configuration?',
]

const sourceTitle = computed(() => {
  if (panelView.value === 'sources') return 'Sources'
  return selectedCitation.value?.label || 'Source'
})
const sourceMeta = computed(() => {
  if (panelView.value !== 'source') return `${panelSources.value.length} references`
  return selectedCitation.value?.path || selectedCitation.value?.page || ''
})
const codeLines = computed(() => {
  if (evidence.value?.type !== 'code') return []
  const start = evidence.value.start || 1
  return evidence.value.content.split('\n').map((text, index) => ({
    number: start + index,
    text,
  }))
})

const scrollToBottom = async () => {
  await nextTick()
  if (messagesEl.value) messagesEl.value.scrollTop = messagesEl.value.scrollHeight
}

const persist = () => {
  try {
    localStorage.setItem(storageKey.value, JSON.stringify(messages.value))
    if (sessionId.value) localStorage.setItem(sessionKey.value, sessionId.value)
  } catch {
    // local persistence is best-effort
  }
}

const restore = () => {
  try {
    sessionId.value = localStorage.getItem(sessionKey.value) || crypto.randomUUID()
    const raw = localStorage.getItem(storageKey.value)
    if (raw) {
      const parsed = JSON.parse(raw)
      if (Array.isArray(parsed)) messages.value = parsed
    }
    const lastAnswer = [...messages.value]
      .reverse()
      .find((message) => message.role === 'assistant' && message.citations?.length)
    if (lastAnswer?.citations) panelSources.value = lastAnswer.citations
  } catch {
    sessionId.value = crypto.randomUUID()
  }
}

const load = async () => {
  if (!projectId.value) return
  try {
    detail.value = await getProject(projectId.value)
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  }
}

const ask = async () => {
  const text = question.value.trim()
  if (!text || loading.value) return
  question.value = ''
  messages.value.push({ role: 'user', content: text })
  loading.value = true
  error.value = ''
  await scrollToBottom()
  let assistantIndex = -1
  let completed = false
  try {
    await askProjectStream(
      projectId.value,
      text,
      sessionId.value,
      mode.value,
      (event) => {
        if (event.type === 'start') {
          sessionId.value = event.session_id
          messages.value.push({
            role: 'assistant',
            content: '',
            citations: event.citations,
            streaming: true,
          })
          assistantIndex = messages.value.length - 1
          panelSources.value = event.citations || []
          panelView.value = 'sources'
          selectedCitation.value = null
          evidence.value = null
          void scrollToBottom()
          return
        }
        if (event.type === 'delta') {
          if (assistantIndex < 0) return
          messages.value[assistantIndex].content += event.content
          void scrollToBottom()
          return
        }
        if (event.type === 'done') {
          if (assistantIndex < 0) return
          messages.value[assistantIndex].content = event.answer
          messages.value[assistantIndex].citations = event.citations
          messages.value[assistantIndex].streaming = false
          panelSources.value = event.citations || []
          sessionId.value = event.session_id
          completed = true
          loading.value = false
          persist()
          void scrollToBottom()
          return
        }
        if (event.type === 'error') {
          throw new Error(event.message)
        }
      },
    )
    if (!completed) throw new Error('stream ended before completion')
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err)
    if (assistantIndex >= 0) {
      messages.value[assistantIndex].streaming = false
      if (!messages.value[assistantIndex].content) {
        messages.value[assistantIndex].content = `Request failed: ${message}`
      }
    } else {
      messages.value.push({ role: 'assistant', content: `Request failed: ${message}` })
    }
    error.value = message
  } finally {
    if (assistantIndex >= 0) {
      messages.value[assistantIndex].streaming = false
    }
    loading.value = false
    await scrollToBottom()
  }
}

const sourceable = (citation: Citation) =>
  Boolean(citation.page) ||
  Boolean(citation.path && citation.start !== null && citation.end !== null)

const showSources = (citations: Citation[]) => {
  panelSources.value = citations
  panelView.value = 'sources'
  selectedCitation.value = null
  evidence.value = null
  mobilePane.value = 'evidence'
}

const backToSources = () => {
  panelView.value = 'sources'
  selectedCitation.value = null
  evidence.value = null
}

const openCitation = async (citation: Citation) => {
  if (!sourceable(citation)) return
  panelView.value = 'source'
  selectedCitation.value = citation
  evidence.value = null
  evidenceLoading.value = true
  mobilePane.value = 'evidence'
  error.value = ''
  try {
    if (citation.type === 'doc' || (citation.type === 'component' && citation.page)) {
      const page = citation.page
      if (!page) return
      evidence.value = {
        type: 'doc',
        title: citation.heading || citation.label,
        content: await getPage(projectId.value, page),
        path: page,
      }
    } else if (citation.type === 'code' && citation.path && citation.start && citation.end) {
      const snippet = await getSource(
        projectId.value,
        citation.path,
        citation.start,
        citation.end,
      )
      evidence.value = {
        type: 'code',
        title: citation.label,
        content: snippet.content,
        path: snippet.path,
        start: snippet.start,
        end: snippet.end,
      }
    }
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  } finally {
    evidenceLoading.value = false
  }
}

const goProject = () => router.push({ name: 'Project', params: { id: projectId.value } })

const boot = async () => {
  restore()
  void load()
  await nextTick()
  await scrollToBottom()
}

watch(projectId, (id) => {
  if (!id) return
  detail.value = null
  messages.value = []
  sessionId.value = ''
  panelSources.value = []
  panelView.value = 'sources'
  selectedCitation.value = null
  evidence.value = null
  question.value = ''
  error.value = ''
  void boot()
})

onMounted(boot)
</script>

<template>
  <div class="ask-page">
    <header class="ask-topbar">
      <button class="back" @click="goProject" title="Back to documentation">←</button>
      <div class="ask-title">
        <span class="eyebrow">RepoDoc / Ask</span>
        <strong>{{ detail?.project.repo || 'Repository Q&A' }}</strong>
      </div>
      <div class="top-meta">
        <span class="pill">Ask RepoDoc</span>
        <ThemeToggle />
      </div>
    </header>

    <div class="mobile-switch">
      <button :class="{ active: mobilePane === 'chat' }" @click="mobilePane = 'chat'">Chat</button>
      <button :class="{ active: mobilePane === 'evidence' }" @click="mobilePane = 'evidence'">
        Evidence
      </button>
    </div>

    <main class="ask-main">
      <section class="chat-pane" :class="{ 'mobile-hide': mobilePane !== 'chat' }">
        <div ref="messagesEl" class="conversation">
          <section v-if="messages.length === 0" class="intro">
            <span class="eyebrow">Repository Q&A</span>
            <h1>Ask the architecture<br />and the code.</h1>
            <p>
              Answers are grounded in documentation sections, code symbols and graph
              relationships. Sources open in the evidence panel.
            </p>
            <div class="examples">
              <button v-for="example in examples" :key="example" class="example" @click="question = example">
                {{ example }}
              </button>
            </div>
          </section>

          <template v-for="(message, index) in messages" :key="index">
            <article class="message" :class="message.role">
              <span class="message-role">{{ message.role === 'user' ? 'You' : 'RepoDoc' }}</span>
              <div v-if="message.role === 'user'" class="user-text">{{ message.content }}</div>
              <MarkdownContent v-else :content="message.content" />
              <span v-if="message.streaming" class="stream-cursor"></span>

              <button
                v-if="message.citations?.length"
                class="sources-link"
                @click="showSources(message.citations || [])"
              >
                Sources<span class="source-count">{{ message.citations.length }}</span>
                <span class="source-open">→</span>
              </button>
            </article>
          </template>

          <div v-if="loading && !isStreaming" class="thinking">
            <span class="thinking-dot"></span>
            <span>Searching docs and code graph…</span>
          </div>
          <p v-if="error" class="error">{{ error }}</p>
        </div>

        <div class="composer-wrap">
          <form class="composer" @submit.prevent="ask">
            <textarea
              v-model="question"
              rows="2"
              placeholder="Ask about this repository…"
              @keydown.enter.exact.prevent="ask"
            ></textarea>
            <div class="composer-tools">
              <div class="mode-switch">
                <button type="button" :class="{ active: mode === 'quick' }" @click="mode = 'quick'">
                  Quick
                </button>
                <button type="button" :class="{ active: mode === 'deep' }" @click="mode = 'deep'">
                  Deep
                </button>
              </div>
              <button class="send" type="submit" :disabled="loading || !question.trim()">
                {{ loading ? '…' : '→' }}
              </button>
            </div>
          </form>
        </div>
      </section>

      <aside class="evidence-pane" :class="{ 'mobile-hide': mobilePane !== 'evidence' }">
        <header class="evidence-head">
          <button v-if="panelView === 'source'" class="back-sources" @click="backToSources">
            ← Sources
          </button>
          <div class="evidence-title">
            <span class="eyebrow">{{ panelView === 'source' ? 'Source' : 'Evidence' }}</span>
            <strong>{{ sourceTitle }}</strong>
          </div>
          <span class="evidence-meta">{{ sourceMeta }}</span>
        </header>

        <div v-if="panelView === 'sources'" class="evidence-body source-list">
          <div v-if="!panelSources.length" class="evidence-empty">
            <span class="evidence-mark">⌕</span>
            <p>Ask a question to see cited documentation and code.</p>
          </div>
          <template v-else>
            <button
              v-for="citation in panelSources"
              :key="citation.id"
              class="source-row"
              :class="{ disabled: !sourceable(citation) }"
              :disabled="!sourceable(citation)"
              @click="openCitation(citation)"
            >
              <span class="source-number">{{ citation.number }}</span>
              <span class="source-kind">{{ citation.type }}</span>
              <span class="source-label">{{ citation.label }}</span>
              <span class="source-open">↗</span>
            </button>
          </template>
        </div>

        <div v-else class="evidence-body">
          <div v-if="evidenceLoading" class="evidence-empty">Loading source…</div>
          <div v-else-if="evidence?.type === 'code'" class="code-view">
            <div class="code-meta">
              <span class="mono">{{ evidence.path }}</span>
              <span class="mono">{{ evidence.start }}–{{ evidence.end }}</span>
            </div>
            <pre class="code-block"><code><span v-for="line in codeLines" :key="line.number" class="code-line"><span class="line-number">{{ line.number }}</span>{{ line.text }}
</span></code></pre>
          </div>
          <div v-else-if="evidence?.type === 'doc'" class="doc-view">
            <MarkdownContent :content="evidence.content" />
          </div>
          <div v-else class="evidence-empty">No source selected.</div>
        </div>
      </aside>
    </main>
  </div>
</template>

<style scoped>
.ask-page {
  height: 100vh;
  display: flex;
  flex-direction: column;
  background: var(--container-bg);
  overflow: hidden;
}

.ask-topbar {
  height: 64px;
  flex: 0 0 64px;
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 0 max(22px, calc((100% - 1920px) / 2));
  border-bottom: 1px solid var(--border-color);
}

.back {
  width: 36px;
  height: 36px;
  border: 1px solid var(--border-color);
  border-radius: 50%;
  background: transparent;
  color: var(--text-color);
  cursor: pointer;
}

.back:hover {
  background: var(--hover-bg);
}

.ask-title {
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 1px;
}

.eyebrow {
  color: var(--subtitle-color);
  font-size: 9px;
  text-transform: uppercase;
  letter-spacing: 0.8px;
}

.ask-title strong {
  color: var(--title-color);
  font-size: 14px;
}

.top-meta {
  margin-left: auto;
  display: flex;
  align-items: center;
  gap: 8px;
}

.pill {
  color: var(--secondary-text);
  font-size: 9px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  border: 1px solid var(--border-color);
  border-radius: 999px;
  padding: 4px 8px;
}

.mobile-switch {
  display: none;
}

.ask-main {
  flex: 1;
  min-height: 0;
  width: min(1920px, 100%);
  margin: 0 auto;
  display: grid;
  grid-template-columns: minmax(560px, 48%) minmax(0, 1fr);
}

.chat-pane {
  min-height: 0;
  display: flex;
  flex-direction: column;
  border-right: 1px solid var(--border-color);
}

.conversation {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  padding: 38px 32px 28px;
  scrollbar-width: thin;
}

.intro {
  max-width: 540px;
  margin: 14vh auto 0;
  text-align: center;
}

.intro h1 {
  margin: 8px 0 12px;
  color: var(--title-color);
  font-size: 30px;
  line-height: 1.14;
  letter-spacing: -0.7px;
}

.intro p {
  margin: 0 auto;
  color: var(--secondary-text);
  font-size: 11px;
  line-height: 1.7;
}

.examples {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 7px;
  margin-top: 24px;
}

.example {
  border: 0;
  border-bottom: 1px solid var(--border-color);
  background: transparent;
  color: var(--secondary-text);
  font-size: 10px;
  padding: 3px 0;
  cursor: pointer;
}

.example:hover {
  color: var(--title-color);
  border-color: var(--text-color);
}

.message {
  margin: 0 auto 30px;
  max-width: 760px;
}

.message.user {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  max-width: 86%;
  margin-right: 0;
}

.message-role {
  color: var(--subtitle-color);
  font-size: 9px;
  text-transform: uppercase;
  letter-spacing: 0.6px;
  margin-bottom: 6px;
}

.user-text {
  background: var(--hover-bg);
  border-radius: 8px;
  padding: 10px 13px;
  color: var(--text-color);
  font-size: 13px;
  line-height: 1.6;
}

.message.assistant :deep(.markdown-body) {
  max-width: none;
  font-size: 14px;
  line-height: 1.76;
}

.message.assistant :deep(.markdown-body h1) {
  font-size: 25px;
}

.message.assistant :deep(.markdown-body h2) {
  font-size: 18px;
}

.message.assistant :deep(.markdown-body h3) {
  font-size: 15px;
}

.sources-link {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  border: 1px solid var(--border-color);
  border-radius: 999px;
  background: transparent;
  color: var(--subtitle-color);
  font-size: 10.5px;
  padding: 5px 10px;
  margin-top: 14px;
  cursor: pointer;
}

.sources-link:hover {
  color: var(--text-color);
  background: var(--hover-bg);
  border-color: color-mix(in srgb, var(--text-color) 18%, var(--border-color));
}

.source-count {
  border: 1px solid var(--border-color);
  border-radius: 999px;
  padding: 0 5px;
  font-size: 9px;
  font-variant-numeric: tabular-nums;
}

.source-open {
  color: var(--subtitle-color);
  font-size: 11px;
}

.stream-cursor {
  display: inline-block;
  width: 6px;
  height: 14px;
  margin-left: 2px;
  vertical-align: -2px;
  background: var(--text-color);
  opacity: 0.45;
  animation: blink 0.9s step-end infinite;
}

@keyframes blink {
  50% { opacity: 0; }
}

.thinking {
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--subtitle-color);
  font-size: 11px;
  margin: 12px 0 22px;
}

.thinking-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--text-color);
  opacity: 0.35;
  animation: pulse 1.2s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 0.2; transform: scale(0.8); }
  50% { opacity: 0.8; transform: scale(1); }
}

.error {
  color: var(--danger);
  font-size: 11px;
}

.composer-wrap {
  flex: 0 0 auto;
  padding: 10px 32px 24px;
}

.composer {
  width: 100%;
  max-width: 760px;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  border: 1px solid var(--border-color);
  border-radius: 12px;
  padding: 9px 11px 8px 13px;
  background: var(--input-bg);
}

.composer:focus-within {
  border-color: color-mix(in srgb, var(--text-color) 22%, var(--border-color));
}

.composer textarea {
  width: 100%;
  resize: none;
  min-height: 42px;
  max-height: 140px;
  border: 0;
  outline: 0;
  background: transparent;
  color: var(--text-color);
  font-size: 13px;
  line-height: 1.55;
}

.composer textarea::placeholder {
  color: var(--placeholder-color);
}

.composer-tools {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}

.mode-switch {
  display: flex;
  gap: 2px;
}

.mode-switch button {
  border: 0;
  background: transparent;
  color: var(--subtitle-color);
  font-size: 9px;
  padding: 3px 5px;
  cursor: pointer;
  border-radius: 4px;
}

.mode-switch button.active {
  color: var(--title-color);
  background: var(--hover-bg);
}

.send {
  width: 29px;
  height: 29px;
  border: 0;
  border-radius: 7px;
  background: var(--text-color);
  color: var(--container-bg);
  font-size: 16px;
  cursor: pointer;
}

.send:disabled {
  opacity: 0.3;
  cursor: not-allowed;
}

.evidence-pane {
  min-height: 0;
  display: flex;
  flex-direction: column;
  background: color-mix(in srgb, var(--card-bg) 28%, transparent);
}

.evidence-head {
  flex: 0 0 auto;
  min-height: 64px;
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 13px 28px;
  border-bottom: 1px solid var(--border-color);
}

.back-sources {
  border: 0;
  background: transparent;
  color: var(--subtitle-color);
  font-size: 10px;
  cursor: pointer;
  padding: 4px 0;
}

.back-sources:hover {
  color: var(--text-color);
}

.evidence-title {
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.evidence-title strong {
  color: var(--title-color);
  font-size: 13px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.evidence-meta {
  margin-left: auto;
  color: var(--subtitle-color);
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 9px;
  max-width: 45%;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.evidence-body {
  flex: 1;
  min-height: 0;
  overflow: auto;
  scrollbar-width: thin;
}

.source-list {
  padding: 10px 28px 24px;
}

.source-row {
  width: 100%;
  display: grid;
  grid-template-columns: 28px 62px minmax(0, 1fr) 16px;
  align-items: center;
  gap: 10px;
  border: 0;
  border-bottom: 1px solid var(--border-color);
  border-radius: 4px;
  background: transparent;
  color: var(--secondary-text);
  text-align: left;
  padding: 12px 8px;
  margin: 0 -8px;
  cursor: pointer;
  font-size: 11.5px;
}

.source-row:last-child {
  border-bottom: 0;
}

.source-row:hover:not(:disabled) {
  color: var(--text-color);
  background: var(--hover-bg);
}

.source-row.disabled {
  opacity: 0.45;
  cursor: default;
}

.source-number {
  color: var(--subtitle-color);
  font-variant-numeric: tabular-nums;
}

.source-kind {
  color: var(--subtitle-color);
  font-size: 8.5px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.source-label {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.evidence-empty {
  height: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 30px;
  color: var(--subtitle-color);
  font-size: 11px;
  text-align: center;
}

.evidence-mark {
  font-size: 28px;
  opacity: 0.4;
}

.code-view {
  min-height: 100%;
  padding: 18px 28px 48px;
}

.code-meta {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  color: var(--subtitle-color);
  font-size: 9px;
  padding-bottom: 11px;
  border-bottom: 1px solid var(--border-color);
  margin-bottom: 12px;
}

.code-block {
  margin: 0;
  color: var(--text-color);
  font-family: ui-monospace, SFMono-Regular, SF Mono, Menlo, Consolas, monospace;
  font-size: 11px;
  line-height: 1.65;
  white-space: pre;
  overflow: visible;
}

.code-line {
  display: block;
  min-width: max-content;
}

.line-number {
  display: inline-block;
  width: 43px;
  margin-right: 13px;
  padding-right: 10px;
  border-right: 1px solid var(--border-color);
  color: var(--subtitle-color);
  text-align: right;
  user-select: none;
}

.doc-view {
  padding: 26px 40px 72px;
}

.doc-view :deep(.markdown-body) {
  max-width: 860px;
  font-size: 14px;
  line-height: 1.72;
}

.doc-view :deep(.markdown-body h1) {
  font-size: 21px;
}

.doc-view :deep(.markdown-body h2) {
  font-size: 16px;
}

@media (max-width: 1080px) {
  .ask-page {
    height: 100dvh;
    min-height: 0;
    overflow: hidden;
  }
  .ask-topbar {
    padding: 0 16px;
  }
  .mobile-switch {
    flex: 0 0 auto;
    display: flex;
    justify-content: center;
    gap: 2px;
    padding: 8px 16px 0;
    border-bottom: 1px solid var(--border-color);
  }
  .mobile-switch button {
    border: 0;
    border-bottom: 1px solid transparent;
    background: transparent;
    color: var(--subtitle-color);
    font-size: 10px;
    padding: 6px 9px;
    cursor: pointer;
  }
  .mobile-switch button.active {
    color: var(--title-color);
    border-bottom-color: var(--text-color);
  }
  .ask-main {
    display: flex;
    flex-direction: column;
    min-height: 0;
    flex: 1;
  }
  .chat-pane,
  .evidence-pane {
    height: 100%;
    min-height: 0;
  }
  .chat-pane.mobile-hide,
  .evidence-pane.mobile-hide {
    display: none;
  }
  .chat-pane {
    border-right: 0;
  }
  .conversation {
    padding: 26px 18px 20px;
  }
  .intro {
    margin-top: 9vh;
  }
  .intro h1 {
    font-size: 28px;
  }
  .message {
    max-width: 640px;
  }
  .composer {
    max-width: 640px;
  }
  .composer-wrap {
    padding: 8px 14px 14px;
  }
  .evidence-head {
    padding: 11px 18px;
  }
  .source-list {
    padding: 10px 18px 20px;
  }
  .code-view {
    padding: 14px 18px 40px;
  }
  .doc-view {
    padding: 20px 18px 50px;
  }
}
</style>
