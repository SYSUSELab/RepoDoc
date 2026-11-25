<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { cancelRun, getRun } from '../api'
import StageVisual from '../components/StageVisual.vue'
import ThemeToggle from '../components/ThemeToggle.vue'
import type { RunStatus } from '../types'

const route = useRoute()
const router = useRouter()
const runId = computed(() => String(route.params.id || ''))
const run = ref<RunStatus | null>(null)
const error = ref('')
const showLogs = ref(false)
let timer: ReturnType<typeof setInterval> | undefined

const terminal = computed(() =>
  ['completed', 'failed', 'cancelled', 'interrupted'].includes(run.value?.status || ''),
)
const completed = computed(() => run.value?.status === 'completed')
const logs = computed(() =>
  (run.value?.logs || [])
    .filter((line) => line.trim())
    .map((line) => line.replace(/^\[repodoc \+[^\]]+\]\s*/, '')),
)
const projectHref = computed(() =>
  run.value?.project_id ? `/project/${run.value.project_id}` : '',
)
const stageDescription = computed(() => {
  const stage = (run.value?.stage || '').toLowerCase()
  if (stage.includes('parse')) return 'Reading source files and extracting code structure.'
  if (stage.includes('component')) return 'Grouping files into coherent architectural components.'
  if (stage.includes('clust')) return 'Discovering component boundaries and relationships.'
  if (stage.includes('plan')) return 'Turning components into a readable document hierarchy.'
  if (stage.includes('document')) return 'Writing architecture pages from graph evidence.'
  if (stage.includes('final') || stage.includes('saving')) return 'Writing usage metadata and finishing the state.'
  if (stage.includes('complete')) return 'The documentation tree is ready.'
  return 'Preparing the repository graph and document pipeline.'
})

const refresh = async () => {
  try {
    run.value = await getRun(runId.value)
    if (terminal.value && timer) {
      clearInterval(timer)
      timer = undefined
    }
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  }
}

const stop = async () => {
  try {
    run.value = await cancelRun(runId.value)
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  }
}

const goProject = () => {
  if (projectHref.value) router.push(projectHref.value)
}
const goHome = () => router.push({ name: 'Home' })

onMounted(() => {
  void refresh()
  timer = setInterval(refresh, 1500)
})
onUnmounted(() => {
  if (timer) clearInterval(timer)
})
</script>

<template>
  <div class="generation-page">
    <header class="gen-topbar">
      <button class="back" @click="goHome" title="Home">←</button>
      <div class="gen-title">
        <span class="eyebrow">RepoDoc generation</span>
        <strong>{{ run?.remote_url || run?.repo_name || 'Preparing repository' }}</strong>
      </div>
      <div class="top-meta">
        <span class="status-pill" :class="run?.status">{{ run?.status || 'queued' }}</span>
        <ThemeToggle />
      </div>
    </header>

    <main class="process-main">
      <section class="process-hero">
        <div class="process-copy">
          <span class="eyebrow">{{ completed ? 'Documentation ready' : 'RepoDoc is thinking' }}</span>
          <h1>{{ run?.stage || 'Starting' }}</h1>
          <p>{{ stageDescription }}</p>

          <div class="process-progress">
            <div class="process-bar">
              <div class="process-fill" :style="{ width: `${run?.progress || 0}%` }"></div>
            </div>
            <strong>{{ Math.floor(run?.progress || 0) }}%</strong>
          </div>
          <div class="process-meta">
            <span class="mono">{{ runId }}</span>
            <span v-if="run?.repo_commit" class="mono">{{ run.repo_commit.slice(0, 8) }}</span>
          </div>

          <div class="process-actions">
            <button v-if="!terminal" class="btn" @click="stop">Stop generation</button>
            <button v-if="completed" class="btn btn-primary" @click="goProject">
              Open documentation →
            </button>
            <button class="log-toggle" @click="showLogs = !showLogs">
              {{ showLogs ? 'Hide log' : 'Show log' }}
              <span>{{ showLogs ? '▴' : '▾' }}</span>
            </button>
          </div>
          <p v-if="error || run?.error" class="error">{{ error || run?.error }}</p>
        </div>

        <StageVisual :stage="run?.stage || ''" />
      </section>

      <section v-if="showLogs" class="log-panel">
        <div class="log-head">
          <span class="eyebrow">Process log</span>
          <span class="muted">{{ logs.length }} lines</span>
        </div>
        <div class="log-lines">
          <p v-for="(line, index) in logs" :key="index">{{ line }}</p>
          <p v-if="!logs.length" class="muted">Waiting for process output…</p>
        </div>
      </section>
    </main>
  </div>
</template>

<style scoped>
.generation-page {
  min-height: 100vh;
  background: var(--container-bg);
  display: flex;
  flex-direction: column;
}

.gen-topbar {
  height: 64px;
  flex: 0 0 64px;
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 0 22px;
  border-bottom: 1px solid var(--border-color);
  background: var(--container-bg);
  z-index: 20;
}

.back {
  width: 36px;
  height: 36px;
  border: 1px solid var(--border-color);
  border-radius: 50%;
  background: transparent;
  color: var(--text-color);
  cursor: pointer;
  font-size: 16px;
}

.back:hover {
  background: var(--hover-bg);
}

.gen-title {
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

.gen-title strong {
  color: var(--title-color);
  font-size: 13px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.top-meta {
  margin-left: auto;
  display: flex;
  align-items: center;
  gap: 8px;
}

.status-pill {
  font-size: 9px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  padding: 4px 8px;
  border-radius: 999px;
  border: 1px solid var(--border-color);
  color: var(--subtitle-color);
}

.status-pill.running {
  color: var(--accent);
  border-color: color-mix(in srgb, var(--accent) 45%, transparent);
}

.status-pill.completed {
  color: var(--success);
  border-color: color-mix(in srgb, var(--success) 45%, transparent);
}

.status-pill.failed {
  color: var(--danger);
  border-color: color-mix(in srgb, var(--danger) 45%, transparent);
}

.process-main {
  flex: 1;
  width: min(1040px, 100%);
  margin: 0 auto;
  padding: 9vh 24px 48px;
  display: flex;
  flex-direction: column;
  gap: 22px;
}

.process-hero {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 56px;
  padding: 0 16px;
  animation: slideUpFade 0.5s ease-out both;
}

.process-copy {
  flex: 1;
  min-width: 0;
}

.process-copy h1 {
  margin: 7px 0 9px;
  color: var(--title-color);
  font-size: 34px;
  letter-spacing: -0.8px;
  line-height: 1.12;
}

.process-copy p {
  max-width: 520px;
  color: var(--secondary-text);
  font-size: 13px;
  line-height: 1.65;
  min-height: 42px;
}

.process-progress {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-top: 26px;
  max-width: 520px;
}

.process-bar {
  flex: 1;
  height: 6px;
  border-radius: 999px;
  background: var(--border-color);
  overflow: hidden;
}

.process-fill {
  height: 100%;
  border-radius: inherit;
  background: var(--title-color);
  transition: width 0.5s ease;
}

.process-progress strong {
  color: var(--title-color);
  font-size: 15px;
  font-variant-numeric: tabular-nums;
}

.process-meta {
  display: flex;
  gap: 12px;
  margin-top: 8px;
  color: var(--subtitle-color);
  font-size: 9px;
}

.process-actions {
  display: flex;
  align-items: center;
  gap: 14px;
  margin-top: 22px;
}

.process-actions .btn {
  border-radius: 7px;
  padding: 7px 11px;
  font-size: 11px;
}

.log-toggle {
  border: 0;
  background: transparent;
  color: var(--subtitle-color);
  font-size: 10px;
  cursor: pointer;
  padding: 6px 0;
  display: inline-flex;
  gap: 6px;
  align-items: center;
}

.log-toggle:hover {
  color: var(--text-color);
}

.error {
  margin-top: 12px;
  color: var(--danger);
  font-size: 11px;
}

.log-panel {
  border-top: 1px solid var(--border-color);
  padding: 16px 16px 0;
  animation: slideUpFade 0.25s ease-out both;
}

.log-head {
  display: flex;
  justify-content: space-between;
  margin-bottom: 8px;
}

.muted {
  color: var(--subtitle-color);
  font-size: 10px;
}

.log-lines {
  max-height: 260px;
  overflow-y: auto;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 10px;
  line-height: 1.7;
  color: var(--secondary-text);
  scrollbar-width: thin;
}

.log-lines p {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

@media (max-width: 760px) {
  .process-main {
    padding: 6vh 18px 36px;
  }
  .process-hero {
    flex-direction: column-reverse;
    gap: 26px;
    padding: 0;
    text-align: center;
  }
  .process-copy p {
    margin-left: auto;
    margin-right: auto;
  }
  .process-progress {
    margin-left: auto;
    margin-right: auto;
  }
  .process-actions {
    justify-content: center;
  }
  .log-panel {
    padding-left: 0;
    padding-right: 0;
  }
}
</style>
