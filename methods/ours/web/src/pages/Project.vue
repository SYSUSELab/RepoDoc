<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { getPage, getProject } from '../api'
import DocTree from '../components/DocTree.vue'
import MarkdownContent from '../components/MarkdownContent.vue'
import ThemeToggle from '../components/ThemeToggle.vue'
import type { ProjectDetail } from '../types'

const route = useRoute()
const router = useRouter()
const projectId = computed(() => String(route.params.id || ''))
const detail = ref<ProjectDetail | null>(null)
const activePath = ref('')
const markdown = ref('')
const filter = ref('')
const loading = ref(true)
const pageLoading = ref(false)
const error = ref('')

const activePage = computed(() =>
  detail.value?.pages.find((page) => page.path === activePath.value),
)
const project = computed(() => detail.value?.project)

const shortCommit = (commit?: string | null) => (commit ? commit.slice(0, 8) : 'unknown')
const formatNumber = (value?: number | null) =>
  value == null ? '—' : new Intl.NumberFormat('en-US').format(value)
const formatCost = (value?: number | null) =>
  value == null ? '—' : `$${value.toFixed(4)}`

const chooseInitialPage = () => {
  if (!detail.value) return ''
  const overview = detail.value.pages.find((page) => page.path === 'overview.md')
  return overview?.path || detail.value.pages[0]?.path || ''
}

const loadPage = async (path: string) => {
  if (!path || !detail.value) return
  pageLoading.value = true
  error.value = ''
  try {
    activePath.value = path
    markdown.value = await getPage(detail.value.project.id, path)
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  } finally {
    pageLoading.value = false
  }
}

const load = async () => {
  if (!projectId.value) return
  loading.value = true
  error.value = ''
  try {
    detail.value = await getProject(projectId.value)
    const initial = chooseInitialPage()
    if (initial) await loadPage(initial)
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  } finally {
    loading.value = false
  }
}

const goBack = () => router.push({ name: 'Home' })
const onNavigate = (path: string) => {
  const clean = (path.split('#')[0] || '').replace(/^\.\//, '').split('/').pop() || ''
  if (clean) void loadPage(clean)
}

watch(() => route.params.id, load)
onMounted(load)
</script>

<template>
  <div class="project-page">
    <header class="project-topbar">
      <button class="back-btn" @click="goBack" title="Back to projects">←</button>
      <div v-if="project" class="repo-title">
        <span class="repo-name">{{ project.repo }}</span>
        <span class="repo-sub">{{ project.experiment }} / {{ project.system }}</span>
      </div>
      <div class="topbar-meta">
        <router-link
          v-if="project"
          class="ask-link"
          :to="{ name: 'Ask', params: { id: project.id } }"
        >
          Ask RepoDoc
        </router-link>
        <span v-if="project?.language" class="pill">{{ project.language }}</span>
        <span class="mono commit">{{ shortCommit(project?.commit) }}</span>
        <ThemeToggle />
      </div>
    </header>

    <div v-if="loading" class="state-page">Loading project…</div>
    <div v-else-if="error" class="state-page error">{{ error }}</div>

    <div v-else-if="detail && project" class="project-shell">
      <aside class="sidebar">
        <div class="sidebar-head">
          <span class="eyebrow">Documentation</span>
          <span class="page-count">{{ project.pages }} pages</span>
        </div>
        <div class="sidebar-body">
          <input v-model="filter" class="filter-input" placeholder="Filter pages…" />
          <DocTree
            :nodes="[detail.doc_tree]"
            :active-path="activePath"
            :filter="filter"
            @select="loadPage"
          />
        </div>
      </aside>

      <main class="doc-pane">
        <div class="doc-toolbar">
          <div class="breadcrumb">
            <span>{{ activePage?.doc_type || 'document' }}</span>
            <span class="sep">/</span>
            <span class="mono">{{ activePath }}</span>
          </div>
          <div class="doc-stats">
            <span><strong>{{ formatNumber(detail.graph.files) }}</strong> files</span>
            <span><strong>{{ formatNumber(detail.graph.nodes) }}</strong> nodes</span>
            <span><strong>{{ formatNumber(project.usage?.tokens?.total) }}</strong> tokens</span>
            <span><strong>{{ formatCost(project.usage?.cost_usd) }}</strong> cost</span>
          </div>
        </div>

        <div class="doc-scroll">
          <div v-if="pageLoading" class="page-loading">Loading page…</div>
          <MarkdownContent v-else :content="markdown" @navigate="onNavigate" />
        </div>
      </main>
    </div>
  </div>
</template>

<style scoped>
.project-page {
  height: 100vh;
  background: var(--container-bg);
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

.project-topbar {
  height: 64px;
  flex: 0 0 64px;
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 0 22px;
  border-bottom: 1px solid var(--border-color);
  background: color-mix(in srgb, var(--container-bg) 90%, transparent);
  backdrop-filter: blur(12px);
  z-index: 100;
}

.back-btn {
  width: 36px;
  height: 36px;
  border: 1px solid var(--border-color);
  border-radius: 50%;
  background: var(--card-bg);
  color: var(--text-color);
  cursor: pointer;
  font-size: 16px;
  transition: all 0.2s ease;
}

.back-btn:hover {
  background: var(--hover-bg);
  transform: translateX(-1px);
}

.repo-title {
  min-width: 0;
  display: flex;
  flex-direction: column;
  line-height: 1.25;
}

.repo-name {
  font-size: 14px;
  font-weight: 650;
  color: var(--title-color);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.repo-sub {
  font-size: 10px;
  color: var(--subtitle-color);
}

.topbar-meta {
  margin-left: auto;
  display: flex;
  align-items: center;
  gap: 8px;
}

.ask-link {
  color: var(--text-color);
  border: 1px solid var(--border-color);
  border-radius: 7px;
  padding: 5px 9px;
  font-size: 10px;
  font-weight: 500;
  transition: all 0.18s ease;
}

.ask-link:hover {
  background: var(--hover-bg);
  text-decoration: none;
}

.pill {
  font-size: 9px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: var(--secondary-text);
  border: 1px solid var(--border-color);
  border-radius: 999px;
  padding: 3px 8px;
}

.commit {
  font-size: 10px;
  color: var(--subtitle-color);
}

.project-shell {
  flex: 1;
  min-height: 0;
  display: flex;
  max-width: 1440px;
  width: 100%;
  margin: 0 auto;
}

.sidebar {
  width: 300px;
  flex: 0 0 300px;
  min-height: 0;
  border-right: 1px solid var(--border-color);
  display: flex;
  flex-direction: column;
  padding: 20px 14px 14px;
  background: color-mix(in srgb, var(--card-bg) 38%, transparent);
}

.sidebar-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  margin-bottom: 10px;
}

.eyebrow {
  font-size: 9px;
  text-transform: uppercase;
  letter-spacing: 0.8px;
  color: var(--subtitle-color);
}

.page-count {
  font-size: 9px;
  color: var(--subtitle-color);
  border: 1px solid var(--border-color);
  border-radius: 999px;
  padding: 1px 7px;
}

.sidebar-body {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  scrollbar-width: none;
}

.sidebar-body::-webkit-scrollbar {
  width: 0;
}

.filter-input {
  width: 100%;
  border: 1px solid var(--border-color);
  border-radius: 8px;
  background: var(--input-bg);
  color: var(--text-color);
  padding: 8px 10px;
  font-size: 11px;
  outline: none;
  margin-bottom: 9px;
  box-shadow: 0 1px 3px transparent;
}

.filter-input:focus {
  border-color: var(--accent);
  box-shadow: 0 0 0 3px var(--focus-color);
}

.doc-pane {
  flex: 1;
  min-width: 0;
  min-height: 0;
  display: flex;
  flex-direction: column;
}

.doc-toolbar {
  flex: 0 0 auto;
  min-height: 48px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 8px 34px;
  border-bottom: 1px solid var(--border-color);
  background: var(--container-bg);
}

.breadcrumb {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 10px;
  color: var(--subtitle-color);
  min-width: 0;
}

.breadcrumb .mono {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.sep {
  opacity: 0.5;
}

.doc-stats {
  display: flex;
  align-items: center;
  gap: 12px;
  color: var(--subtitle-color);
  font-size: 10px;
  flex: 0 0 auto;
}

.doc-stats strong {
  color: var(--text-color);
  font-weight: 600;
}

.doc-scroll {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  padding: 28px 54px 70px;
  scrollbar-width: thin;
}

.page-loading,
.state-page {
  color: var(--subtitle-color);
  font-size: 13px;
}

.page-loading {
  padding: 30px;
  text-align: center;
}

.state-page {
  display: grid;
  place-items: center;
  flex: 1;
  padding: 60px;
}

.state-page.error {
  color: var(--danger);
}

@media (max-width: 900px) {
  .sidebar {
    width: 240px;
    flex-basis: 240px;
  }
  .doc-scroll {
    padding: 24px 26px 60px;
  }
  .doc-stats {
    display: none;
  }
}

@media (max-width: 700px) {
  .project-shell {
    flex-direction: column;
  }
  .sidebar {
    width: 100%;
    flex: 0 0 auto;
    max-height: 260px;
    border-right: 0;
    border-bottom: 1px solid var(--border-color);
  }
  .doc-pane {
    min-height: 0;
  }
  .doc-toolbar {
    padding: 8px 18px;
  }
}
</style>
