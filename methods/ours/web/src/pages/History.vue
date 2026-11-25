<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { deleteProject, listProjects } from '../api'
import ProjectCard from '../components/ProjectCard.vue'
import TopControls from '../components/TopControls.vue'
import type { Project } from '../types'

const router = useRouter()
const projects = ref<Project[]>([])
const loading = ref(true)
const error = ref('')
const actionError = ref('')
const deleting = ref<string | null>(null)
const query = ref('')

const filtered = computed(() => {
  const q = query.value.trim().toLowerCase()
  if (!q) return projects.value
  return projects.value.filter(
    (project) =>
      project.repo.toLowerCase().includes(q) ||
      project.experiment.toLowerCase().includes(q) ||
      (project.commit || '').toLowerCase().includes(q),
  )
})

const load = async () => {
  loading.value = true
  error.value = ''
  try {
    projects.value = await listProjects()
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  } finally {
    loading.value = false
  }
}

const open = (id: string) => router.push({ name: 'Project', params: { id } })

const removeProject = async (project: Project) => {
  const ok = window.confirm(
    `Delete generated documentation for "${project.repo}"?\n\n` +
      'The cloned source repository in web-data/repos/ is kept.',
  )
  if (!ok) return
  deleting.value = project.id
  actionError.value = ''
  try {
    await deleteProject(project.id)
    projects.value = projects.value.filter((item) => item.id !== project.id)
  } catch (err) {
    actionError.value = err instanceof Error ? err.message : String(err)
  } finally {
    deleting.value = null
  }
}

onMounted(load)
</script>

<template>
  <div class="history-page">
    <TopControls />
    <main class="history-shell">
      <header class="history-header">
        <div>
          <span class="eyebrow">Workspace</span>
          <h1>Documentation <span v-if="!loading" class="count">{{ projects.length }}</span></h1>
          <p>Generated RepoDoc sites in this workspace.</p>
        </div>
      </header>

      <div class="history-tools">
        <label class="search-box">
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path
              fill="currentColor"
              d="M15.5 14h-.79l-.28-.27A6.47 6.47 0 0 0 16 9.5 6.5 6.5 0 1 0 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79L19 20.5 20.5 19l-5-5Zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14Z"
            />
          </svg>
          <input v-model="query" placeholder="Search repositories, commits…" />
          <button v-if="query" type="button" class="clear" @click="query = ''" title="Clear">×</button>
        </label>
        <button
          class="refresh"
          :class="{ spinning: loading }"
          :disabled="loading"
          @click="load"
          title="Refresh"
          aria-label="Refresh"
        >
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path
              fill="currentColor"
              d="M17.65 6.35A7.95 7.95 0 0 0 12 4a8 8 0 1 0 7.73 10h-2.08A6 6 0 1 1 12 6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35Z"
            />
          </svg>
        </button>
      </div>

      <p v-if="actionError" class="action-error">{{ actionError }}</p>

      <div v-if="loading" class="state">Loading projects…</div>
      <div v-else-if="error" class="state error">{{ error }}</div>
      <div v-else-if="filtered.length === 0" class="state empty">
        <span class="empty-mark">◇</span>
        <strong>No documentation yet</strong>
        <span>Start a new repository from the home page.</span>
      </div>
      <div v-else class="project-grid">
        <ProjectCard
          v-for="(project, index) in filtered"
          :key="project.id"
          :project="project"
          :style="{ animationDelay: `${index * 0.04}s` }"
          :deleting="deleting === project.id"
          @open="open(project.id)"
          @delete="removeProject"
        />
      </div>
    </main>
  </div>
</template>

<style scoped>
.history-page {
  min-height: 100vh;
  background: var(--container-bg);
}

.history-shell {
  max-width: 1180px;
  margin: 0 auto;
  padding: 104px 24px 60px;
}

.history-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  animation: slideUpFade 0.55s ease-out both;
  margin-bottom: 28px;
}

.eyebrow {
  color: var(--subtitle-color);
  font-size: 9px;
  text-transform: uppercase;
  letter-spacing: 1px;
}

h1 {
  color: var(--title-color);
  font-size: 34px;
  font-weight: 680;
  letter-spacing: -0.9px;
  margin: 5px 0 5px;
  display: flex;
  align-items: center;
  gap: 10px;
}

.count {
  font-size: 11px;
  font-weight: 500;
  letter-spacing: 0;
  color: var(--subtitle-color);
  border: 1px solid var(--border-color);
  border-radius: 999px;
  padding: 1px 8px;
}

.history-header p {
  color: var(--subtitle-color);
  font-size: 11px;
}

.history-tools {
  display: flex;
  gap: 10px;
  margin-bottom: 22px;
  max-width: 470px;
  animation: slideUpFade 0.55s 0.06s ease-out both;
}

.search-box {
  flex: 1;
  min-width: 0;
  display: flex;
  align-items: center;
  gap: 9px;
  border: 1px solid var(--border-color);
  border-radius: 11px;
  background: var(--input-bg);
  padding: 0 12px;
  height: 42px;
  transition:
    border-color 0.2s ease,
    box-shadow 0.2s ease;
}

.search-box:focus-within {
  border-color: color-mix(in srgb, var(--accent) 70%, var(--border-color));
  box-shadow: 0 0 0 4px var(--focus-color);
}

.search-box svg {
  width: 15px;
  height: 15px;
  color: var(--subtitle-color);
  flex: 0 0 auto;
}

.search-box input {
  flex: 1;
  min-width: 0;
  border: 0;
  outline: 0;
  background: transparent;
  color: var(--text-color);
  font-size: 12px;
}

.search-box input::placeholder {
  color: var(--placeholder-color);
}

.clear {
  border: 0;
  background: transparent;
  color: var(--subtitle-color);
  font-size: 17px;
  line-height: 1;
  cursor: pointer;
  padding: 2px;
}

.clear:hover {
  color: var(--text-color);
}

.refresh {
  width: 42px;
  height: 42px;
  flex: 0 0 42px;
  border: 1px solid var(--border-color);
  border-radius: 11px;
  background: var(--card-bg);
  color: var(--secondary-text);
  cursor: pointer;
  display: grid;
  place-items: center;
  transition: all 0.2s ease;
}

.refresh svg {
  width: 16px;
  height: 16px;
}

.refresh:hover:not(:disabled) {
  color: var(--text-color);
  background: var(--hover-bg);
  border-color: color-mix(in srgb, var(--accent) 35%, var(--border-color));
}

.refresh:disabled {
  cursor: wait;
  opacity: 0.65;
}

.refresh.spinning svg {
  animation: spin 0.9s linear infinite;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

.project-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
  gap: 16px;
}

.state {
  padding: 40px;
  text-align: center;
  color: var(--subtitle-color);
  font-size: 13px;
  border: 1px dashed var(--border-color);
  border-radius: 14px;
}

.state.error {
  color: var(--danger);
}

.action-error {
  color: var(--danger);
  font-size: 11px;
  margin: -8px 0 14px;
}

.state.empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 5px;
  padding: 60px 20px;
}

.state.empty strong {
  color: var(--text-color);
  font-size: 14px;
}

.state.empty span:last-child {
  font-size: 12px;
}

.empty-mark {
  font-size: 28px;
  color: var(--accent);
  opacity: 0.7;
  margin-bottom: 4px;
}
</style>
