<script setup lang="ts">
import { onUnmounted, ref, watch } from 'vue'
import { searchGithub } from '../api'
import type { RepoCandidate } from '../types'

const props = defineProps<{ busy: boolean }>()
const emit = defineEmits<{ submit: [payload: { source: string }] }>()

const source = ref('')
const candidates = ref<RepoCandidate[]>([])
const showCandidates = ref(false)
const searching = ref(false)
const error = ref('')
let searchTimer: ReturnType<typeof setTimeout> | undefined

const isLocalPath = (value: string) =>
  value.startsWith('/') ||
  value.startsWith('.') ||
  value.startsWith('~') ||
  value.startsWith('data/') ||
  value.includes('\\')

const looksLikeUrl = (value: string) =>
  /^https?:\/\//i.test(value) || value.startsWith('git@') || value.startsWith('github.com/')

const search = async (query: string) => {
  const q = query.trim()
  if (q.length < 2 || isLocalPath(q) || looksLikeUrl(q)) {
    candidates.value = []
    showCandidates.value = false
    return
  }
  searching.value = true
  try {
    candidates.value = await searchGithub(q)
    showCandidates.value = candidates.value.length > 0
  } catch {
    candidates.value = []
    showCandidates.value = false
  } finally {
    searching.value = false
  }
}

watch(source, (value) => {
  error.value = ''
  if (searchTimer) clearTimeout(searchTimer)
  if (isLocalPath(value) || looksLikeUrl(value) || !value.trim()) {
    candidates.value = []
    showCandidates.value = false
    return
  }
  searchTimer = setTimeout(() => void search(value), 320)
})

const choose = (candidate: RepoCandidate) => {
  source.value = candidate.path || candidate.url || candidate.full_name
  candidates.value = []
  showCandidates.value = false
}

const submit = () => {
  error.value = ''
  const value = source.value.trim()
  if (!value) {
    error.value = 'Enter a GitHub repository URL.'
    return
  }
  showCandidates.value = false
  emit('submit', { source: value })
}

onUnmounted(() => {
  if (searchTimer) clearTimeout(searchTimer)
})
</script>

<template>
  <form class="run-form" @submit.prevent="submit">
    <div class="input-shell">
      <span class="input-icon">⌘</span>
      <input
        v-model="source"
        class="repo-input"
        placeholder="GitHub repository URL"
        autocomplete="off"
        @focus="showCandidates = candidates.length > 0"
      />
      <span v-if="searching" class="searching">searching…</span>
      <button class="generate-btn" type="submit" :disabled="props.busy" title="Generate">
        <span v-if="props.busy" class="spinner"></span>
        <span v-else>→</span>
      </button>
    </div>

    <div v-if="showCandidates && candidates.length" class="candidate-menu">
      <button
        v-for="candidate in candidates"
        :key="candidate.full_name"
        type="button"
        class="candidate"
        @mousedown.prevent="choose(candidate)"
      >
        <span class="candidate-name">{{ candidate.full_name }}</span>
        <span class="candidate-meta">
          <span v-if="candidate.language">{{ candidate.language }}</span>
          <span v-if="candidate.stars">★ {{ candidate.stars }}</span>
          <span class="candidate-source">{{ candidate.source }}</span>
        </span>
        <span v-if="candidate.description" class="candidate-desc">
          {{ candidate.description }}
        </span>
      </button>
    </div>

    <p v-if="error" class="form-error">{{ error }}</p>
  </form>
</template>

<style scoped>
.run-form {
  width: 100%;
  max-width: 760px;
  margin: 0 auto 22px;
  animation: slideUpFade 0.7s 0.1s ease-out both;
  position: relative;
  z-index: 20;
}

.input-shell {
  display: flex;
  align-items: center;
  background: var(--input-bg);
  border: 1px solid var(--border-color);
  border-radius: 12px;
  box-shadow: 0 4px 14px var(--shadow-color);
  transition: all 0.2s ease;
  overflow: hidden;
}

.input-shell:focus-within {
  border-color: var(--accent);
  box-shadow: 0 4px 18px var(--focus-color);
}

.input-icon {
  padding: 0 14px;
  color: var(--icon-color);
  font-size: 16px;
}

.repo-input {
  flex: 1;
  min-width: 0;
  border: 0;
  outline: 0;
  padding: 14px 8px;
  color: var(--text-color);
  background: transparent;
  font-size: 14px;
}

.repo-input::placeholder {
  color: var(--placeholder-color);
}

.searching {
  color: var(--subtitle-color);
  font-size: 10px;
  white-space: nowrap;
}

.generate-btn {
  width: 50px;
  align-self: stretch;
  border: 0;
  border-left: 1px solid var(--border-color);
  background: transparent;
  color: var(--secondary-text);
  cursor: pointer;
  font-size: 18px;
  transition: all 0.2s ease;
}

.generate-btn:hover:not(:disabled) {
  color: var(--text-color);
  background: var(--hover-bg);
}

.generate-btn:disabled {
  cursor: wait;
  opacity: 0.55;
}

.spinner {
  display: inline-block;
  width: 14px;
  height: 14px;
  border: 2px solid var(--border-color);
  border-top-color: var(--accent);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

.candidate-menu {
  position: absolute;
  top: 54px;
  left: 0;
  right: 0;
  background: var(--card-bg);
  border: 1px solid var(--border-color);
  border-radius: 10px;
  box-shadow: 0 10px 30px var(--shadow-color);
  overflow: hidden;
  z-index: 50;
}

.candidate {
  width: 100%;
  border: 0;
  border-bottom: 1px solid var(--border-color);
  background: transparent;
  color: var(--text-color);
  padding: 10px 14px;
  text-align: left;
  cursor: pointer;
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 2px 12px;
}

.candidate:last-child {
  border-bottom: 0;
}

.candidate:hover {
  background: var(--hover-bg);
}

.candidate-name {
  font-size: 13px;
  font-weight: 600;
}

.candidate-meta {
  display: flex;
  gap: 8px;
  color: var(--subtitle-color);
  font-size: 10px;
}

.candidate-source {
  text-transform: uppercase;
  letter-spacing: 0.4px;
}

.candidate-desc {
  grid-column: 1 / -1;
  color: var(--secondary-text);
  font-size: 11px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.form-error {
  margin-top: 8px;
  color: var(--danger);
  font-size: 12px;
  text-align: center;
}
</style>
