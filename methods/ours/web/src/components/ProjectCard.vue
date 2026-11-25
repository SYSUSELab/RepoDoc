<script setup lang="ts">
import { computed } from 'vue'
import type { Project } from '../types'

const props = defineProps<{ project: Project; deleting?: boolean }>()
const emit = defineEmits<{
  open: [project: Project]
  delete: [project: Project]
}>()

const displayName = computed(() => props.project.repo.replace('__', ' / '))
const initials = computed(() => {
  const parts = displayName.value.split(/[\s/_-]+/).filter(Boolean)
  return (parts[0]?.[0] || '?') + (parts[1]?.[0] || '')
})
const sourceLabel = computed(() =>
  props.project.source === 'remote' ? 'GitHub' : 'Local',
)

const formatNumber = (value?: number | null) =>
  value == null ? '—' : new Intl.NumberFormat('en-US').format(value)
const formatCost = (value?: number | null) =>
  value == null ? '—' : `$${value.toFixed(4)}`
const shortCommit = (commit?: string | null) => (commit ? commit.slice(0, 8) : '—')
const formatDate = (value?: string | null) => {
  if (!value) return '—'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  })
}
</script>

<template>
  <article class="project-card" @click="emit('open', props.project)">
    <div class="card-top">
      <div class="monogram">{{ initials }}</div>
      <div class="identity">
        <h3 :title="props.project.repo">{{ displayName }}</h3>
        <p :title="props.project.experiment">{{ props.project.experiment }}</p>
      </div>
      <div class="card-actions">
        <button
          class="delete-btn"
          :class="{ busy: props.deleting }"
          type="button"
          title="Delete generated docs"
          aria-label="Delete generated docs"
          :disabled="props.deleting"
          @click.stop="emit('delete', props.project)"
        >
          <span v-if="props.deleting">…</span>
          <svg v-else viewBox="0 0 24 24" aria-hidden="true">
            <path
              fill="currentColor"
              d="M7 21a2 2 0 0 1-2-2V6h12v13a2 2 0 0 1-2 2H7Zm8-17h3V2H6v2h3v2h6V4Zm-7 2V4h6v2H8Z"
            />
          </svg>
        </button>
        <span class="open-mark">↗</span>
      </div>
    </div>

    <div class="metrics">
      <div>
        <strong>{{ props.project.pages }}</strong>
        <span>pages</span>
      </div>
      <div>
        <strong>{{ formatNumber(props.project.usage?.tokens?.total) }}</strong>
        <span>tokens</span>
      </div>
      <div>
        <strong>{{ formatCost(props.project.usage?.cost_usd) }}</strong>
        <span>cost</span>
      </div>
    </div>

    <div class="card-foot">
      <span class="language"><i></i>{{ props.project.language || 'unknown' }}</span>
      <span class="mono commit">{{ shortCommit(props.project.commit) }}</span>
      <span class="date">{{ formatDate(props.project.updated_at) }}</span>
      <span class="source">{{ sourceLabel }}</span>
    </div>
  </article>
</template>

<style scoped>
.project-card {
  background: var(--card-bg);
  border: 1px solid var(--border-color);
  border-radius: 16px;
  padding: 18px 20px 15px;
  cursor: pointer;
  transition:
    transform 0.22s ease,
    border-color 0.22s ease;
  animation: flyIn 0.35s ease-out both;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.project-card:hover {
  transform: translateY(-2px);
  border-color: color-mix(in srgb, var(--text-color) 22%, var(--border-color));
}

.card-top {
  display: flex;
  align-items: center;
  gap: 11px;
  min-width: 0;
}

.monogram {
  width: 38px;
  height: 38px;
  flex: 0 0 38px;
  border-radius: 11px;
  display: grid;
  place-items: center;
  background: var(--hover-bg);
  color: var(--secondary-text);
  font-size: 13px;
  font-weight: 700;
  letter-spacing: -0.3px;
  text-transform: uppercase;
}

.identity {
  min-width: 0;
  flex: 1;
}

.identity h3 {
  color: var(--title-color);
  font-size: 14px;
  font-weight: 650;
  line-height: 1.3;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.identity p {
  margin-top: 2px;
  color: var(--subtitle-color);
  font-size: 10px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.card-actions {
  display: flex;
  align-items: center;
  gap: 4px;
  flex: 0 0 auto;
}

.delete-btn {
  width: 27px;
  height: 27px;
  border: 0;
  border-radius: 50%;
  display: grid;
  place-items: center;
  background: transparent;
  color: var(--subtitle-color);
  cursor: pointer;
  opacity: 0.55;
  transition: all 0.2s ease;
}

.project-card:hover .delete-btn,
.delete-btn:focus-visible {
  opacity: 1;
}

.delete-btn:hover:not(:disabled) {
  color: var(--danger);
  background: color-mix(in srgb, var(--danger) 10%, transparent);
}

.delete-btn.busy {
  opacity: 1;
  cursor: wait;
  font-size: 12px;
}

.delete-btn svg {
  width: 13px;
  height: 13px;
}

.open-mark {
  width: 28px;
  height: 28px;
  flex: 0 0 28px;
  border-radius: 50%;
  display: grid;
  place-items: center;
  color: var(--secondary-text);
  background: var(--hover-bg);
  font-size: 14px;
  transition: transform 0.22s ease;
}

.project-card:hover .open-mark {
  transform: translate(1px, -1px);
}

.metrics {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  border-top: 1px solid var(--border-color);
  border-bottom: 1px solid var(--border-color);
  padding: 11px 0;
}

.metrics div {
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 1px;
  padding-left: 12px;
  border-left: 1px solid var(--border-color);
}

.metrics div:first-child {
  padding-left: 0;
  border-left: 0;
}

.metrics strong {
  color: var(--text-color);
  font-size: 13px;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
  overflow: hidden;
  text-overflow: ellipsis;
}

.metrics span {
  color: var(--subtitle-color);
  font-size: 9px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.card-foot {
  display: flex;
  align-items: center;
  gap: 9px;
  color: var(--subtitle-color);
  font-size: 9px;
  min-width: 0;
}

.language {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  text-transform: uppercase;
  letter-spacing: 0.45px;
  color: var(--secondary-text);
}

.language i {
  width: 5px;
  height: 5px;
  border-radius: 50%;
  background: var(--subtitle-color);
}

.commit {
  font-size: 9px;
}

.date {
  margin-left: auto;
  white-space: nowrap;
}

.source {
  border: 1px solid var(--border-color);
  border-radius: 999px;
  padding: 1px 6px;
  font-size: 8px;
  text-transform: uppercase;
  letter-spacing: 0.4px;
}
</style>
