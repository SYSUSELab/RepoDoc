<script setup lang="ts">
import { computed, ref } from 'vue'
import type { DocNode } from '../types'

defineOptions({ name: 'DocTree' })

const props = defineProps<{
  nodes: DocNode[]
  activePath: string
  filter?: string
  depth?: number
}>()

const emit = defineEmits<{ select: [path: string] }>()
const collapsed = ref<Set<string>>(new Set())

const hasMatch = (node: DocNode): boolean => {
  const needle = (props.filter || '').trim().toLowerCase()
  if (!needle) return true
  return (
    node.title.toLowerCase().includes(needle) ||
    node.path.toLowerCase().includes(needle) ||
    node.children.some(hasMatch)
  )
}

const visibleNodes = computed(() => props.nodes.filter(hasMatch))

const toggle = (node: DocNode) => {
  const next = new Set(collapsed.value)
  if (next.has(node.path)) next.delete(node.path)
  else next.add(node.path)
  collapsed.value = next
}

const expanded = (node: DocNode) =>
  Boolean(props.filter?.trim()) || !collapsed.value.has(node.path)
</script>

<template>
  <div class="doc-tree">
    <template v-for="node in visibleNodes" :key="node.path">
      <button
        class="doc-row"
        :class="{ active: node.path === props.activePath, parent: node.children.length }"
        :style="{ paddingLeft: `${8 + Math.max(0, props.depth || 0) * 14}px` }"
        @click="emit('select', node.path)"
      >
        <span
          class="twisty"
          :class="{ invisible: !node.children.length }"
          @click.stop="toggle(node)"
        >
          {{ expanded(node) ? '▾' : '▸' }}
        </span>
        <span class="doc-icon">{{ node.children.length ? '▦' : '▤' }}</span>
        <span class="doc-label" :title="node.path">{{ node.title }}</span>
        <span v-if="node.children.length" class="child-count">{{ node.children.length }}</span>
        <span v-else class="leaf-dot"></span>
      </button>
      <div v-if="node.children.length && expanded(node)" class="doc-children">
        <DocTree
          :nodes="node.children"
          :active-path="props.activePath"
          :filter="props.filter"
          :depth="(props.depth || 0) + 1"
          @select="emit('select', $event)"
        />
      </div>
    </template>
  </div>
</template>

<style scoped>
.doc-tree {
  display: flex;
  flex-direction: column;
  gap: 1px;
}

.doc-row {
  width: 100%;
  min-width: 0;
  display: flex;
  align-items: center;
  gap: 5px;
  padding: 6px 8px;
  border: 0;
  border-radius: 6px;
  background: transparent;
  color: var(--text-color);
  cursor: pointer;
  text-align: left;
  font-size: 12px;
  transition:
    background 0.15s ease,
    color 0.15s ease;
  position: relative;
}

.doc-row:hover {
  background: var(--hover-bg);
}

.doc-row.active {
  background: var(--accent-soft);
  color: var(--title-color);
}

.doc-row.active::before {
  content: '';
  position: absolute;
  left: 0;
  top: 8px;
  bottom: 8px;
  width: 2px;
  border-radius: 2px;
  background: var(--accent);
}

.doc-row.parent {
  font-weight: 600;
}

.doc-children {
  margin-left: 10px;
  padding-left: 5px;
  border-left: 1px solid var(--border-color);
}

.twisty {
  width: 12px;
  flex: 0 0 12px;
  color: var(--subtitle-color);
  font-size: 9px;
  text-align: center;
}

.twisty.invisible {
  opacity: 0;
  pointer-events: none;
}

.doc-icon {
  flex: 0 0 auto;
  color: var(--subtitle-color);
  font-size: 11px;
}

.doc-label {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  flex: 1;
}

.child-count {
  color: var(--subtitle-color);
  font-size: 9px;
  font-variant-numeric: tabular-nums;
  background: var(--hover-bg);
  padding: 0 4px;
  border-radius: 999px;
}

.leaf-dot {
  width: 3px;
  height: 3px;
  border-radius: 50%;
  background: var(--border-color);
  flex: 0 0 auto;
}

.doc-row.active .leaf-dot {
  background: var(--accent);
}
</style>
