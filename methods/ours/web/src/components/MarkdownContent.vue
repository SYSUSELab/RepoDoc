<script setup lang="ts">
import { inject, nextTick, onMounted, ref, watch, type Ref } from 'vue'
import MarkdownIt from 'markdown-it'
import anchor from 'markdown-it-anchor'
import mermaid from 'mermaid'
import hljs from 'highlight.js'

const props = defineProps<{ content: string }>()
const emit = defineEmits<{
  headings: [headings: { level: number; title: string; id: string }[]]
  navigate: [path: string]
}>()

const theme = inject<{ isDarkMode: Ref<boolean> } | null>('theme', null)
const root = ref<HTMLElement | null>(null)
const html = ref('')
const zoomHtml = ref('')
const zoomVisible = ref(false)

const escapeHtml = (value: string) =>
  value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')

const md = new MarkdownIt({
  html: true,
  linkify: true,
  typographer: true,
  highlight: (str, lang) => {
    if (lang && hljs.getLanguage(lang)) {
      try {
        return `<pre class="hljs"><code>${hljs.highlight(str, {
          language: lang,
          ignoreIllegals: true,
        }).value}</code></pre>`
      } catch {
        // fall through
      }
    }
    return `<pre class="hljs"><code>${escapeHtml(str)}</code></pre>`
  },
}).use(anchor, { permalink: false })

const defaultFence = md.renderer.rules.fence?.bind(md.renderer.rules)
md.renderer.rules.fence = (tokens, idx, options, env, self) => {
  const token = tokens[idx]
  if (token?.info.trim() === 'mermaid') {
    return `<div class="mermaid">${escapeHtml(token.content || '')}</div>`
  }
  return defaultFence
    ? defaultFence(tokens, idx, options, env, self)
    : self.renderToken(tokens, idx, options)
}

const render = () => {
  const env: Record<string, unknown> = {}
  html.value = md.render(props.content || '', env)
  const tokens = md.parse(props.content || '', env)
  const headings: { level: number; title: string; id: string }[] = []
  for (let i = 0; i < tokens.length; i += 1) {
    const token = tokens[i]
    if (token?.type !== 'heading_open') continue
    const level = Number(token.tag.replace('h', '')) || 1
    const attrs = (token.attrs || []) as [string, string][]
    const id = attrs.find((attr) => attr[0] === 'id')?.[1] || ''
    const inline = tokens[i + 1]
    const title = inline?.type === 'inline' ? String(inline.content || '') : ''
    if (id && title) headings.push({ level, title, id })
  }
  emit('headings', headings)
  nextTick(() => {
    void renderMermaid()
  })
}

const renderMermaid = async () => {
  if (!root.value) return
  const nodes = root.value.querySelectorAll<HTMLElement>('.mermaid')
  if (!nodes.length) return
  try {
    mermaid.initialize({
      startOnLoad: false,
      securityLevel: 'loose',
      theme: document.body.classList.contains('dark') ? 'dark' : 'default',
    })
    await mermaid.run({ nodes })
  } catch (err) {
    console.warn('mermaid render failed', err)
  }
}

const onClick = (event: MouseEvent) => {
  const target = event.target as HTMLElement
  const diagram = target.closest('.mermaid') as HTMLElement | null
  if (diagram) {
    zoomHtml.value = diagram.innerHTML
    zoomVisible.value = true
    return
  }
  const link = target.closest('a') as HTMLAnchorElement | null
  if (!link) return
  const href = link.getAttribute('href') || ''
  if (!href || href.startsWith('#') || /^https?:\/\//.test(href)) return
  if (!href.endsWith('.md')) return
  event.preventDefault()
  emit('navigate', decodeURIComponent(href))
}

const closeZoom = () => {
  zoomVisible.value = false
  zoomHtml.value = ''
}

const scrollToHeading = async (id: string) => {
  await nextTick()
  if (!root.value) return
  const element = root.value.querySelector(`#${CSS.escape(id)}`)
  element?.scrollIntoView({ behavior: 'smooth', block: 'start' })
}

defineExpose({ scrollToHeading })
watch(() => props.content, render)
if (theme) watch(() => theme.isDarkMode.value, render)
onMounted(render)
</script>

<template>
  <div ref="root" class="markdown-body" @click="onClick" v-html="html"></div>

  <div v-if="zoomVisible" class="zoom-modal" @click.self="closeZoom">
    <button class="zoom-close" @click="closeZoom" aria-label="Close">×</button>
    <div class="zoom-content" v-html="zoomHtml"></div>
  </div>
</template>

<style scoped>
.markdown-body {
  color: var(--text-color);
  font-family: 'Noto Serif SC', 'Iowan Old Style', 'Palatino Linotype', Georgia, serif;
  font-size: 16px;
  line-height: 1.72;
  max-width: 860px;
  margin: 0 auto;
}

.markdown-body :deep(h1),
.markdown-body :deep(h2),
.markdown-body :deep(h3),
.markdown-body :deep(h4) {
  color: var(--title-color);
  font-weight: 650;
  line-height: 1.3;
  letter-spacing: -0.15px;
  scroll-margin-top: 88px;
}

.markdown-body :deep(h1) {
  font-size: 30px;
  margin: 0 0 18px;
}

.markdown-body :deep(h2) {
  font-size: 21px;
  margin: 32px 0 12px;
  padding-bottom: 5px;
  border-bottom: 1px solid var(--border-color);
}

.markdown-body :deep(h3) {
  font-size: 17px;
  margin: 24px 0 8px;
}

.markdown-body :deep(p) {
  margin: 0 0 12px;
}

.markdown-body :deep(ul),
.markdown-body :deep(ol) {
  margin: 0 0 14px;
  padding-left: 1.6em;
}

.markdown-body :deep(li) {
  margin: 4px 0;
}

.markdown-body :deep(a) {
  color: var(--link-color);
  text-decoration: none;
}

.markdown-body :deep(a:hover) {
  text-decoration: underline;
}

.markdown-body :deep(code) {
  font-family: ui-monospace, SFMono-Regular, SF Mono, Menlo, Consolas, monospace;
  font-size: 0.84em;
  background: var(--code-bg);
  padding: 0.15em 0.38em;
  border-radius: 5px;
}

.markdown-body :deep(pre) {
  background: var(--code-bg);
  border: 1px solid var(--border-color);
  border-radius: 8px;
  padding: 14px 16px;
  overflow: auto;
  margin: 0 0 16px;
  font-size: 12px;
  line-height: 1.55;
}

.markdown-body :deep(pre code) {
  background: transparent;
  padding: 0;
  font-size: inherit;
}

.markdown-body :deep(blockquote) {
  margin: 0 0 16px;
  padding: 2px 14px;
  border-left: 3px solid var(--border-color);
  color: var(--secondary-text);
}

.markdown-body :deep(table) {
  width: 100%;
  border-collapse: collapse;
  margin: 0 0 16px;
  font-size: 13px;
}

.markdown-body :deep(th),
.markdown-body :deep(td) {
  border: 1px solid var(--border-color);
  padding: 6px 9px;
  text-align: left;
}

.markdown-body :deep(th) {
  background: var(--hover-bg);
  color: var(--title-color);
}

.markdown-body :deep(tr:nth-child(2n)) {
  background: color-mix(in srgb, var(--hover-bg) 60%, transparent);
}

.markdown-body :deep(hr) {
  border: 0;
  border-top: 1px solid var(--border-color);
  margin: 24px 0;
}

.markdown-body :deep(.mermaid) {
  margin: 18px auto;
  padding: 12px;
  border: 1px solid var(--border-color);
  border-radius: 10px;
  background: var(--card-bg);
  overflow-x: auto;
  cursor: zoom-in;
}

.markdown-body :deep(.mermaid svg) {
  display: block;
  margin: 0 auto;
  max-width: 100%;
  height: auto;
}

.zoom-modal {
  position: fixed;
  inset: 0;
  z-index: 3000;
  background: rgba(0, 0, 0, 0.78);
  display: grid;
  place-items: center;
  padding: 40px;
}

.zoom-content {
  max-width: 92vw;
  max-height: 88vh;
  overflow: auto;
  background: var(--card-bg);
  border-radius: 12px;
  padding: 24px;
}

.zoom-content :deep(svg) {
  max-width: none;
  height: auto;
}

.zoom-close {
  position: fixed;
  top: 18px;
  right: 22px;
  width: 40px;
  height: 40px;
  border: 0;
  border-radius: 50%;
  background: rgba(255, 255, 255, 0.16);
  color: #fff;
  font-size: 26px;
  cursor: pointer;
}
</style>
