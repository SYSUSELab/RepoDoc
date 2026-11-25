<script setup lang="ts">
import { onMounted, onUnmounted, ref } from 'vue'

const lines = [
  'Turn a repository into architecture documentation.',
  'Graph first. Components next. Pages that explain why.',
]
const text = ref('')
const finished = ref(false)
let lineIndex = 0
let charIndex = 0
let deleting = false
let timer: ReturnType<typeof setTimeout> | undefined

const tick = () => {
  const current = lines[lineIndex]!
  if (deleting) {
    text.value = current.slice(0, charIndex - 1)
    charIndex -= 1
  } else {
    text.value = current.slice(0, charIndex + 1)
    charIndex += 1
  }
  let delay = deleting ? 18 : 38 + Math.random() * 22
  if (!deleting && text.value === current) {
    if (lineIndex === lines.length - 1) {
      finished.value = true
      return
    }
    deleting = true
    delay = 1600
  } else if (deleting && text.value === '') {
    deleting = false
    lineIndex += 1
    delay = 300
  }
  timer = setTimeout(tick, delay)
}

onMounted(() => {
  timer = setTimeout(tick, 220)
})
onUnmounted(() => {
  if (timer) clearTimeout(timer)
})
</script>

<template>
  <header class="app-header">
    <h1>RepoDoc</h1>
    <p class="subtitle">
      <span>{{ text }}</span><span class="cursor" :class="{ hidden: finished }">|</span>
    </p>
  </header>
</template>

<style scoped>
.app-header {
  text-align: center;
  margin-bottom: 34px;
  animation: slideUpFade 0.7s ease-out both;
}

h1 {
  font-size: 38px;
  line-height: 1.15;
  letter-spacing: -0.8px;
  color: var(--title-color);
  margin-bottom: 10px;
}

.subtitle {
  min-height: 1.5em;
  font-size: 14px;
  color: var(--subtitle-color);
}

.cursor {
  display: inline-block;
  margin-left: 2px;
  animation: blink 1s step-end infinite;
}

.cursor.hidden {
  opacity: 0;
  animation: none;
}
</style>
