<script setup lang="ts">
import { inject, type Ref } from 'vue'

const theme = inject<{ isDarkMode: Ref<boolean>; toggleTheme: () => void }>('theme')
</script>

<template>
  <button
    class="icon-btn"
    :title="theme?.isDarkMode.value ? 'Switch to light mode' : 'Switch to dark mode'"
    :aria-label="theme?.isDarkMode.value ? 'Switch to light mode' : 'Switch to dark mode'"
    @click="theme?.toggleTheme()"
  >
    <span v-if="theme?.isDarkMode.value">☀</span>
    <span v-else>☾</span>
  </button>
</template>

<style scoped>
.icon-btn {
  width: 40px;
  height: 40px;
  border: 0;
  border-radius: 50%;
  background: transparent;
  color: var(--text-color);
  cursor: pointer;
  font-size: 18px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s ease;
}

.icon-btn:hover {
  background: var(--hover-bg);
  transform: scale(1.05);
}
</style>
