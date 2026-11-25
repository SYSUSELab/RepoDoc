<script setup lang="ts">
import { useRoute } from 'vue-router'
import ThemeToggle from './ThemeToggle.vue'

const route = useRoute()
</script>

<template>
  <div class="top-controls">
    <router-link v-if="route.name !== 'Home'" class="nav-link" to="/">New</router-link>
    <router-link v-if="route.name !== 'History'" class="nav-link" to="/history">
      History
    </router-link>
    <ThemeToggle />
  </div>
</template>

<style scoped>
.top-controls {
  position: fixed;
  top: 18px;
  right: 22px;
  z-index: 1000;
  display: flex;
  align-items: center;
  gap: 6px;
}

.nav-link {
  color: var(--secondary-text);
  font-size: 12px;
  font-weight: 500;
  padding: 8px 10px;
  border-radius: 8px;
  transition: all 0.2s ease;
}

.nav-link:hover {
  color: var(--text-color);
  background: var(--hover-bg);
  text-decoration: none;
}
</style>
