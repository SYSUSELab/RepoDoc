<script setup lang="ts">
import { onMounted, provide, ref } from 'vue'

const isDarkMode = ref(false)

const applyTheme = (dark: boolean) => {
  isDarkMode.value = dark
  document.body.classList.toggle('dark', dark)
  localStorage.setItem('repodoc-theme', dark ? 'dark' : 'light')
}

const toggleTheme = () => applyTheme(!isDarkMode.value)

onMounted(() => {
  applyTheme(localStorage.getItem('repodoc-theme') === 'dark')
})

provide('theme', { isDarkMode, toggleTheme })
</script>

<template>
  <router-view />
</template>
