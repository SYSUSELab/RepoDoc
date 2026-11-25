<script setup lang="ts">
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { startRun } from '../api'
import AppHeader from '../components/AppHeader.vue'
import RunForm from '../components/RunForm.vue'
import TopControls from '../components/TopControls.vue'

const router = useRouter()
const starting = ref(false)
const error = ref('')

const submit = async (payload: { source: string }) => {
  starting.value = true
  error.value = ''
  try {
    const run = await startRun(payload)
    await router.push({ name: 'Run', params: { id: run.run_id } })
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  } finally {
    starting.value = false
  }
}
</script>

<template>
  <div class="home-page">
    <TopControls />
    <main class="home-main">
      <AppHeader />
      <RunForm :busy="starting" @submit="submit" />
      <p v-if="error" class="global-error">{{ error }}</p>
      <p class="footnote">GitHub repository in. Architecture documentation out.</p>
    </main>
  </div>
</template>

<style scoped>
.home-page {
  min-height: 100vh;
  background:
    radial-gradient(circle at 50% 28%, rgba(74, 144, 226, 0.06), transparent 34%),
    var(--container-bg);
  display: grid;
  place-items: center;
}

.home-main {
  width: 100%;
  max-width: 900px;
  padding: 24px;
  transform: translateY(-3vh);
}

.footnote {
  text-align: center;
  color: var(--subtitle-color);
  font-size: 11px;
  letter-spacing: 0.2px;
  margin-top: 22px;
  opacity: 0.75;
}

.global-error {
  max-width: 760px;
  margin: 12px auto 0;
  color: var(--danger);
  text-align: center;
  font-size: 12px;
}

@media (max-width: 640px) {
  .home-main {
    transform: none;
    padding: 86px 20px 40px;
  }
}
</style>
