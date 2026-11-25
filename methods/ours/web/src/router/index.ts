import { createRouter, createWebHashHistory } from 'vue-router'
import Ask from '../pages/Ask.vue'
import Generation from '../pages/Generation.vue'
import History from '../pages/History.vue'
import Home from '../pages/Home.vue'
import Project from '../pages/Project.vue'

export default createRouter({
  history: createWebHashHistory(),
  routes: [
    { path: '/', name: 'Home', component: Home },
    { path: '/history', name: 'History', component: History },
    { path: '/run/:id', name: 'Run', component: Generation },
    { path: '/project/:id', name: 'Project', component: Project },
    { path: '/project/:id/ask', name: 'Ask', component: Ask },
  ],
})
