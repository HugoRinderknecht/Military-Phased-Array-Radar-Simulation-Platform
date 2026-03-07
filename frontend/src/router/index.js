import { createRouter, createWebHistory } from 'vue-router'
import { useUserStore } from '@/stores/user'

const routes = [
  {
    path: '/login',
    name: 'Login',
    component: () => import('@/views/Login.vue'),
    meta: { requiresAuth: false }
  },
  {
    path: '/',
    name: 'Main',
    component: () => import('@/views/Main.vue'),
    meta: { requiresAuth: true },
    redirect: '/dashboard',
    children: [
      {
        path: 'dashboard',
        name: 'Dashboard',
        component: () => import('@/views/Dashboard.vue')
      },
      {
        path: 'radar',
        name: 'RadarModels',
        component: () => import('@/views/RadarModels.vue')
      },
      {
        path: 'scene',
        name: 'Scenes',
        component: () => import('@/views/Scenes.vue')
      },
      {
        path: 'simulation',
        name: 'Simulation',
        component: () => import('@/views/Simulation.vue')
      },
      {
        path: 'results',
        name: 'Results',
        component: () => import('@/views/Results.vue')
      }
    ]
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

// 路由守卫
router.beforeEach((to, from, next) => {
  const userStore = useUserStore()

  if (to.meta.requiresAuth && !userStore.isLoggedIn) {
    next('/login')
  } else if (to.path === '/login' && userStore.isLoggedIn) {
    next('/')
  } else {
    next()
  }
})

export default router
