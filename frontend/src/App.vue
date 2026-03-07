<template>
  <router-view />
</template>

<script setup>
import { onMounted } from 'vue'
import { useUserStore } from '@/stores/user'

const userStore = useUserStore()

onMounted(() => {
  // 检查本地存储的token
  const token = localStorage.getItem('token')
  if (token) {
    userStore.token = token
  }
})
</script>

<style>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

html, body, #app {
  width: 100%;
  height: 100%;
  font-family: 'Microsoft YaHei', Arial, sans-serif;
}

.page-container {
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
}
</style>
