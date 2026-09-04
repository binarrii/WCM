<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue';
import { Database, Monitor, Moon, Sun, Video } from '@lucide/vue';
import FaceDashboard from './views/FaceDashboard.vue';
import VideoReview from './views/VideoReview.vue';
import { navigateTo, routeFromHash } from './services/navigation';
import './app.css';

const currentRoute = ref(routeFromHash(window.location.hash));
const currentTheme = ref(localStorage.getItem('theme') || 'system');
const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');

const pages = {
  people: {
    component: FaceDashboard,
    title: '人物库管理',
    description: '维护人物档案、注册图片并进行人脸检索'
  },
  video: {
    component: VideoReview,
    title: '视频审核标记',
    description: '分析远程视频并按时间轴快速复核命中画面'
  }
};
const page = computed(() => pages[currentRoute.value]);

const applyTheme = () => {
  const resolved = currentTheme.value === 'system'
    ? (mediaQuery.matches ? 'dark' : 'light')
    : currentTheme.value;
  document.documentElement.setAttribute('data-theme', resolved);
};
const setTheme = (theme) => {
  currentTheme.value = theme;
  localStorage.setItem('theme', theme);
};
const syncRoute = () => { currentRoute.value = routeFromHash(window.location.hash); };
const handleSystemTheme = () => { if (currentTheme.value === 'system') applyTheme(); };

watch(currentTheme, applyTheme, { immediate: true });
onMounted(() => {
  if (!window.location.hash) navigateTo('people');
  window.addEventListener('hashchange', syncRoute);
  mediaQuery.addEventListener('change', handleSystemTheme);
});
onBeforeUnmount(() => {
  window.removeEventListener('hashchange', syncRoute);
  mediaQuery.removeEventListener('change', handleSystemTheme);
});
</script>

<template>
  <div class="shell">
    <aside class="sidebar">
      <div class="sidebar-brand">
        <span class="brand-orb"></span>
        <span class="brand-copy">
          <strong>WCM Core</strong>
          <small>智能内容审核库</small>
        </span>
      </div>

      <nav class="main-menu" aria-label="主菜单">
        <span class="menu-heading">主菜单</span>
        <button type="button" :class="['menu-item', { active: currentRoute === 'people' }]" @click="navigateTo('people')">
          <Database /><span>人物库管理</span>
        </button>
        <button type="button" :class="['menu-item', { active: currentRoute === 'video' }]" @click="navigateTo('video')">
          <Video /><span>视频审核标记</span>
        </button>
      </nav>
    </aside>

    <section class="workspace">
      <header class="workspace-header">
        <div><h1>{{ page.title }}</h1><p>{{ page.description }}</p></div>
        <div class="header-actions">
          <div class="shell-theme-switcher" aria-label="主题设置">
            <button v-for="theme in ['light', 'dark', 'system']" :key="theme" type="button" :class="{ active: currentTheme === theme }" :title="theme === 'light' ? '浅色模式' : theme === 'dark' ? '深色模式' : '跟随系统'" @click="setTheme(theme)">
              <Sun v-if="theme === 'light'" /><Moon v-else-if="theme === 'dark'" /><Monitor v-else />
            </button>
          </div>
          <div class="shell-status"><span></span>系统服务正常</div>
        </div>
      </header>

      <component :is="page.component" :key="currentRoute" />
    </section>
  </div>
</template>
