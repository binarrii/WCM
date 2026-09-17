<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue';
import { ClipboardList, Database, FolderOpen, LogOut, Monitor, Moon, Settings, ShieldCheck, SlidersHorizontal, Sun, Users, Video } from '@lucide/vue';
import FaceDashboard from './views/FaceDashboard.vue';
import ParameterConfig from './views/ParameterConfig.vue';
import ReviewTasks from './views/ReviewTasks.vue';
import MediaLibrary from './views/MediaLibrary.vue';
import SystemManagement from './views/SystemManagement.vue';
import VideoReview from './views/VideoReview.vue';
import AuthPage from './views/AuthPage.vue';
import AccountSecurity from './views/AccountSecurity.vue';
import UserManagement from './views/UserManagement.vue';
import ReauthDialog from './components/ReauthDialog.vue';
import UserAvatar from './components/UserAvatar.vue';
import { auth, can, clearSession, logout, refreshSession, roleNames } from './services/auth';
import { navigateTo, routeFromHash } from './services/navigation';
import './app.css';

const currentRoute = ref(routeFromHash(window.location.hash));
const currentTheme = ref(localStorage.getItem('theme') || 'system');
const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');

const pages = {
  people: {
    permission: 'people.read',
    component: FaceDashboard,
    title: '人物库管理',
    description: '维护人物档案、注册图片并进行人脸检索'
  },
  video: {
    permission: 'review.read',
    component: VideoReview,
    title: '视频审核',
    description: '分析远程视频并按时间轴快速复核命中画面'
  },
  tasks: {
    permission: 'review.read',
    component: ReviewTasks,
    title: '审核任务',
    description: '检索、查看并继续复核历史视频审核任务'
  },
  media: {
    component: MediaLibrary,
    title: '媒资库',
    description: '你的媒体资源管理空间'
  },
  parameters: {
    permission: 'parameters.manage',
    component: ParameterConfig,
    title: '参数配置',
    description: '集中管理系统通用参数，保存后即时刷新运行时内存'
  },
  system: {
    permission: 'system.manage',
    component: SystemManagement,
    title: '系统管理',
    description: '查看服务同步状态，管理 InsightFace 副本同步'
  }
};
pages.account = { component: AccountSecurity, title: '账户安全', description: '管理登录密码、Passkey 和双重验证' };
pages.users = { component: UserManagement, title: '用户与权限', description: '管理用户账户、角色与操作权限', permission: 'users.manage' };
const routeAllowed = route => !pages[route]?.permission || can(pages[route].permission);
const page = computed(() => pages[currentRoute.value] || pages.account);
const enforceRoute = () => {
  if (auth.user && !routeAllowed(currentRoute.value)) navigateTo(Object.keys(pages).find(routeAllowed) || 'account');
};
let sessionTimer;
const refreshOnFocus = () => { if (auth.user) refreshSession(); };
const signOut = async () => { try { await logout(); } catch { auth.error = '退出失败，请重试'; } };

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
watch([() => auth.user, currentRoute], enforceRoute);
onMounted(() => {
  refreshSession();
  sessionTimer = window.setInterval(refreshOnFocus, 60000);
  window.addEventListener('focus', refreshOnFocus);
  window.addEventListener('wcm-session-expired', clearSession);
  if (!window.location.hash) navigateTo('people');
  window.addEventListener('hashchange', syncRoute);
  mediaQuery.addEventListener('change', handleSystemTheme);
});
onBeforeUnmount(() => {
  window.clearInterval(sessionTimer);
  window.removeEventListener('focus', refreshOnFocus);
  window.removeEventListener('wcm-session-expired', clearSession);
  window.removeEventListener('hashchange', syncRoute);
  mediaQuery.removeEventListener('change', handleSystemTheme);
});
</script>

<template>
  <main v-if="!auth.ready || (auth.error && !auth.user)" class="auth-loading"><p>{{ auth.error || '正在加载账户…' }}</p><button v-if="auth.error" class="auth-secondary" @click="refreshSession">重试</button></main>
  <AuthPage v-else-if="!auth.user" />
  <div v-else class="shell">
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
        <button v-if="can('people.read')" type="button" aria-label="人物库管理" :class="['menu-item', { active: currentRoute === 'people' }]" @click="navigateTo('people')">
          <Database /><span>人物库管理</span>
        </button>
        <button v-if="can('review.read')" type="button" aria-label="视频审核" :class="['menu-item', { active: currentRoute === 'video' }]" @click="navigateTo('video')">
          <Video /><span>视频审核</span>
        </button>
        <button v-if="can('review.read')" type="button" aria-label="审核任务" :class="['menu-item', { active: currentRoute === 'tasks' }]" @click="navigateTo('tasks')">
          <ClipboardList /><span>审核任务</span>
        </button>
        <button type="button" aria-label="媒资库" :class="['menu-item', { active: currentRoute === 'media' }]" @click="navigateTo('media')">
          <FolderOpen /><span>媒资库</span>
        </button>
        <button v-if="can('parameters.manage')" type="button" aria-label="参数配置" :class="['menu-item', { active: currentRoute === 'parameters' }]" @click="navigateTo('parameters')">
          <SlidersHorizontal /><span>参数配置</span>
        </button>
        <button v-if="can('system.manage')" type="button" aria-label="系统管理" :class="['menu-item', { active: currentRoute === 'system' }]" @click="navigateTo('system')">
          <Settings /><span>系统管理</span>
        </button>
        <button v-if="can('users.manage')" type="button" aria-label="用户与权限" :class="['menu-item', { active: currentRoute === 'users' }]" @click="navigateTo('users')"><Users /><span>用户与权限</span></button>
        <button type="button" aria-label="账户安全" :class="['menu-item', { active: currentRoute === 'account' }]" @click="navigateTo('account')"><ShieldCheck /><span>账户安全</span></button>
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
          <div class="shell-account"><button class="account-entry" :aria-label="`账户安全：${auth.user.display_name}`" @click="navigateTo('account')"><UserAvatar :user="auth.user" /><span class="account-copy"><strong>{{ auth.user.display_name }}</strong><small>{{ roleNames[auth.user.role] }}</small></span></button><button title="退出登录" aria-label="退出登录" @click="signOut"><LogOut /></button></div>
        </div>
      </header>

      <p v-if="auth.error" class="auth-error" role="alert">{{ auth.error }}</p>
      <component v-if="routeAllowed(currentRoute)" :is="page.component" :key="`${auth.user.id}:${currentRoute}`" />
    </section>
    <ReauthDialog />
  </div>
</template>
