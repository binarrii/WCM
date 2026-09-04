<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue';
import { AlertCircle, ChevronLeft, ChevronRight, ExternalLink, ListVideo, RefreshCw, Search } from '@lucide/vue';
import { navigateToReviewTask } from '../services/navigation';
import { reviewTaskService } from '../services/reviewTaskService';

const query = ref('');
const status = ref('');
const page = ref(1);
const pageSize = 30;
const total = ref(0);
const tasks = ref([]);
const loading = ref(false);
const error = ref('');
let searchTimer;
let requestSequence = 0;

const pageCount = computed(() => Math.max(1, Math.ceil(total.value / pageSize)));
const statusLabel = value => ({ processing: '处理中', completed: '已完成', failed: '失败' }[value] || value);
const formatTime = value => value ? new Intl.DateTimeFormat('zh-CN', {
  year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false
}).format(new Date(value)) : '—';
const parameterSummary = task => {
  const parameters = task.parameters || {};
  const similarity = Number.isFinite(Number(parameters.threshold))
    ? Math.round((1 - Number(parameters.threshold)) * 100)
    : null;
  return [
    `${parameters.sample_interval ?? 1}s/帧`,
    `候选 ${parameters.top_k ?? 10}`,
    similarity == null ? null : `相似度 ${similarity}%`
  ].filter(Boolean).join(' · ');
};

const loadTasks = async () => {
  const sequence = ++requestSequence;
  loading.value = true;
  error.value = '';
  try {
    const payload = await reviewTaskService.list({
      query: query.value.trim(), status: status.value, page: page.value, pageSize
    });
    if (sequence !== requestSequence) return;
    tasks.value = payload.items;
    total.value = payload.total;
    if (page.value > pageCount.value) page.value = pageCount.value;
  } catch (reason) {
    if (sequence === requestSequence) {
      error.value = reason.response?.data?.detail || reason.message || '任务列表加载失败';
    }
  } finally {
    if (sequence === requestSequence) loading.value = false;
  }
};
const searchNow = () => { clearTimeout(searchTimer); page.value = 1; loadTasks(); };
const changePage = value => { page.value = value; loadTasks(); };

watch(query, () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(searchNow, 300);
});
watch(status, searchNow);
onMounted(loadTasks);
onBeforeUnmount(() => clearTimeout(searchTimer));
</script>

<template>
  <main class="review-tasks animate-fade-in">
    <section class="task-toolbar">
      <form class="task-search" @submit.prevent="searchNow">
        <Search />
        <input v-model="query" type="search" placeholder="检索任务 ID、视频地址或失败原因" aria-label="检索审核任务" />
      </form>
      <label class="task-status-filter"><span>任务状态</span><select v-model="status"><option value="">全部状态</option><option value="processing">处理中</option><option value="completed">已完成</option><option value="failed">失败</option></select></label>
      <button class="task-refresh" type="button" :disabled="loading" @click="loadTasks"><RefreshCw :class="{ spinner: loading }" />刷新</button>
    </section>

    <p v-if="error" class="task-error" role="alert"><AlertCircle />{{ error }}</p>

    <section class="task-table-card">
      <header><div><h2>任务记录</h2><span>共 {{ total }} 条</span></div><small>点击任意一行进入视频审核页并自动加载结果</small></header>
      <div class="task-table-scroll">
        <table>
          <thead><tr><th>状态</th><th>视频地址</th><th>提交参数</th><th>结果</th><th>提交时间</th><th aria-label="操作"></th></tr></thead>
          <tbody>
            <tr v-for="task in tasks" :key="task.id" tabindex="0" @click="navigateToReviewTask(task.id)" @keydown.enter="navigateToReviewTask(task.id)">
              <td><span :class="['task-status', task.status]">{{ statusLabel(task.status) }}</span></td>
              <td class="task-video"><strong :title="task.video_url">{{ task.video_url }}</strong><small>{{ task.id }}</small><em v-if="task.error" :title="task.error">{{ task.error }}</em></td>
              <td class="task-parameters">{{ parameterSummary(task) }}</td>
              <td>{{ task.result_count }} 条</td>
              <td class="task-date">{{ formatTime(task.created_at) }}</td>
              <td><ExternalLink class="open-task" /></td>
            </tr>
          </tbody>
        </table>
        <div v-if="loading && !tasks.length" class="task-empty"><RefreshCw class="spinner" /><p>正在加载任务…</p></div>
        <div v-else-if="!tasks.length" class="task-empty"><ListVideo /><p>没有找到审核任务</p><small>可调整检索条件，或先提交一个视频审核。</small></div>
      </div>
      <footer class="task-pagination"><span>第 {{ page }} / {{ pageCount }} 页</span><button type="button" :disabled="page <= 1 || loading" @click="changePage(page - 1)"><ChevronLeft />上一页</button><button type="button" :disabled="page >= pageCount || loading" @click="changePage(page + 1)">下一页<ChevronRight /></button></footer>
    </section>
  </main>
</template>

<style scoped src="./review-tasks.css"></style>
