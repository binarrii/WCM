<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue';
import { AlertCircle, CheckCircle2, ChevronLeft, ChevronRight, Download, ExternalLink, ListVideo, RefreshCw, Search, Trash2 } from '@lucide/vue';
import ConfirmDialog from '../components/ConfirmDialog.vue';
import { saveBlob } from '../services/downloads';
import { navigateToReviewTask } from '../services/navigation';
import { reviewTaskService } from '../services/reviewTaskService';

const query = ref('');
const status = ref('');
const page = ref(1);
const pageSize = 30;
const total = ref(0);
const tasks = ref([]);
const loading = ref(false);
const deleting = ref(false);
const downloadingIds = ref(new Set());
const batchDownloading = ref(false);
const error = ref('');
const notice = ref('');
const selectedIds = ref(new Set());
const pendingDelete = ref(null);
let searchTimer;
let requestSequence = 0;

const pageCount = computed(() => Math.max(1, Math.ceil(total.value / pageSize)));
const allSelected = computed(() => tasks.value.length > 0 && tasks.value.every(task => selectedIds.value.has(task.id)));
const someSelected = computed(() => !allSelected.value && tasks.value.some(task => selectedIds.value.has(task.id)));
const downloadableSelectedIds = computed(() => tasks.value
  .filter(task => selectedIds.value.has(task.id) && task.status === 'completed')
  .map(task => task.id));
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
    const visibleIds = new Set(payload.items.map(task => task.id));
    selectedIds.value = new Set([...selectedIds.value].filter(id => visibleIds.has(id)));
    if (page.value > pageCount.value) page.value = pageCount.value;
  } catch (reason) {
    if (sequence === requestSequence) {
      error.value = reason.response?.data?.detail || reason.message || '任务列表加载失败';
    }
  } finally {
    if (sequence === requestSequence) loading.value = false;
  }
};
const clearSelection = () => { selectedIds.value = new Set(); };
const searchNow = () => { clearTimeout(searchTimer); clearSelection(); page.value = 1; loadTasks(); };
const changePage = value => { clearSelection(); page.value = value; loadTasks(); };
const toggleTask = taskId => {
  const next = new Set(selectedIds.value);
  if (next.has(taskId)) next.delete(taskId);
  else next.add(taskId);
  selectedIds.value = next;
};
const toggleAll = event => {
  selectedIds.value = event.target.checked ? new Set(tasks.value.map(task => task.id)) : new Set();
};
const downloadTaskResults = async task => {
  if (task.status !== 'completed' || downloadingIds.value.has(task.id)) return;
  downloadingIds.value = new Set([...downloadingIds.value, task.id]);
  error.value = '';
  notice.value = '';
  try {
    const blob = await reviewTaskService.downloadResults(task.id);
    saveBlob(blob, `analysis-${task.id}.json`);
    notice.value = '分析结果已开始下载';
  } catch (reason) {
    error.value = reason.response?.data?.detail || reason.message || '分析结果下载失败';
  } finally {
    const next = new Set(downloadingIds.value);
    next.delete(task.id);
    downloadingIds.value = next;
  }
};
const downloadSelectedResults = async () => {
  const ids = downloadableSelectedIds.value;
  if (!ids.length || batchDownloading.value) return;
  batchDownloading.value = true;
  error.value = '';
  notice.value = '';
  try {
    const blob = await reviewTaskService.downloadManyResults(ids);
    saveBlob(blob, `analysis-results-${ids.length}.zip`);
    notice.value = `已开始下载 ${ids.length} 条任务的分析结果`;
  } catch (reason) {
    error.value = reason.response?.data?.detail || reason.message || '批量下载失败';
  } finally {
    batchDownloading.value = false;
  }
};
const requestDeleteTask = task => {
  pendingDelete.value = { mode: 'single', task, ids: [task.id] };
};
const requestDeleteSelected = () => {
  const ids = [...selectedIds.value];
  if (ids.length) pendingDelete.value = { mode: 'batch', ids };
};
const cancelDelete = () => {
  if (!deleting.value) pendingDelete.value = null;
};
const confirmDelete = async () => {
  const request = pendingDelete.value;
  if (!request) return;
  deleting.value = true;
  error.value = '';
  notice.value = '';
  try {
    if (request.mode === 'single') {
      await reviewTaskService.deleteOne(request.task.id);
      selectedIds.value.delete(request.task.id);
      selectedIds.value = new Set(selectedIds.value);
      if (tasks.value.length === 1 && page.value > 1) page.value -= 1;
      notice.value = '已删除 1 条审核任务';
    } else {
      const result = await reviewTaskService.deleteMany(request.ids);
      if (request.ids.length >= tasks.value.length && page.value > 1) page.value -= 1;
      clearSelection();
      notice.value = `已删除 ${result.deleted} 条审核任务`;
    }
    await loadTasks();
  } catch (reason) {
    error.value = reason.response?.data?.detail || reason.message || (request.mode === 'single' ? '任务删除失败' : '批量删除失败');
  } finally {
    deleting.value = false;
    pendingDelete.value = null;
  }
};

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
    <p v-if="notice" class="task-notice" role="status"><CheckCircle2 />{{ notice }}</p>

    <section class="task-table-card">
      <header><div class="task-table-title"><h2>任务记录</h2><span>共 {{ total }} 条</span></div><div class="task-table-header-actions"><template v-if="selectedIds.size"><span>已选择 {{ selectedIds.size }} 条</span><button class="download-selected" type="button" :disabled="deleting || batchDownloading || !downloadableSelectedIds.length" :title="downloadableSelectedIds.length ? `下载 ${downloadableSelectedIds.length} 条已完成任务的结果` : '所选任务暂无可下载结果'" @click="downloadSelectedResults"><Download />{{ batchDownloading ? '打包中…' : `批量下载 (${downloadableSelectedIds.length})` }}</button><button type="button" :disabled="deleting || batchDownloading" @click="requestDeleteSelected"><Trash2 />批量删除</button></template><small v-else>点击任意一行进入视频审核页并自动加载结果</small></div></header>
      <div class="task-table-scroll">
        <table>
          <thead><tr><th class="task-select-cell"><input type="checkbox" :checked="allSelected" :indeterminate="someSelected" :disabled="!tasks.length || deleting" aria-label="选择当前页全部任务" @change="toggleAll" /></th><th>状态</th><th>视频地址</th><th>提交参数</th><th>结果</th><th>提交时间</th><th aria-label="操作"></th></tr></thead>
          <tbody>
            <tr v-for="task in tasks" :key="task.id" :class="{ selected: selectedIds.has(task.id) }" tabindex="0" @click="navigateToReviewTask(task.id)" @keydown.enter="navigateToReviewTask(task.id)">
              <td class="task-select-cell" @click.stop @keydown.enter.stop><input type="checkbox" :checked="selectedIds.has(task.id)" :disabled="deleting" :aria-label="`选择任务 ${task.id}`" @change="toggleTask(task.id)" /></td>
              <td><span :class="['task-status', task.status]">{{ statusLabel(task.status) }}</span></td>
              <td class="task-video"><strong :title="task.video_url">{{ task.video_url }}</strong><small>{{ task.id }}</small><em v-if="task.error" :title="task.error">{{ task.error }}</em></td>
              <td class="task-parameters">{{ parameterSummary(task) }}</td>
              <td>{{ task.result_count }} 条</td>
              <td class="task-date">{{ formatTime(task.created_at) }}</td>
              <td class="task-actions" @click.stop @keydown.enter.stop><button type="button" title="打开任务" :disabled="deleting" @click="navigateToReviewTask(task.id)"><ExternalLink /></button><button type="button" :title="task.status === 'completed' ? '下载分析结果' : '分析结果尚未就绪'" :disabled="deleting || task.status !== 'completed' || downloadingIds.has(task.id)" @click="downloadTaskResults(task)"><Download /></button><button class="delete-task" type="button" title="删除任务" :disabled="deleting || batchDownloading" @click="requestDeleteTask(task)"><Trash2 /></button></td>
            </tr>
          </tbody>
        </table>
        <div v-if="loading && !tasks.length" class="task-empty"><RefreshCw class="spinner" /><p>正在加载任务…</p></div>
        <div v-else-if="!tasks.length" class="task-empty"><ListVideo /><p>没有找到审核任务</p><small>可调整检索条件，或先提交一个视频审核。</small></div>
      </div>
      <footer class="task-pagination"><span>第 {{ page }} / {{ pageCount }} 页</span><button type="button" :disabled="page <= 1 || loading" @click="changePage(page - 1)"><ChevronLeft />上一页</button><button type="button" :disabled="page >= pageCount || loading" @click="changePage(page + 1)">下一页<ChevronRight /></button></footer>
    </section>

    <ConfirmDialog
      :open="Boolean(pendingDelete)"
      :title="pendingDelete?.mode === 'batch' ? '批量删除审核任务' : '删除审核任务'"
      :message="pendingDelete?.mode === 'batch' ? `将永久删除选中的 ${pendingDelete.ids.length} 条审核任务及其分析结果。` : '该任务及其分析结果将被永久删除，此操作无法撤销。'"
      :confirm-label="pendingDelete?.mode === 'batch' ? `删除 ${pendingDelete.ids.length} 条` : '确认删除'"
      :busy="deleting"
      @cancel="cancelDelete"
      @confirm="confirmDelete"
    >
      <template v-if="pendingDelete?.mode === 'single'" #details>
        <strong>{{ pendingDelete.task.video_url }}</strong>
        <span>{{ pendingDelete.task.id }}</span>
      </template>
    </ConfirmDialog>
  </main>
</template>

<style scoped src="./review-tasks.css"></style>
