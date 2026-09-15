<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue';
import { AlertCircle, ArrowRight, CheckCircle2, Clock3, RefreshCw, Server, ShieldCheck } from '@lucide/vue';
import { insightfaceService } from '../services/insightfaceService';
import { canTriggerSync, formatSyncTime, manualRequestLabel, replicaPresentation } from '../services/replicationStatus';
import './system-management.css';

const status = ref(null);
const loading = ref(false);
const submitting = ref(null);
const error = ref('');
const notice = ref('');
const updatedAt = ref(null);
const replicas = computed(() => status.value?.replicas || []);
const readable = computed(() => replicas.value.filter(node => node.read_eligible).length);
const eligible = computed(() => replicas.value.filter(node => canTriggerSync(status.value, node)));
const totalLag = computed(() => replicas.value.reduce((sum, node) => sum + Number(node.lag || 0), 0));
let timer;
let disposed = false;
let generation = 0;

const reasonText = (reason, fallback) => {
  const detail = reason.response?.data?.detail;
  return typeof detail === 'string' ? detail : fallback;
};
const loadStatus = async () => {
  if (loading.value || disposed) return;
  const current = ++generation;
  loading.value = true;
  try {
    const value = await insightfaceService.status();
    if (disposed || current !== generation) return;
    status.value = value;
    updatedAt.value = new Date().toISOString();
    error.value = '';
  } catch (reason) {
    if (!disposed && current === generation) error.value = reasonText(reason, '同步状态读取失败，当前数据可能已过期，请刷新重试。');
  } finally {
    if (current === generation) loading.value = false;
  }
};

const triggerSync = async (nodeId = null) => {
  if (submitting.value !== null || error.value || disposed) return;
  submitting.value = nodeId ?? '*';
  notice.value = '';
  try {
    const result = await insightfaceService.sync(nodeId);
    if (disposed) return;
    const names = result.requested.map(node => node.id).join('、');
    const skipped = result.skipped?.length ? `；${result.skipped.map(node => node.id).join('、')} 保留当前状态` : '';
    notice.value = `已提交 ${names} 的同步请求，由同步进程检查并追赶${skipped}。`;
  } catch (reason) {
    if (!disposed) notice.value = '';
    if (!disposed) error.value = reasonText(reason, '暂无法确认同步请求是否提交，请刷新查看手动请求状态；重复请求会合并。');
  } finally {
    if (!disposed) {
      submitting.value = null;
      // Invalidate a GET started before this POST, so it cannot restore stale state.
      generation += 1;
      loading.value = false;
      // Keep submission errors visible until the next explicit or periodic refresh.
      if (!error.value) await loadStatus();
    }
  }
};

const poll = async () => {
  if (disposed) return;
  if (!document.hidden && submitting.value === null) await loadStatus();
  if (!disposed) timer = window.setTimeout(poll, 5000);
};
onMounted(poll);
onBeforeUnmount(() => { disposed = true; generation += 1; window.clearTimeout(timer); });
</script>

<template>
  <main class="system-management animate-fade-in">
    <section class="sync-toolbar">
      <div><h2><Server />InsightFace 同步</h2><p>查看副本状态，检查并同步已提交的人物变更。</p></div>
      <div class="sync-toolbar-actions">
        <span class="sync-updated"><Clock3 />{{ updatedAt ? `更新于 ${formatSyncTime(updatedAt)}` : '正在获取状态' }}<small>每 5 秒刷新</small></span>
        <button type="button" class="sync-button" :disabled="loading || submitting !== null" @click="loadStatus"><RefreshCw :class="{ spinner: loading }" />刷新状态</button>
        <button type="button" class="sync-button primary" :disabled="!eligible.length || submitting !== null || Boolean(error)" @click="triggerSync()"><RefreshCw :class="{ spinner: submitting === '*' }" />{{ submitting === '*' ? '提交中…' : '同步全部可用副本' }}</button>
      </div>
    </section>

    <p v-if="error" class="sync-message danger" role="alert"><AlertCircle />{{ error }}</p>
    <p v-if="notice" class="sync-message success" role="status"><CheckCircle2 />{{ notice }}</p>
    <section v-if="!status" class="sync-empty"><RefreshCw :class="{ spinner: loading }" /><h3>{{ loading ? '正在读取同步状态' : '暂时无法读取同步状态' }}</h3><p>请检查服务连接，或点击“刷新状态”重试。</p></section>
    <section v-else-if="!status.enabled" class="sync-empty"><Server /><h3>尚未启用 InsightFace 副本同步</h3><p>启用并初始化副本后，可在这里查看状态和手动触发同步。</p></section>
    <template v-else>
      <p v-if="!status.initialized" class="sync-message warning" role="status"><AlertCircle />同步基线尚未初始化，请先完成副本初始化。</p>
      <p v-if="status.primary_recovering" class="sync-message danger" role="alert"><AlertCircle />主节点正在恢复，手动同步暂不可用。请完成恢复及核验后再操作。</p>
      <p v-else-if="status.pending_primary_operations" class="sync-message warning" role="status"><AlertCircle />主节点有 {{ status.pending_primary_operations }} 项未完成的人物操作。副本仅同步已提交的变更；操作长时间未结束时，请检查恢复记录。</p>

      <section class="sync-metrics" aria-label="同步概览">
        <article><span>已提交序号</span><strong>{{ status.committed_sequence }}</strong><small>主节点已持久化的变更</small></article>
        <article><span>可读副本</span><strong>{{ readable }}<em>/ {{ replicas.length }}</em></strong><small>已追平且心跳有效</small></article>
        <article><span>副本落后总数</span><strong>{{ totalLag }}</strong><small>各副本待追赶序号差之和</small></article>
        <article><span>人物操作</span><strong>{{ status.pending_primary_operations }}</strong><small>处理中或等待恢复</small></article>
      </section>

      <section class="sync-policy"><ShieldCheck /><div><strong>自动同步持续运行，手动同步可提前重试</strong><p>请求提交后请查看各副本的执行结果。已隔离或未初始化的副本需要先完成恢复或初始化。</p></div></section>

      <section class="sync-node-grid" aria-label="InsightFace 副本列表">
        <article v-for="node in replicas" :key="node.id" class="sync-node">
          <header><div class="sync-node-identity"><span class="sync-node-icon"><Server /></span><div><h3>副本 {{ node.id }}</h3><code>{{ node.url }}</code></div></div><span :class="['sync-badge', replicaPresentation(node).tone]">{{ replicaPresentation(node).label }}</span></header>
          <div class="sync-version"><span><small>当前序号</small><strong>{{ node.applied_seq }}</strong></span><ArrowRight /><span><small>已提交序号</small><strong>{{ status.committed_sequence }}</strong></span><span class="sync-lag"><small>落后</small><strong>{{ node.lag }}</strong></span></div>
          <dl class="sync-node-details">
            <div><dt>最近心跳</dt><dd>{{ formatSyncTime(node.heartbeat) }}</dd></div>
            <div><dt>连续重试次数</dt><dd>{{ node.attempts }}</dd></div>
            <div><dt>下次重试</dt><dd>{{ formatSyncTime(node.next_retry) }}</dd></div>
            <div><dt>状态更新时间</dt><dd>{{ formatSyncTime(node.updated_at) }}</dd></div>
          </dl>
          <div v-if="node.last_error" class="sync-node-error"><strong>最近错误 / 隔离原因</strong><code>{{ node.last_error }}</code></div>
          <p v-if="node.state === 'quarantined'" class="sync-node-guidance">该副本已隔离。请先终止旧同步执行者和在途请求，再按恢复流程处理。</p>
          <p v-else-if="node.state === 'new'" class="sync-node-guidance">先从一致性备份初始化并完成核验，才能开始同步。</p>
          <p v-else-if="!node.heartbeat_fresh" class="sync-node-guidance">当前没有有效心跳，请检查副本及同步进程是否运行。</p>
          <div class="sync-manual-status"><span><Clock3 />最近手动请求</span><strong>{{ manualRequestLabel(node) }}</strong><small v-if="node.manual_request">目标序号 {{ node.manual_request.target_sequence }} · 提交 {{ formatSyncTime(node.manual_request.requested_at) }}<template v-if="node.manual_request.completed_at"> · 完成 {{ formatSyncTime(node.manual_request.completed_at) }}</template></small></div>
          <footer><span>{{ node.read_eligible ? '允许人脸搜索读取' : '暂不接收新的搜索读取' }}</span><button type="button" class="sync-button" :aria-label="`同步副本 ${node.id}`" :disabled="!canTriggerSync(status, node) || submitting !== null || Boolean(error)" @click="triggerSync(node.id)"><RefreshCw :class="{ spinner: submitting === node.id }" />{{ submitting === node.id ? '提交中…' : '立即同步' }}</button></footer>
        </article>
        <div v-if="!replicas.length" class="sync-empty"><Server /><h3>尚未配置副本</h3><p>配置副本并完成基线核验后，状态会显示在这里。</p></div>
      </section>
    </template>
  </main>
</template>
