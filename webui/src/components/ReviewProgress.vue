<script setup>
import { computed, ref } from 'vue';
import { taskProgress } from '../services/reviewProgress';
const props = defineProps({ task: { type: Object, required: true }, collapsible: { type: Boolean, default: true } });
const state = computed(() => taskProgress(props.task));
const expanded = ref(false);
</script>

<template>
  <div :class="['review-task-progress', task.status]">
    <div class="progress-heading">
      <span>{{ state.label }}</span>
      <span v-if="collapsible && state.details" class="progress-summary">{{ state.details }}</span>
      <strong>{{ state.percent == null ? '—' : `${state.percent.toFixed(1)}%` }}</strong>
      <button v-if="collapsible && state.windows.length" type="button" class="progress-toggle" :aria-expanded="expanded" :aria-controls="`review-progress-${task.id}`" @click.stop="expanded = !expanded">{{ expanded ? '收起详情 ▴' : '展开详情 ▾' }}</button>
    </div>
    <div class="progress-track" role="progressbar" aria-label="审核任务处理进度" :aria-valuenow="state.percent ?? undefined" aria-valuemin="0" aria-valuemax="100" :aria-valuetext="`${state.label} ${state.percent ?? ''} ${state.details}`" title="审核中按已处理视频位置估算；处理结束不代表内容安全" :class="{ indeterminate: state.percent == null && state.active }">
      <span :style="{ width: `${state.percent ?? 0}%` }"></span>
    </div>
    <div v-if="state.subProgress" class="sub-progress">
      <div class="sub-progress-heading">
        <span>{{ state.subProgress.label }}</span>
        <small>{{ state.subProgress.details }}</small>
        <strong>{{ state.subProgress.percent.toFixed(1) }}%</strong>
      </div>
      <div class="progress-track sub-progress-track" role="progressbar" :aria-label="`${state.subProgress.label}进度`" :aria-valuenow="state.subProgress.percent" aria-valuemin="0" aria-valuemax="100" :aria-valuetext="`${state.subProgress.label} ${state.subProgress.details}`">
        <span :style="{ width: `${state.subProgress.percent}%` }"></span>
      </div>
    </div>
    <small v-if="!collapsible && state.details" :title="state.details">{{ state.details }}</small>
    <div v-if="state.windows.length && (!collapsible || expanded)" :id="`review-progress-${task.id}`" class="active-windows">
    <div v-for="window in state.windows" :key="window.title" class="active-window">
      <strong>{{ window.title }}</strong>
      <small>{{ window.samples }}</small>
      <small v-if="window.stages" class="running-stage">{{ state.active ? '正在处理' : '停止位置' }} · {{ window.stages }}</small>
    </div>
    </div>
  </div>
</template>

<style scoped>
.review-task-progress { --progress-color: var(--primary, #6366f1); min-width: 150px; color: var(--text-secondary, #64748b); }
.progress-heading { display: flex; align-items: center; flex-wrap: wrap; gap: 6px 12px; font-size: 12px; margin-bottom: 7px; }
.progress-heading strong { margin-left: auto; color: var(--progress-color); font-variant-numeric: tabular-nums; }
.progress-summary { font-size: 11px; }
.progress-toggle { padding: 2px 4px; border: 0; border-radius: 4px; background: transparent; color: var(--progress-color); font: inherit; font-size: 11px; cursor: pointer; }
.progress-toggle:hover { background: var(--color-primary-glow, #6366f11a); }
.progress-toggle:focus-visible { outline: 2px solid var(--progress-color); outline-offset: 2px; }
.progress-track { height: 6px; overflow: hidden; border-radius: 999px; background: var(--bg-secondary, #e2e8f0); }
.progress-track span { display: block; height: 100%; border-radius: inherit; background: var(--progress-color); transition: width .3s ease; }
.sub-progress { margin-top: 9px; padding-left: 12px; border-left: 2px solid color-mix(in srgb, var(--progress-color) 35%, transparent); }
.sub-progress-heading { display: flex; align-items: center; gap: 8px; margin-bottom: 5px; font-size: 11px; }
.sub-progress-heading small { display: inline; margin: 0; color: inherit; }
.sub-progress-heading strong { margin-left: auto; color: var(--progress-color); font-variant-numeric: tabular-nums; }
.sub-progress-track { height: 4px; opacity: .85; }
.indeterminate span { width: 35% !important; animation: review-progress-pulse 1.5s ease-in-out infinite; }
.completed { --progress-color: var(--status-green, #10b981); }
.partial { --progress-color: var(--status-yellow, #f59e0b); }
.failed { --progress-color: var(--status-red, #ef4444); }
small { display: block; margin-top: 7px; font-size: 11px; line-height: 1.5; }
.active-windows { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 320px), 1fr)); gap: 8px 24px; }
.active-window { margin-top: 9px; padding-top: 8px; border-top: 1px solid var(--border-color, #e2e8f0); font-size: 11px; line-height: 1.6; overflow-wrap: anywhere; }
.active-window small { margin-top: 3px; }
.running-stage { color: var(--progress-color); }
@keyframes review-progress-pulse { from { transform: translateX(-100%); } to { transform: translateX(390%); } }
@media (prefers-reduced-motion: reduce) { .indeterminate span { animation: none; } .progress-track span { transition: none; } }
</style>
