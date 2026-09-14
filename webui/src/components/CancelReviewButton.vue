<script setup>
import { computed, ref } from 'vue';
import { LoaderCircle, Square } from '@lucide/vue';
import ConfirmDialog from './ConfirmDialog.vue';
import { reviewTaskService } from '../services/reviewTaskService';
import { reviewTaskActive } from '../services/reviewStatus';

const props = defineProps({
  task: { type: Object, default: null },
  compact: Boolean,
  disabled: Boolean
});
const emit = defineEmits(['updated', 'error', 'settled']);
const open = ref(false);
const busy = ref(false);
const stopping = computed(() => busy.value || props.task?.status === 'cancelling');
const cancelTask = async () => {
  const id = props.task?.id;
  if (!id || busy.value) return;
  busy.value = true;
  try {
    emit('updated', await reviewTaskService.cancel(id));
  } catch (reason) {
    emit('error', reason.response?.data?.detail || reason.message || '取消任务失败，请重试');
  } finally {
    busy.value = false;
    open.value = false;
    emit('settled', id);
  }
};
</script>

<template>
  <button v-if="reviewTaskActive(task)" type="button" :class="['cancel-review-button', { compact }]"
    :disabled="disabled || stopping" :title="stopping ? '取消中…' : '取消任务'"
    :aria-label="stopping ? '取消中' : '取消任务'" @click.stop="open = true">
    <LoaderCircle v-if="stopping" class="spinner" /><Square v-else />
    <span v-if="!compact">{{ stopping ? '取消中…' : '取消任务' }}</span>
  </button>
  <ConfirmDialog v-model:open="open" title="取消审核任务"
    message="将停止该任务的下载和审核，并保留任务记录。取消后如需继续，请重新开始分析。"
    confirm-label="确认取消任务" cancel-label="继续审核" :busy="busy" @confirm="cancelTask">
    <template #details><strong>{{ task?.id }}</strong></template>
  </ConfirmDialog>
</template>

<style scoped>
.cancel-review-button { box-sizing: border-box; height: 32px; display: inline-flex; align-items: center; justify-content: center; gap: 7px; padding: 0 12px; border: 1px solid var(--status-red-border); border-radius: 9px; background: var(--status-red-bg); color: var(--status-red); font: 600 .8rem var(--font-sans); white-space: nowrap; cursor: pointer; }
.cancel-review-button svg { width: 16px; height: 16px; }
.cancel-review-button:hover:not(:disabled) { filter: brightness(.95); }
.cancel-review-button:disabled { opacity: .6; cursor: default; }
.cancel-review-button:focus-visible { outline: 2px solid var(--status-red); outline-offset: 2px; }
.cancel-review-button.compact { width: 28px; height: 28px; padding: 0; margin-left: 4px; vertical-align: middle; border-color: transparent; background: transparent; }
</style>
