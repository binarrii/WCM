<script setup>
import { ChevronLeft, ChevronRight } from '@lucide/vue';
import { computed, ref, watch } from 'vue';
import { paginationItems, parsePageTarget } from '../services/pagination';

const props = defineProps({
  page: { type: Number, required: true },
  pageCount: { type: Number, required: true },
  total: { type: Number, default: 0 },
  pageSize: { type: Number, default: 30 },
  disabled: { type: Boolean, default: false }
});
const emit = defineEmits(['change', 'update:pageSize']);
const pageSizes = [10, 20, 30, 50, 100];
const currentPage = computed(() => Math.min(props.pageCount, Math.max(1, props.page)));
const items = computed(() => paginationItems(currentPage.value, props.pageCount));
const jumpPage = ref(String(currentPage.value));
const jumpTarget = computed(() => parsePageTarget(jumpPage.value, props.pageCount));

watch([currentPage, () => props.pageCount], () => { jumpPage.value = String(currentPage.value); });

const changePage = target => {
  if (!props.disabled && parsePageTarget(target, props.pageCount) !== null && target !== currentPage.value) {
    emit('change', target);
  }
};
const jump = () => {
  if (jumpTarget.value !== null) changePage(jumpTarget.value);
};
const changePageSize = event => {
  const size = Number(event.target.value);
  if (!props.disabled && pageSizes.includes(size) && size !== props.pageSize) emit('update:pageSize', size);
};
</script>

<template>
  <nav class="pagination-bar" aria-label="分页导航" :aria-busy="disabled">
    <div class="pagination-info">
      <p class="pagination-summary" role="status" aria-atomic="true">
        共 <strong>{{ total }}</strong> 条<span class="pagination-divider">/</span>共 <strong>{{ pageCount }}</strong> 页
      </p>
      <select class="pagination-size" aria-label="每页显示数量" :value="pageSize" :disabled="disabled" @change="changePageSize">
        <option v-for="size in pageSizes" :key="size" :value="size">{{ size }} 条 / 页</option>
      </select>
    </div>

    <div class="pagination-controls">
      <div class="pagination-pages">
        <button class="pagination-button pagination-arrow" type="button" aria-label="上一页" title="上一页"
          :disabled="currentPage <= 1 || disabled" @click="changePage(currentPage - 1)">
          <ChevronLeft aria-hidden="true" />
        </button>
        <div class="pagination-page-list">
          <template v-for="item in items" :key="item">
            <span v-if="typeof item === 'string'" class="pagination-ellipsis" aria-hidden="true">…</span>
            <button v-else class="pagination-button pagination-number" type="button"
              :class="{ 'is-current': item === currentPage }" :aria-label="`第 ${item} 页`"
              :aria-current="item === currentPage ? 'page' : undefined" :disabled="disabled"
              @click="changePage(item)">{{ item }}</button>
          </template>
        </div>
        <button class="pagination-button pagination-arrow" type="button" aria-label="下一页" title="下一页"
          :disabled="currentPage >= pageCount || disabled" @click="changePage(currentPage + 1)">
          <ChevronRight aria-hidden="true" />
        </button>
      </div>

      <form v-if="pageCount > 1" class="pagination-jump" @submit.prevent="jump">
        <label>
          <span>前往</span>
          <input v-model="jumpPage" type="number" inputmode="numeric" min="1" :max="pageCount" step="1"
            aria-label="跳转页码" :title="`输入 1 至 ${pageCount} 的页码，按 Enter 跳转`"
            :disabled="disabled" @keydown.esc="jumpPage = String(currentPage)" />
          <span>页</span>
        </label>
        <button class="pagination-button pagination-go" type="submit"
          :disabled="disabled || jumpTarget === null || jumpTarget === currentPage">跳转</button>
      </form>
    </div>
  </nav>
</template>

<style scoped>
.pagination-bar { flex: 0 0 auto; display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 12px 20px; padding: 11px 16px; border-top: 1px solid var(--border-color); container-type: inline-size; font-size: .72rem; font-variant-numeric: tabular-nums; }
.pagination-info, .pagination-controls, .pagination-pages, .pagination-page-list, .pagination-jump, .pagination-jump label { display: flex; align-items: center; }
.pagination-info { flex-wrap: wrap; gap: 8px 16px; }
.pagination-summary { margin: 0; color: var(--text-secondary); white-space: nowrap; }
.pagination-summary strong { color: var(--text-primary); font-weight: 600; }
.pagination-divider { margin-inline: 9px; color: var(--text-muted); }
.pagination-controls { flex-wrap: wrap; gap: 12px 20px; margin-left: auto; }
.pagination-pages, .pagination-page-list { gap: 5px; }
.pagination-button, .pagination-jump input, .pagination-size { height: 32px; border: 1px solid var(--border-color); border-radius: 7px; background: var(--bg-card); color: var(--text-secondary); font: inherit; font-variant-numeric: tabular-nums; transition: background-color .15s, border-color .15s, color .15s; }
.pagination-button { flex-shrink: 0; display: inline-flex; align-items: center; justify-content: center; min-width: 32px; padding: 0 8px; cursor: pointer; }
.pagination-arrow { padding: 0; }
.pagination-button svg { width: 14px; height: 14px; }
.pagination-number { border-color: transparent; background: transparent; }
.pagination-button:hover:not(:disabled):not(.is-current), .pagination-size:hover:not(:disabled) { color: var(--color-primary); border-color: var(--border-hover); background: var(--color-primary-glow); }
.pagination-button.is-current { color: #fff; border-color: var(--color-primary); background: var(--color-primary); font-weight: 700; }
.pagination-button:focus-visible, .pagination-jump input:focus-visible, .pagination-size:focus-visible { outline: 2px solid var(--color-primary); outline-offset: 2px; }
.pagination-button:disabled, .pagination-jump input:disabled, .pagination-size:disabled { opacity: .4; cursor: default; }
.pagination-ellipsis { display: grid; place-items: center; width: 24px; height: 32px; color: var(--text-secondary); }
.pagination-size { padding-inline: 9px; cursor: pointer; }
.pagination-size option { color: var(--text-primary); background: var(--bg-card); }
.pagination-jump { gap: 8px; color: var(--text-secondary); white-space: nowrap; }
.pagination-jump label { gap: 7px; }
.pagination-jump input { width: 52px; padding: 0 5px; text-align: center; appearance: textfield; -moz-appearance: textfield; }
.pagination-jump input::-webkit-inner-spin-button, .pagination-jump input::-webkit-outer-spin-button { margin: 0; -webkit-appearance: none; }
.pagination-go { padding-inline: 10px; }
@container (max-width: 620px) {
  .pagination-info { width: 100%; justify-content: space-between; }
  .pagination-controls { width: 100%; margin-left: 0; justify-content: space-between; gap: 8px 12px; }
  .pagination-pages, .pagination-page-list { gap: 3px; }
  .pagination-button { min-width: 28px; padding-inline: 6px; }
  .pagination-ellipsis { width: 18px; }
  .pagination-jump { margin-left: auto; }
}
@media (prefers-reduced-motion: reduce) {
  .pagination-button, .pagination-jump input, .pagination-size { transition: none; }
}
</style>
