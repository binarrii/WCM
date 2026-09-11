<script setup>
import { ChevronLeft, ChevronRight } from '@lucide/vue';

defineProps({
  page: { type: Number, required: true },
  pageCount: { type: Number, required: true },
  disabled: { type: Boolean, default: false }
});

defineEmits(['change']);
</script>

<template>
  <footer class="pagination-bar" aria-label="分页导航">
    <span>第 {{ page }} / {{ pageCount }} 页</span>
    <button type="button" :disabled="page <= 1 || disabled" @click="$emit('change', page - 1)">
      <ChevronLeft />上一页
    </button>
    <button type="button" :disabled="page >= pageCount || disabled" @click="$emit('change', page + 1)">
      下一页<ChevronRight />
    </button>
  </footer>
</template>

<style scoped>
.pagination-bar { flex: 0 0 auto; display: flex; justify-content: flex-end; align-items: center; gap: 8px; padding: 11px 16px; border-top: 1px solid var(--border-color); }
.pagination-bar span { margin-right: 5px; color: var(--text-muted); font-size: .7rem; }
.pagination-bar button { display: flex; align-items: center; gap: 4px; padding: 6px 9px; color: var(--text-secondary); background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 7px; cursor: pointer; font-size: .72rem; }
.pagination-bar button:disabled { opacity: .4; cursor: default; }
.pagination-bar svg { width: 14px; }
</style>
