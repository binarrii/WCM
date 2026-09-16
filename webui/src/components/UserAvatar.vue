<script setup>
import { computed, ref, watch } from 'vue';
import { API_BASE } from '../services/api';
import { avatarInitial, avatarSource } from '../services/avatar';

const props = defineProps({ user: Object, size: { type: Number, default: 36 }, preview: { type: String, default: '' } });
const failed = ref(false);
const source = computed(() => props.preview || avatarSource(props.user, API_BASE));
watch(source, () => { failed.value = false; });
</script>

<template>
  <span class="user-avatar" :style="{ '--avatar-size': `${size}px` }" aria-hidden="true">
    <img v-if="source && !failed" :src="source" alt="" @error="failed = true" />
    <span v-else>{{ avatarInitial(user) }}</span>
  </span>
</template>

<style scoped>
.user-avatar { width: var(--avatar-size); height: var(--avatar-size); flex: 0 0 var(--avatar-size); display: inline-grid; place-items: center; overflow: hidden; border-radius: 50%; background: var(--color-primary-glow); color: var(--color-primary); box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--color-primary) 12%, transparent); font: 650 calc(var(--avatar-size) * .4)/1 var(--font-display); vertical-align: middle; }
.user-avatar img { display: block; width: 100%; height: 100%; object-fit: cover; }
</style>
