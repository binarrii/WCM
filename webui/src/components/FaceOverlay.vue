<script setup>
import { computed, ref, watch } from 'vue';
import { objectDisplayMarker } from '../services/objectOverlay.js';
const props = defineProps({
  faces: { type: Array, default: () => [] },
  rect: Object,
  videoSize: Object,
  selected: String,
  mode: String
});
const emit = defineEmits(['select']);
const openCrosshair = ref('');
const markers = computed(() => props.faces.map(face => objectDisplayMarker(face, props.rect, props.videoSize)));
watch([markers, () => props.selected, () => props.mode], () => {
  if (props.mode === 'hidden' || !markers.value.some(face => face.crosshair && face.key === openCrosshair.value
    && face.candidates.some(candidate => candidate.markerId === props.selected))) openCrosshair.value = '';
});
const selectMarker = face => {
  if (face.crosshair) {
    openCrosshair.value = openCrosshair.value === face.key ? '' : face.key;
    if (!openCrosshair.value) return;
  } else openCrosshair.value = '';
  emit('select', face.candidates[0].markerId);
};
const closeDetails = event => {
  openCrosshair.value = '';
  event.currentTarget.querySelector('.face-target')?.focus();
};
const labelStyle = face => {
  const { width, height } = props.rect;
  const box = face.box;
  const labelWidth = Math.min(180, width - 16);
  const x = box.x * width;
  const y = box.y * height;
  const faceHeight = box.h * height;
  const left = Math.max(8, Math.min(x, width - labelWidth - 8));
  const above = y - 8;
  const below = height - y - faceHeight - 8;
  const useAbove = above >= Math.min(face.candidates.length * 38 + 8, 240) || above > below;
  // Anchor to the face edge so the pointer can cross into the tooltip without
  // a dead gap, even when the rendered name wraps or the candidate list scrolls.
  return {
    left: `${left - x}px`, top: useAbove ? '0' : `${faceHeight}px`,
    transform: useAbove ? 'translateY(-100%)' : 'none',
    width: 'max-content', maxWidth: `${labelWidth}px`, maxHeight: `${Math.max(38, Math.min(240, useAbove ? above : below))}px`
  };
};
const layerStyle = computed(() => props.rect ? Object.fromEntries(
  Object.entries(props.rect).map(([key, value]) => [key, `${value}px`])
) : {});
const boxStyle = box => ({
  left: `${box.x * 100}%`, top: `${box.y * 100}%`,
  width: `${box.w * 100}%`, height: `${box.h * 100}%`
});
</script>

<template>
  <div v-if="rect && mode !== 'hidden'" class="face-overlay" :style="layerStyle">
    <div v-for="face in markers" :key="face.key" class="face-hit"
      :class="{ selected: face.candidates.some(c => c.markerId === selected), full: mode === 'boxes', flag: face.objectType === 'flag', logo: face.objectType === 'logo', nudity: face.objectType === 'nudity', crosshair: face.crosshair, 'details-open': face.crosshair && openCrosshair === face.key }"
      :style="boxStyle(face.box)" @keydown.esc.stop.prevent="closeDetails">
      <button class="face-target" type="button"
        :style="face.crosshair ? { clipPath: face.targetClip } : null"
        :aria-expanded="face.crosshair ? openCrosshair === face.key : undefined"
        :aria-label="`查看候选：${face.candidates.map(c => c.name).join('、')}`"
        @click="selectMarker(face)">
        <svg v-if="face.crosshair" viewBox="0 0 32 32" aria-hidden="true" focusable="false">
          <circle cx="16" cy="16" r="13" />
          <path d="M16 7v18M7 16h18" />
        </svg>
        <template v-else><i v-for="corner in ['tl', 'tr', 'bl', 'br']" :key="corner" :class="corner" /></template>
      </button>
      <div class="face-label" :style="labelStyle(face)">
        <button v-for="candidate in face.candidates" :key="candidate.markerId + candidate.name" type="button"
          :aria-pressed="candidate.markerId === selected" @click="$emit('select', candidate.markerId)">
          <span>{{ candidate.name }}</span>
          <small v-if="candidate.needsReview">待复核</small>
          <small v-else-if="Number.isFinite(candidate.similarity)">{{ Math.round(candidate.similarity * 100) }}%</small>
        </button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.face-overlay { --face-frame-color: #ff3040; position: absolute; pointer-events: none; }
.face-hit.flag { --face-frame-color: #ffb020; }
.face-hit.logo { --face-frame-color: #24d7e8; }
.face-hit.nudity { --face-frame-color: #ed7aff; }
.face-hit { position: absolute; color: var(--face-frame-color); pointer-events: none; }
.face-target { position: absolute; inset: 0; width: 100%; height: 100%; padding: 0; border: 1px solid transparent; background: transparent; pointer-events: auto; cursor: pointer; border-radius: 3px; }
.face-target i { position: absolute; width: 13px; height: 13px; border: solid currentColor; border-width: 0; color: var(--face-frame-color); filter: drop-shadow(0 1px 1px #000b) drop-shadow(0 0 3px currentColor); }
.tl { top: 0; left: 0; border-top-width: 2px !important; border-left-width: 2px !important; }
.tr { top: 0; right: 0; border-top-width: 2px !important; border-right-width: 2px !important; }
.bl { bottom: 0; left: 0; border-bottom-width: 2px !important; border-left-width: 2px !important; }
.br { bottom: 0; right: 0; border-bottom-width: 2px !important; border-right-width: 2px !important; }
.face-hit:hover, .face-hit:focus-within, .face-hit.selected { z-index: 2; }
.face-hit:hover .face-target, .face-hit:focus-within .face-target, .selected .face-target, .full .face-target { border-color: var(--face-frame-color); box-shadow: 0 0 2px currentColor; }
.face-target:focus-visible { outline: 2px solid white; outline-offset: 3px; }
.face-label { display: none; position: absolute; padding: 4px 0; overflow-y: auto; pointer-events: auto; }
.face-hit:hover .face-label, .face-hit:focus-within .face-label, .selected .face-label, .full .face-label { display: grid; }
.face-label button { display: flex; gap: 6px; justify-content: flex-start; align-items: center; padding: 7px 10px; border: 0; background: #201f2eed; color: #fff; cursor: pointer; font-family: inherit; font-size: 10px; line-height: 1.5; text-align: left; }
.face-label button:first-child { border-radius: 6px 6px 0 0; }
.face-label button:last-child { border-radius: 0 0 6px 6px; }
.face-label button:only-child { border-radius: 6px; }
.face-label button:hover, .face-label button:focus-visible { background: #514054; }
.face-label span { overflow-wrap: anywhere; }
.face-label small { flex-shrink: 0; font-size: 8px; color: #ffb5b8; }
.face-hit.crosshair { --face-frame-color: #42ffb1; }
.face-hit.crosshair .face-target { border: 0; border-radius: 50%; box-shadow: none; color: inherit; }
.crosshair svg { display: block; width: 100%; height: 100%; fill: none; stroke: currentColor; stroke-width: 2; filter: drop-shadow(0 0 2px #000b); }
.face-hit.crosshair .face-label { display: none; }
.face-hit.crosshair.details-open .face-label { display: grid; }
</style>
