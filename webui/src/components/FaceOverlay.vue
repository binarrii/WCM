<script setup>
import { computed } from 'vue';
const props = defineProps({
  faces: { type: Array, default: () => [] },
  rect: Object,
  selected: String,
  mode: String
});
defineEmits(['select']);
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
    width: `${labelWidth}px`, maxHeight: `${Math.max(38, Math.min(240, useAbove ? above : below))}px`
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
    <div v-for="face in faces" :key="face.key" class="face-hit"
      :class="{ selected: face.candidates.some(c => c.markerId === selected), full: mode === 'boxes' }"
      :style="boxStyle(face.box)">
      <button class="face-target" type="button"
        :aria-label="`查看候选：${face.candidates.map(c => c.name).join('、')}`"
        @click="$emit('select', face.candidates[0].markerId)">
        <i v-for="corner in ['tl', 'tr', 'bl', 'br']" :key="corner" :class="corner" />
      </button>
      <div class="face-label" :style="labelStyle(face)">
        <button v-for="candidate in face.candidates" :key="candidate.markerId + candidate.name" type="button"
          :aria-pressed="candidate.markerId === selected" @click="$emit('select', candidate.markerId)">
          <span>候选 · {{ candidate.name }}</span>
          <small v-if="Number.isFinite(candidate.similarity)">{{ Math.round(candidate.similarity * 100) }}%</small>
        </button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.face-overlay { position: absolute; pointer-events: none; }
.face-hit { position: absolute; color: #ff656a; pointer-events: none; }
.face-target { position: absolute; inset: 0; width: 100%; height: 100%; padding: 0; border: 1px solid transparent; background: transparent; pointer-events: auto; cursor: pointer; border-radius: 3px; }
.face-target i { position: absolute; width: 13px; height: 13px; border: solid currentColor; border-width: 0; color: #ff656a; filter: drop-shadow(0 1px 2px #000); }
.tl { top: 0; left: 0; border-top-width: 2px !important; border-left-width: 2px !important; }
.tr { top: 0; right: 0; border-top-width: 2px !important; border-right-width: 2px !important; }
.bl { bottom: 0; left: 0; border-bottom-width: 2px !important; border-left-width: 2px !important; }
.br { bottom: 0; right: 0; border-bottom-width: 2px !important; border-right-width: 2px !important; }
.face-hit:hover, .face-hit:focus-within, .face-hit.selected { z-index: 2; }
.face-hit:hover .face-target, .face-hit:focus-within .face-target, .selected .face-target, .full .face-target { border-color: #ff656a; box-shadow: 0 0 0 1px #0003; }
.face-target:focus-visible { outline: 2px solid white; outline-offset: 3px; }
.face-label { display: none; position: absolute; padding: 4px 0; overflow-y: auto; pointer-events: auto; }
.face-hit:hover .face-label, .face-hit:focus-within .face-label, .selected .face-label, .full .face-label { display: grid; }
.face-label button { display: flex; gap: 12px; justify-content: space-between; align-items: center; padding: 7px 10px; border: 0; background: #201f2eed; color: #fff; cursor: pointer; font-family: inherit; font-size: 12px; line-height: 1.5; text-align: left; }
.face-label button:first-child { border-radius: 6px 6px 0 0; }
.face-label button:last-child { border-radius: 0 0 6px 6px; }
.face-label button:only-child { border-radius: 6px; }
.face-label button:hover, .face-label button:focus-visible { background: #514054; }
.face-label span { overflow-wrap: anywhere; }
.face-label small { flex-shrink: 0; color: #ffb5b8; }
</style>
