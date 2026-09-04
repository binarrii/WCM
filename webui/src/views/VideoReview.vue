<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue';
import { AlertCircle, ChevronLeft, ChevronRight, Download, FileJson, Play, Search, Settings, Video } from '@lucide/vue';
import { mediaService } from '../services/mediaService';
import {
  formatTimestamp,
  layoutMarkers,
  markerIsActive,
  normalizeResults,
  validateVideoUrl
} from '../services/videoTimeline';

const DEFAULT_SIMILARITY = 0.5;
const inputUrl = ref('');
const videoUrl = ref('');
const sampleInterval = ref(1);
const minSimilarity = ref(DEFAULT_SIMILARITY);
const topK = ref(10);
const loading = ref(false);
const error = ref('');
const rawResults = ref(null);
const markers = ref([]);
const category = ref('');
const currentSeconds = ref(0);
const durationMs = ref(0);
const videoRef = ref(null);
const playerPanelRef = ref(null);
const timelineRef = ref(null);
const eventsRef = ref(null);
const jsonInputRef = ref(null);
const timelineWidth = ref(0);
const eventRows = new Map();
let resizeObserver;
let panelResizeObserver;
const playerPanelHeight = ref(0);

const categories = computed(() => [...new Set(markers.value.flatMap(marker => marker.findings.map(finding => finding.category)))]);
const visibleMarkers = computed(() => markers.value.filter(marker => !category.value || marker.findings.some(finding => finding.category === category.value)));
const markerLayout = computed(() => layoutMarkers(visibleMarkers.value, durationMs.value, timelineWidth.value));
const laneCount = computed(() => Math.max(1, ...markerLayout.value.map(item => item.lane + 1)));
const clock = computed(() => formatTimestamp(currentSeconds.value * 1000));
const duration = computed(() => durationMs.value ? formatTimestamp(durationMs.value) : '--:--:--.---');
const previousIndex = computed(() => {
  for (let index = visibleMarkers.value.length - 1; index >= 0; index -= 1) {
    if (visibleMarkers.value[index].time_ms / 1000 < currentSeconds.value - 0.05) return index;
  }
  return -1;
});
const nextIndex = computed(() => visibleMarkers.value.findIndex(marker => marker.time_ms / 1000 > currentSeconds.value + 0.05));

const details = marker => marker.findings.map(finding => `${finding.category}：${finding.description}`).join('\n');
const markerActive = marker => markerIsActive(marker, currentSeconds.value);
const markerStyle = index => {
  const item = markerLayout.value[index];
  return item ? { left: `${item.left}px`, width: `${item.width}px`, top: `${8 + item.lane * 26}px` } : {};
};
const setEventRow = (id, element) => {
  if (element) eventRows.set(id, element);
  else eventRows.delete(id);
};
const revealEvent = (id) => {
  const row = eventRows.get(id);
  const list = eventsRef.value;
  if (!row || !list?.clientHeight) return;
  const bounds = row.getBoundingClientRect();
  const top = list.getBoundingClientRect().top + list.clientTop;
  const bottom = top + list.clientHeight;
  if (bounds.top < top || bounds.height > list.clientHeight) list.scrollTop += Math.floor(bounds.top - top);
  else if (bounds.bottom > bottom) list.scrollTop += Math.ceil(bounds.bottom - bottom);
};
const jump = async (index) => {
  const marker = visibleMarkers.value[index];
  if (!marker) return;
  currentSeconds.value = marker.time_ms / 1000;
  if (videoRef.value) videoRef.value.currentTime = currentSeconds.value;
  await nextTick();
  revealEvent(marker.id);
};

const loadResults = (payload) => {
  const normalized = normalizeResults(payload);
  rawResults.value = payload;
  markers.value = normalized;
  category.value = '';
  currentSeconds.value = 0;
};
const analyze = async () => {
  error.value = '';
  let url;
  try { url = validateVideoUrl(inputUrl.value); } catch (reason) { error.value = reason.message; return; }
  if (!Number.isFinite(sampleInterval.value) || sampleInterval.value <= 0) {
    error.value = '采样间隔必须大于 0 秒'; return;
  }
  videoUrl.value = url;
  rawResults.value = null;
  markers.value = [];
  category.value = '';
  loading.value = true;
  try {
    const payload = await mediaService.analyzeVideo({
      url,
      sampleInterval: sampleInterval.value,
      topK: topK.value,
      minSimilarity: minSimilarity.value
    });
    loadResults(payload);
  } catch (reason) {
    error.value = reason.response?.data?.detail || reason.message || '视频分析失败';
  } finally {
    loading.value = false;
  }
};
const importResults = async (event) => {
  const file = event.target.files?.[0];
  event.target.value = '';
  if (!file) return;
  error.value = '';
  try {
    const payload = JSON.parse(await file.text());
    const url = validateVideoUrl(inputUrl.value);
    loadResults(payload);
    videoUrl.value = url;
  } catch (reason) {
    error.value = reason instanceof SyntaxError ? 'JSON 文件格式错误' : reason.message;
  }
};
const downloadResults = () => {
  if (rawResults.value == null) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(rawResults.value, null, 2)], { type: 'application/json' }));
  const link = document.createElement('a');
  link.href = url;
  link.download = 'analysis.json';
  link.click();
  URL.revokeObjectURL(url);
};
const handleMetadata = () => {
  durationMs.value = Number.isFinite(videoRef.value?.duration) ? Math.round(videoRef.value.duration * 1000) : 0;
};
const handleTimeUpdate = () => { currentSeconds.value = videoRef.value?.currentTime || 0; };
const handleSeek = (event) => {
  currentSeconds.value = Number(event.target.value);
  if (videoRef.value) videoRef.value.currentTime = currentSeconds.value;
};
const updateWidth = () => { timelineWidth.value = timelineRef.value?.clientWidth || 0; };
const observeTimeline = (element) => {
  resizeObserver?.disconnect();
  if (!element) {
    timelineWidth.value = 0;
    return;
  }
  resizeObserver?.observe(element);
  updateWidth();
};
const updatePlayerPanelHeight = () => { playerPanelHeight.value = playerPanelRef.value?.offsetHeight || 0; };
const observePlayerPanel = (element) => {
  panelResizeObserver?.disconnect();
  if (!element) {
    playerPanelHeight.value = 0;
    return;
  }
  panelResizeObserver?.observe(element);
  updatePlayerPanelHeight();
};

watch(category, () => {
  eventRows.clear();
  nextTick(updateWidth);
});
watch(timelineRef, observeTimeline, { flush: 'post' });
watch(playerPanelRef, observePlayerPanel, { flush: 'post' });
onMounted(() => {
  resizeObserver = new ResizeObserver(updateWidth);
  panelResizeObserver = new ResizeObserver(updatePlayerPanelHeight);
  observeTimeline(timelineRef.value);
  observePlayerPanel(playerPanelRef.value);
});
onBeforeUnmount(() => {
  resizeObserver?.disconnect();
  panelResizeObserver?.disconnect();
});
</script>

<template>
  <div class="video-review animate-fade-in">
    <section class="review-setup-card">
      <div class="setup-heading">
        <div><h2>创建视频复核时间轴</h2><p>输入可访问的视频地址，系统会识别人脸及其他疑似违规内容。</p></div>
        <span class="review-safety-note">标记仅用于人工复核，不代表违规结论</span>
      </div>
      <form class="review-form" @submit.prevent="analyze">
        <label class="url-field"><span>视频 HTTP(S) 地址</span><input v-model.trim="inputUrl" type="url" placeholder="http://10.252.25.251:18080/videos/example.mp4" required /></label>
        <label><span>采样间隔</span><div class="input-unit"><input v-model.number="sampleInterval" type="number" min="0.1" step="0.1" /><small>秒</small></div></label>
        <label><span>人脸候选数</span><select v-model.number="topK"><option v-for="value in [1, 3, 5, 10]" :key="value" :value="value">{{ value }}</option></select></label>
        <button class="analyze-button" type="submit" :disabled="loading"><Settings v-if="loading" class="spinner" /><Search v-else />{{ loading ? '分析中…' : '开始分析' }}</button>
      </form>
      <div class="similarity-control">
        <div><label for="video-similarity">最低人脸相似度</label><output>{{ Math.round(minSimilarity * 100) }}%</output></div>
        <input id="video-similarity" v-model.number="minSimilarity" type="range" min="0.1" max="1" step="0.1" />
        <div class="similarity-scale"><span v-for="value in 10" :key="value">{{ value * 10 }}%</span></div>
      </div>
      <div class="setup-secondary-actions">
        <input ref="jsonInputRef" class="hidden-file" type="file" accept="application/json,.json" @change="importResults" />
        <button type="button" @click="jsonInputRef.click()"><FileJson />导入已有结果</button>
        <button type="button" :disabled="rawResults == null" @click="downloadResults"><Download />下载分析结果</button>
        <span>导入 JSON 时仍需填写视频地址，独立脚本及章节 MP4 导出方式保持不变。</span>
      </div>
      <p v-if="error" class="review-error" role="alert"><AlertCircle />{{ error }}</p>
    </section>

    <section v-if="videoUrl || rawResults != null" class="review-workspace" :style="{ '--review-player-height': playerPanelHeight ? `${playerPanelHeight}px` : 'auto' }">
      <div ref="playerPanelRef" class="review-player-panel">
        <video ref="videoRef" :src="videoUrl" controls preload="metadata" @loadedmetadata="handleMetadata" @timeupdate="handleTimeUpdate" @error="error = '视频无法播放，请确认地址可访问且服务支持 Range 请求'" />
        <div class="review-toolbar">
          <button type="button" :disabled="previousIndex < 0" @click="jump(previousIndex)"><ChevronLeft />上一标记</button>
          <button type="button" :disabled="nextIndex < 0" @click="jump(nextIndex)">下一标记<ChevronRight /></button>
          <span>{{ clock }}</span>
        </div>
        <label class="progress-label" for="review-progress">视频进度</label>
        <input id="review-progress" class="review-progress" type="range" min="0" :max="durationMs / 1000 || 1" step="0.001" :value="currentSeconds" @input="handleSeek" />
        <div ref="timelineRef" class="review-timeline" :style="{ height: `${laneCount * 26 + 12}px` }" aria-label="审核时间轴">
          <button v-for="(marker, index) in visibleMarkers" :key="marker.id" type="button" :class="['timeline-marker', { range: marker.end_time_ms > marker.time_ms, active: markerActive(marker) }]" :style="markerStyle(index)" :title="`${marker.timestamp}\n${details(marker)}`" :aria-label="`跳转 ${marker.timestamp}`" @click="jump(index)"></button>
        </div>
        <div class="timeline-scale"><span>00:00:00.000</span><span>{{ duration }}</span></div>
        <p class="timeline-hint"><i class="range-key"></i>人物连续命中区间 <i class="point-key"></i>单次命中；重叠标记自动分行。</p>
      </div>

      <aside class="review-events-panel">
        <div class="events-header"><h2>{{ visibleMarkers.length }} 条标记 <small>· 共 {{ markers.length }} 条</small></h2><label><span>类别</span><select v-model="category"><option value="">全部类别</option><option v-for="value in categories" :key="value" :value="value">{{ value }}</option></select></label></div>
        <div ref="eventsRef" class="review-events">
          <button v-for="(marker, index) in visibleMarkers" :key="marker.id" :ref="element => setEventRow(marker.id, element)" type="button" :class="['review-event', { active: markerActive(marker) }]" @click="jump(index)"><strong>{{ marker.timestamp }}</strong><span>{{ details(marker) }}</span></button>
          <div v-if="rawResults != null && !visibleMarkers.length" class="empty-review"><Video /><p>没有审核标记</p><small>无标记不代表内容安全，请结合人工复核。</small></div>
        </div>
      </aside>
    </section>

    <section v-else class="review-empty-state"><Play /><h2>等待分析视频</h2><p>分析完成后，这里会显示播放器、审核时间轴和分类结果列表。</p></section>
  </div>
</template>

<style scoped src="./video-review.css"></style>
