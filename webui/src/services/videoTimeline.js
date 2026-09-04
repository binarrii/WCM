const TIMESTAMP = /^(\d+):([0-5]\d):([0-5]\d)(?:\.(\d{1,3}))?$/;

export function timestampMs(value) {
  if (typeof value === 'boolean' || !['string', 'number'].includes(typeof value)) {
    throw new Error(`无效时间戳：${String(value)}`);
  }
  const text = String(value).trim();
  if (text.includes(':')) {
    const match = text.match(TIMESTAMP);
    if (!match) throw new Error(`时间戳格式错误：${text}`);
    const [, hours, minutes, seconds, fraction = '0'] = match;
    return ((Number(hours) * 3600 + Number(minutes) * 60 + Number(seconds)) * 1000)
      + Number(fraction.padEnd(3, '0'));
  }
  const seconds = Number(text);
  if (!Number.isFinite(seconds) || seconds < 0) throw new Error(`无效时间戳：${text}`);
  return Math.round(seconds * 1000);
}

export function formatTimestamp(milliseconds) {
  let value = Math.max(0, Math.round(milliseconds));
  const hours = Math.floor(value / 3600000); value %= 3600000;
  const minutes = Math.floor(value / 60000); value %= 60000;
  const seconds = Math.floor(value / 1000);
  return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}.${String(value % 1000).padStart(3, '0')}`;
}

function timestampRange(value) {
  if (typeof value === 'string' && value.includes('~')) {
    const parts = value.split('~');
    if (parts.length !== 2) throw new Error(`无效时间区间：${value}`);
    const start = timestampMs(parts[0]);
    const end = timestampMs(parts[1]);
    if (end < start) throw new Error(`时间区间结束时间早于开始时间：${value}`);
    return [start, end];
  }
  const point = timestampMs(value);
  return [point, point];
}

export function normalizeResults(payload) {
  if (payload?.status === 'error' || payload?.status === 'failed') {
    throw new Error(payload.error || payload.detail || '视频分析失败');
  }
  const results = Array.isArray(payload) ? payload : payload?.results;
  if (!Array.isArray(results)) throw new Error('审核结果必须是数组，或包含 results 数组的对象');
  const grouped = new Map();
  results.forEach((item, index) => {
    if (!item || typeof item !== 'object' || !Object.hasOwn(item, 'timestamp')) {
      throw new Error(`第 ${index + 1} 条结果缺少 timestamp`);
    }
    const [start, end] = timestampRange(item.timestamp);
    const category = item.category || '未分类';
    const description = item.description || '';
    if (typeof category !== 'string' || typeof description !== 'string') {
      throw new Error(`第 ${index + 1} 条 category/description 必须是文本`);
    }
    const key = [start, end, end > start ? category : '', end > start ? description : ''].join('\u0000');
    if (!grouped.has(key)) grouped.set(key, { start, end, findings: [] });
    const finding = { category, description };
    const findings = grouped.get(key).findings;
    if (!findings.some(value => value.category === category && value.description === description)) {
      findings.push(finding);
    }
  });
  return [...grouped.values()]
    .sort((a, b) => a.start - b.start || a.end - b.end)
    .map((marker, index) => ({
      id: `${marker.start}-${marker.end}-${index}`,
      timestamp: formatTimestamp(marker.start) + (marker.end > marker.start ? `~${formatTimestamp(marker.end)}` : ''),
      time_ms: marker.start,
      end_time_ms: marker.end,
      findings: marker.findings
    }));
}

export function layoutMarkers(markers, durationMs, width) {
  if (!Number.isFinite(durationMs) || durationMs <= 0 || width <= 0) return [];
  const lanes = [];
  return markers.map((marker) => {
    const left = Math.min(width - 12, marker.time_ms / durationMs * width);
    const size = Math.min(width - left, Math.max(12, (marker.end_time_ms - marker.time_ms) / durationMs * width));
    let lane = lanes.findIndex(right => right + 4 <= left);
    if (lane < 0) lane = lanes.length;
    lanes[lane] = left + size;
    return { left: Math.max(0, left), width: Math.max(1, size), lane };
  });
}

export function markerIsActive(marker, currentSeconds) {
  const current = currentSeconds * 1000;
  return marker.end_time_ms > marker.time_ms
    ? current >= marker.time_ms - 50 && current <= marker.end_time_ms + 50
    : Math.abs(marker.time_ms - current) < 100;
}

export function similarityToDistance(similarity) {
  const value = Number(similarity);
  if (!Number.isFinite(value) || value < 0.1 || value > 1) throw new Error('最低相似度必须在 10%～100% 之间');
  return value === 1 ? 0.000001 : Number((1 - value).toFixed(6));
}

export function validateVideoUrl(value) {
  let parsed;
  try { parsed = new URL(String(value).trim()); } catch { throw new Error('请输入有效的视频 HTTP(S) 地址'); }
  if (!['http:', 'https:'].includes(parsed.protocol)) throw new Error('视频地址仅支持 HTTP(S)');
  return parsed.href;
}

export function buildAnalyzePayload({ url, sampleInterval = 1, topK = 10, minSimilarity = 0.5 }) {
  const interval = Number(sampleInterval);
  const limit = Number(topK);
  if (!Number.isFinite(interval) || interval <= 0) throw new Error('采样间隔必须大于 0 秒');
  if (!Number.isInteger(limit) || limit < 1 || limit > 10) throw new Error('人脸候选数必须在 1～10 之间');
  return {
    url: validateVideoUrl(url),
    sample_interval: interval,
    top_k: limit,
    threshold: similarityToDistance(minSimilarity)
  };
}
