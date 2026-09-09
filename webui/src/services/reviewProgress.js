const finite = value => typeof value === 'number' && Number.isFinite(value);

export const sampleTime = seconds => {
  if (!finite(seconds)) return '—';
  const milliseconds = Math.max(0, Math.round(seconds * 1000));
  return `${elapsedTime(milliseconds / 1000)}.${String(milliseconds % 1000).padStart(3, '0')}`;
};

export const elapsedTime = seconds => {
  if (!finite(seconds)) return '';
  const value = Math.max(0, Math.floor(seconds));
  return [Math.floor(value / 3600), Math.floor(value / 60) % 60, value % 60]
    .map(part => String(part).padStart(2, '0')).join(':');
};

export const taskProgress = task => {
  const progress = task?.progress || {};
  const finished = task?.status !== 'processing' && (
    progress.phase === 'finished' || ['completed', 'partial'].includes(task?.status)
  );
  const phase = finished ? 'finished' : task?.status === 'failed' ? 'failed' : progress.phase;
  const percent = finished ? 100 : finite(progress.percent) ? Math.max(0, Math.min(99, progress.percent)) : null;
  const label = { downloading: '下载中', reviewing: '审核中', saving: '保存结果', finished: '处理结束', failed: '已停止' }[phase] || '等待进度';
  const details = [];
  if (finite(progress.completed_windows)) details.push(`窗口 ${progress.completed_windows}${finite(progress.total_windows) ? ` / ${progress.total_windows}` : ''}`);
  if (finite(progress.completed_samples)) details.push(`采样 ${progress.completed_samples}${finite(progress.total_samples) ? ` / ${progress.total_samples}` : ''}`);
  if (finite(progress.elapsed_seconds)) details.push(`用时 ${elapsedTime(progress.elapsed_seconds)}`);
  const stageLabels = { visual: '视觉', ocr: '文字', face: '人脸' };
  const windows = (Array.isArray(progress.active_windows) ? progress.active_windows : []).map(window => ({
    title: `窗口 #${window.index} · ${sampleTime(window.start_seconds)}～${sampleTime(window.end_seconds)}`,
    samples: `采样点 ${window.sample_timestamps.map(sampleTime).join('、')}`,
    stages: Object.entries(window.stages || {}).map(([stage, samples]) => `${stageLabels[stage] || stage}：${samples.map(sampleTime).join('、')}`).join('；')
  }));
  return { label, percent, details: details.join(' · '), windows, active: task?.status === 'processing' };
};

/** Run the next poll only after the previous request settles; stop invalidates in-flight results. */
export const createTaskPoller = ({ getTask, onTask, onError = () => {}, delay = 2000 }) => {
  let generation = 0;
  let timer;
  const stop = () => { generation += 1; clearTimeout(timer); };
  const start = id => {
    stop();
    const current = generation;
    const poll = async () => {
      let active = true;
      try {
        const task = await getTask(id);
        if (current !== generation) return;
        onTask(task);
        active = task.status === 'processing';
      } catch (error) {
        if (current !== generation) return;
        onError(error);
      }
      if (active && current === generation) timer = setTimeout(poll, delay);
    };
    poll();
  };
  return { start, stop };
};
