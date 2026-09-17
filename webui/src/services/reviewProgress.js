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

export const downloadSize = bytes => {
  const value = finite(bytes) ? Math.max(0, bytes) : 0;
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  const index = value > 0 ? Math.max(0, Math.min(units.length - 1, Math.floor(Math.log(value) / Math.log(1024)))) : 0;
  return `${(value / 1024 ** index).toFixed(index ? 1 : 0)} ${units[index]}`;
};

export const taskProgress = task => {
  const progress = task?.progress || {};
  const finished = !['processing', 'cancelling', 'cancelled'].includes(task?.status) && (
    progress.phase === 'finished' || ['completed', 'partial'].includes(task?.status)
  );
  const phase = ['cancelling', 'cancelled'].includes(task?.status) ? task.status
    : finished ? 'finished' : task?.status === 'failed' ? 'failed' : progress.phase;
  const percent = finished ? 100 : finite(progress.percent) ? Math.max(0, Math.min(99, progress.percent)) : null;
  const label = { queued: '排队中', downloading: '下载中', preparing: '准备复核视频', archiving_media: '保存复核视频', reviewing: '审核中', resampling: '审核收尾', saving: '保存结果', finished: '处理结束', failed: '已停止', cancelling: '取消中…', cancelled: '已取消' }[phase] || '等待进度';
  const details = [];
  if (!['queued', 'downloading', 'preparing', 'archiving_media'].includes(phase)) {
    if (finite(progress.completed_windows)) details.push(`窗口 ${progress.completed_windows}${finite(progress.total_windows) ? ` / ${progress.total_windows}` : ''}`);
    if (finite(progress.completed_samples)) details.push(`采样 ${progress.completed_samples}${finite(progress.total_samples) ? ` / ${progress.total_samples}` : ''}`);
  }
  if (finite(progress.elapsed_seconds)) details.push(`用时 ${elapsedTime(progress.elapsed_seconds)}`);
  const stageLabels = { visual: '视觉', ocr: '文字', face: '人脸' };
  const windows = (phase !== 'cancelled' && Array.isArray(progress.active_windows) ? progress.active_windows : []).map(window => ({
    title: `窗口 #${window.index} · ${sampleTime(window.start_seconds)}～${sampleTime(window.end_seconds)}`,
    samples: `采样点 ${window.sample_timestamps.map(sampleTime).join('、')}`,
    stages: Object.entries(window.stages || {}).map(([stage, samples]) => `${stageLabels[stage] || stage}：${samples.map(sampleTime).join('、')}`).join('；')
  }));
  const rawSubProgress = progress.sub_progress;
  let subProgress = task?.status === 'processing' && phase === 'resampling'
    && rawSubProgress?.stage === 'face_resampling' && finite(rawSubProgress.total) && rawSubProgress.total > 0
    ? (() => {
        const completed = finite(rawSubProgress.completed)
          ? Math.max(0, Math.min(rawSubProgress.total, rawSubProgress.completed)) : 0;
        return {
          label: '困难帧补采',
          percent: completed / rawSubProgress.total * 100,
          details: `${completed} / ${rawSubProgress.total} 帧`
        };
      })()
    : null;
  if (task?.status === 'processing' && phase === 'downloading' && rawSubProgress?.stage === 'download') {
    const completed = finite(rawSubProgress.completed) ? Math.max(0, rawSubProgress.completed) : 0;
    const total = finite(rawSubProgress.total) && rawSubProgress.total > 0 && rawSubProgress.total >= completed
      ? rawSubProgress.total : null;
    subProgress = {
      label: '视频下载',
      percent: rawSubProgress.complete === true ? 100 : total ? Math.min(99.9, completed / total * 100) : null,
      details: total ? `${downloadSize(completed)} / ${downloadSize(total)}` : `已下载 ${downloadSize(completed)} · 总大小未知`
    };
  }
  return { label, percent, details: details.join(' · '), windows, subProgress, active: ['processing', 'cancelling'].includes(task?.status) };
};
