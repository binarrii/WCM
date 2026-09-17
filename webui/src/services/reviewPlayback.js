export function reviewPlayback(task, source, now = Date.now()) {
  const media = task?.media;
  if (media) {
    if (Date.parse(media.expires_at) <= now) return { url: '', offset: 0, message: '复核视频已超过保存期限，请重新提交审核。' };
    return { url: media.url, offset: Number(media.video_start_seconds) || 0, message: '' };
  }
  if (task && ['queued', 'processing', 'cancelling'].includes(task.status)) {
    return { url: '', offset: 0, message: '正在准备复核视频，准备完成后可播放。' };
  }
  let path = '';
  try { path = new URL(source).pathname.toLowerCase(); } catch { /* URL validation belongs to the form. */ }
  if (source && !/\.(mp4|webm|mov)$/.test(path)) {
    return { url: '', offset: 0, message: '此来源需要生成复核视频，请提交新的审核任务。历史结果和 JSON 导入不会自动转换原视频。' };
  }
  return { url: source, offset: 0, message: '' };
}

export function playbackErrorMessage(error) {
  return ({
    1: '视频加载被中断，请重新加载。',
    2: '视频读取失败，请检查网络、登录状态和复核视频保存期限。',
    3: '视频解码失败，请检查文件是否完整或更换浏览器。',
    4: '当前视频来源或编码无法播放；历史任务可重新提交以生成兼容视频。'
  })[error?.code] || '视频播放失败，请重新加载；审核任务的状态与结果仍保留。';
}
