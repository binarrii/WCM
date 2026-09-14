export const reviewSocketUrl = (apiBase, pageUrl) => {
  const url = new URL(`${apiBase.replace(/\/$/, '')}/ws/analyze_media`, pageUrl);
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
  return url.href;
};

/** The existing websocket endpoint provides a task id immediately; never resubmit on disconnect. */
export const submitReview = (url, payload, onTaskAccepted, socketFactory = address => new WebSocket(address), onTaskEvent = () => {}, signal) => new Promise((resolve, reject) => {
  const socket = socketFactory(url);
  let settled = false;
  let taskId;
  let heartbeatTimer;
  const abort = () => fail('审核页面已关闭');
  const keepAlive = () => { clearTimeout(heartbeatTimer); heartbeatTimer = setTimeout(() => fail('审核连接超时'), 45000); };
  const fail = (message, cancelled = false) => {
    if (settled) return;
    settled = true;
    clearTimeout(heartbeatTimer);
    signal?.removeEventListener('abort', abort);
    const error = new Error(message);
    error.taskId = taskId;
    error.cancelled = cancelled;
    reject(error);
    socket.close();
  };
  keepAlive();
  socket.onopen = () => { if (!settled) { socket.send(JSON.stringify(payload)); keepAlive(); } };
  socket.onmessage = event => {
    if (settled) return;
    keepAlive();
    let data;
    try { data = JSON.parse(event.data); } catch { fail('审核服务返回了无效响应'); return; }
    if (data.type === 'progress' || data.type === 'changed') {
      onTaskEvent(data);
    } else if (data.status === 'accepted') {
      taskId = data.taskId;
      onTaskAccepted(taskId);
    } else if (data.status === 'completed') {
      settled = true;
      clearTimeout(heartbeatTimer);
      signal?.removeEventListener('abort', abort);
      if (data.task) onTaskEvent({ type: 'completed', task: data.task });
      resolve(data.results);
      socket.close();
    } else if (data.status === 'cancelled') {
      onTaskEvent({ type: 'cancelled', task: { id: data.taskId || taskId, status: 'cancelled' } });
      fail('审核任务已取消', true);
    } else if (data.status === 'error') {
      fail(data.error || '视频分析失败');
    }
  };
  socket.onerror = () => fail('审核连接中断');
  socket.onclose = () => fail('审核连接已断开');
  if (signal?.aborted) abort();
  else signal?.addEventListener('abort', abort, {once:true});
});
