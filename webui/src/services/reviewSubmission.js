export const reviewSocketUrl = (apiBase, pageUrl) => {
  const url = new URL(`${apiBase.replace(/\/$/, '')}/ws/analyze_media`, pageUrl);
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
  return url.href;
};

/** The existing websocket endpoint provides a task id immediately; never resubmit on disconnect. */
export const submitReview = (url, payload, onTaskAccepted, socketFactory = address => new WebSocket(address)) => new Promise((resolve, reject) => {
  const socket = socketFactory(url);
  let settled = false;
  let taskId;
  const fail = message => {
    if (settled) return;
    settled = true;
    const error = new Error(message);
    error.taskId = taskId;
    reject(error);
    socket.close();
  };
  socket.onopen = () => socket.send(JSON.stringify(payload));
  socket.onmessage = event => {
    if (settled) return;
    let data;
    try { data = JSON.parse(event.data); } catch { fail('审核服务返回了无效响应'); return; }
    if (data.status === 'accepted') {
      taskId = data.taskId;
      onTaskAccepted(taskId);
    } else if (data.status === 'completed') {
      settled = true;
      resolve(data.results);
      socket.close();
    } else if (data.status === 'error') {
      fail(data.error || '视频分析失败');
    }
  };
  socket.onerror = () => fail('审核连接中断');
  socket.onclose = () => fail('审核连接已断开');
});
