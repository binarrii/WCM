export const reviewStreamUrl = (apiBase, pageUrl) => {
  const url = new URL(`${apiBase.replace(/\/$/, '')}/ws/analyze_media`, pageUrl);
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
  return url.href;
};

export const mergeReviewTask = (current, next) => {
  if (!current) return next;
  if (current.status !== 'processing' && next.status === 'processing') return current;
  const merged = { ...current, ...next };
  if (next.status === 'processing' && (current.progress?.sequence ?? -1) > (next.progress?.sequence ?? -1)) {
    merged.progress = current.progress;
  }
  return merged;
};

/** One push subscription per page. Timers only supervise/reconnect the socket. */
export const createReviewStream = ({
  url, onEvent, onReady = () => {}, onState = () => {}, onError = () => {},
  socketFactory = address => new WebSocket(address), retryDelay = 500, heartbeatTimeout = 45000
}) => {
  let generation = 0;
  let socket;
  let retryTimer;
  let heartbeatTimer;
  let subscriptionKey;
  const release = () => {
    clearTimeout(heartbeatTimer);
    if (socket) {
      socket.onopen = socket.onmessage = socket.onerror = socket.onclose = null;
      socket.close();
      socket = null;
    }
  };
  const stop = () => {
    generation += 1;
    clearTimeout(retryTimer);
    release();
    subscriptionKey = null;
    onState('stopped');
  };
  const start = (taskIds, { watchList = false } = {}) => {
    const payload = { type: 'subscribe', task_ids: [...new Set(taskIds)].sort(), watch_list: watchList };
    const key = JSON.stringify(payload);
    if (key === subscriptionKey) return;
    stop();
    subscriptionKey = key;
    const current = generation;
    let attempts = 0;
    const valid = () => current === generation;
    const deliver = (callback, value) => {
      Promise.resolve().then(() => { if (valid()) return callback(value); }).catch(error => { if (valid()) onError(error); });
    };
    const reconnect = () => {
      if (!valid()) return;
      release();
      clearTimeout(retryTimer);
      onState('reconnecting');
      retryTimer = setTimeout(connect, Math.min(15000, retryDelay * 2 ** Math.min(attempts++, 6)));
    };
    const keepAlive = () => {
      clearTimeout(heartbeatTimer);
      heartbeatTimer = setTimeout(reconnect, heartbeatTimeout);
    };
    const connect = () => {
      if (!valid()) return;
      onState(attempts ? 'reconnecting' : 'connecting');
      try { socket = socketFactory(url); } catch { reconnect(); return; }
      const connection = socket;
      const live = () => valid() && connection === socket;
      keepAlive();
      connection.onopen = () => { if (live()) { connection.send(key); keepAlive(); } };
      connection.onmessage = event => {
        if (!live()) return;
        keepAlive();
        let data;
        try { data = JSON.parse(event.data); } catch { reconnect(); return; }
        if (data.type === 'snapshot') {
          attempts = 0;
          onState('connected');
          deliver(onEvent, data);
          deliver(onReady);
        } else if (data.type === 'error') {
          stop();
          onError(new Error(data.error || '进度订阅失败'));
        } else if (data.type !== 'heartbeat') deliver(onEvent, data);
      };
      connection.onerror = () => { if (live()) reconnect(); };
      connection.onclose = event => {
        if (!live()) return;
        if (event.code === 1008) { stop(); onError(new Error('进度订阅无效')); }
        else reconnect();
      };
    };
    connect();
  };
  return { start, stop };
};
