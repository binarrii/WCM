export const replicationPullUrl = (apiBase, pageUrl) => {
  const url = new URL(`${apiBase.replace(/\/$/, '')}/insightface/replication/ws`, pageUrl);
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
  return url.href;
};

/** Client-driven reads over one socket; at most one outstanding request. */
export const createReplicationPull = ({
  url, onData, onState = () => {}, onPending = () => {}, onError = () => {},
  socketFactory = address => new WebSocket(address), interval = 5000,
  requestTimeout = 20000, retryDelay = 500
}) => {
  let running = false;
  let socket;
  let opened = false;
  let pending;
  let queued = false;
  let nextId = 0;
  let attempts = 0;
  let pollTimer;
  let retryTimer;
  let deadline;

  const release = () => {
    clearTimeout(pollTimer);
    clearTimeout(deadline);
    opened = false;
    pending = null;
    queued = false;
    onPending(false);
    if (socket) {
      const previous = socket;
      socket = null;
      previous.onopen = previous.onmessage = previous.onerror = previous.onclose = null;
      previous.close();
    }
  };
  const reconnect = () => {
    if (!running) return;
    release();
    clearTimeout(retryTimer);
    onState('reconnecting');
    retryTimer = setTimeout(connect, Math.min(15000, retryDelay * 2 ** Math.min(attempts++, 6)));
  };
  const pull = () => {
    clearTimeout(pollTimer);
    if (!running || !opened) return;
    if (pending) { queued = true; return; }
    pending = { id: ++nextId, discard: false };
    onPending(true);
    deadline = setTimeout(reconnect, requestTimeout);
    try { socket.send(JSON.stringify({ type: 'status', request_id: pending.id })); }
    catch { reconnect(); }
  };
  const connect = () => {
    if (!running) return;
    onState(attempts ? 'reconnecting' : 'connecting');
    try { socket = socketFactory(url); } catch { reconnect(); return; }
    const connection = socket;
    const live = () => running && connection === socket;
    deadline = setTimeout(reconnect, requestTimeout);
    connection.onopen = () => {
      if (!live()) return;
      clearTimeout(deadline);
      opened = true;
      onState('connected');
      pull();
    };
    connection.onmessage = event => {
      if (!live()) return;
      let message;
      try { message = JSON.parse(event.data); } catch { reconnect(); return; }
      // A stale or unsolicited response cannot satisfy the current request.
      if (!message || !pending || message.request_id !== pending.id) return;
      if (message.type !== 'error' && (message.type !== 'status' || typeof message.data?.enabled !== 'boolean')) {
        reconnect(); return;
      }
      clearTimeout(deadline);
      const discard = pending.discard;
      pending = null;
      attempts = 0;
      onPending(false);
      if (!discard) {
        if (message.type === 'status') onData(message.data);
        else onError(new Error(typeof message.error === 'string' ? message.error : '同步状态读取失败，请稍后刷新。'));
      }
      if (!live()) return;
      if (queued) { queued = false; pull(); }
      else pollTimer = setTimeout(pull, interval);
    };
    connection.onerror = () => { if (live()) reconnect(); };
    connection.onclose = () => { if (live()) reconnect(); };
  };
  const start = () => {
    if (running) return;
    running = true;
    attempts = 0;
    connect();
  };
  const stop = () => {
    running = false;
    clearTimeout(retryTimer);
    release();
    onState('stopped');
  };
  const refresh = ({ discardPending = false } = {}) => {
    if (!running) return;
    if (discardPending && pending) pending.discard = true;
    if (opened) pull();
    else if (!socket) { clearTimeout(retryTimer); connect(); }
  };
  return { start, stop, refresh };
};
