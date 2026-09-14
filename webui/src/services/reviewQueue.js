import { reviewTaskActive, reviewResultsReady } from './reviewStatus.js';

/** Submit once over HTTP; viewing/reconnecting never submits another job. */
export const submitQueuedReview = async ({ payload, submit, read, stream, onTaskAccepted = () => {}, onTaskEvent = () => {}, signal, pollInterval = 5000 }) => {
  const accepted = await submit(payload, signal);
  const taskId = accepted.id;
  onTaskAccepted(taskId);
  return new Promise((resolve, reject) => {
    let done = false;
    let reading = false;
    let timer;
    let subscription;
    const finish = (task, error) => {
      if (done) return;
      done = true;
      clearInterval(timer);
      signal?.removeEventListener('abort', abort);
      subscription?.stop();
      if (error) { error.taskId = taskId; reject(error); }
      else { onTaskEvent({ type: 'completed', task }); resolve(task.results); }
    };
    const abort = () => finish(null, new Error('审核页面已关闭，任务继续在后台执行'));
    const refresh = async () => {
      if (done || reading) return;
      reading = true;
      try {
        const task = await read(taskId);
        if (done) return;
        onTaskEvent({ type: 'snapshot', tasks: [task], missing_ids: [] });
        if (!reviewTaskActive(task)) {
          if (reviewResultsReady(task)) finish(task);
          else {
            const error = new Error(task.status === 'cancelled' ? '审核任务已取消' : task.error || '审核失败');
            error.cancelled = task.status === 'cancelled';
            finish(null, error);
          }
        }
      } catch (error) {
        if (error.response?.status === 404) finish(null, new Error('审核任务已删除'));
        // A transient connection failure is repaired by the next DB-backed read.
      } finally { reading = false; }
    };
    subscription = stream({
      onEvent: event => { if (!done) { onTaskEvent(event); if (event.type !== 'progress') refresh(); } },
      onError: () => {},
    });
    if (signal?.aborted) { abort(); return; }
    signal?.addEventListener('abort', abort, { once: true });
    subscription.start([taskId]);
    if (done) return;
    timer = setInterval(refresh, pollInterval);
    refresh();
  });
};
