/** Observe frames submitted for display, including paused seeks, without polling. */
export function observeVideoFrames(video, onPresented) {
  let callbackId = null;
  let generation = 0;
  let stopped = false;
  const supported = typeof video?.requestVideoFrameCallback === 'function';
  const cancel = () => {
    if (callbackId != null) video.cancelVideoFrameCallback?.(callbackId);
    callbackId = null;
  };
  const schedule = () => {
    if (!supported || stopped) return;
    const expectedGeneration = generation;
    callbackId = video.requestVideoFrameCallback((_, metadata) => {
      if (stopped || generation !== expectedGeneration) return;
      callbackId = null;
      onPresented(Number.isFinite(metadata?.mediaTime) ? metadata.mediaTime : null);
      schedule();
    });
  };
  const invalidate = () => {
    generation += 1;
    cancel();
    onPresented(null);
    schedule();
  };
  video?.addEventListener('seeking', invalidate);
  video?.addEventListener('emptied', invalidate);
  invalidate();
  return {
    supported,
    invalidate,
    stop() {
      stopped = true;
      generation += 1;
      cancel();
      video?.removeEventListener('seeking', invalidate);
      video?.removeEventListener('emptied', invalidate);
      onPresented(null);
    }
  };
}
