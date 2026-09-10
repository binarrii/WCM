/** Processing has finished and results (including incomplete checks) can be reviewed. */
export const reviewResultsReady = task => {
  const status = typeof task === 'string' ? task : task?.status;
  if (status === 'processing') return false;
  return status === 'completed' || status === 'partial'
    || (status === 'failed' && (task?.has_results === true || Array.isArray(task?.results)));
};

/** Hide minor coverage warnings in the task list without changing saved results. */
export const showReviewTaskWarning = task => {
  if (!task?.error) return false;
  if (task.status !== 'completed') return true;
  const { total_samples: total, incomplete_samples: incomplete } = task.review_summary || {};
  const measured = Number.isInteger(total) && total > 0
    && Number.isInteger(incomplete) && incomplete >= 0 && incomplete <= total;
  // Use counts so rounding a percentage cannot hide the warning at exactly 5%.
  return !measured || incomplete * 20 >= total;
};
