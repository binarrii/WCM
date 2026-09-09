/** Processing has finished and results (including incomplete checks) can be reviewed. */
export const reviewResultsReady = task => {
  const status = typeof task === 'string' ? task : task?.status;
  if (status === 'processing') return false;
  return status === 'completed' || status === 'partial'
    || (status === 'failed' && (task?.has_results === true || Array.isArray(task?.results)));
};
