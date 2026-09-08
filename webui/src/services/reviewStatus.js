/** Processing has finished and results (including incomplete checks) can be reviewed. */
export const reviewResultsReady = status => status === 'completed' || status === 'partial';
