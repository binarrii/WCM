export function replicaPresentation(node) {
  if (node.state === 'ready') {
    if (node.read_eligible) return { label: '可读 · 已追平', tone: 'success' };
    if (!node.heartbeat_fresh) return { label: '心跳过期', tone: 'warning' };
    return { label: '等待追平', tone: 'warning' };
  }
  return {
    new: { label: '待初始化', tone: 'neutral' },
    draining: { label: '等待读取结束', tone: 'active' },
    syncing: { label: '同步中', tone: 'active' },
    retry: { label: '等待重试', tone: 'warning' },
    quarantined: { label: '已隔离', tone: 'danger' },
    disabled: { label: '已停用', tone: 'neutral' }
  }[node.state] || { label: '未知状态', tone: 'warning' };
}

export function canTriggerSync(status, node) {
  return Boolean(status?.enabled && status.initialized && !status.primary_recovering
    && ['ready', 'retry'].includes(node.state));
}

export function manualRequestLabel(node) {
  const request = node.manual_request;
  if (!request) return '尚未手动触发';
  if (request.completed_at) return '已完成';
  if (['quarantined', 'disabled', 'new'].includes(node.state)) return '等待人工处理';
  if (['draining', 'syncing'].includes(node.state)) return '执行中';
  if (node.state === 'retry') return '未完成 · 等待重试';
  return '等待同步进程处理';
}

export function formatSyncTime(value) {
  if (!value) return '—';
  // The API returns UTC offsets; accept earlier naive UTC responses during rollout.
  const normalized = /(?:Z|[+-]\d\d:\d\d)$/i.test(value) ? value : `${value.replace(' ', 'T')}Z`;
  const date = new Date(normalized);
  return Number.isNaN(date.getTime()) ? '—' : new Intl.DateTimeFormat('zh-CN', {
    month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false
  }).format(date);
}
