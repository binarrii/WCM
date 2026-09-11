export function formatParameterValue(value, type, indent = 2) {
  if (type === 'json') return JSON.stringify(value, null, indent);
  if (type === 'number' || type === 'boolean' || type === 'enum') return String(value);
  return value ?? '';
}

export function summarizeParameterValue(value, type, maxLength = 120) {
  const formatted = formatParameterValue(value, type, 0).replace(/\s+/g, ' ').trim();
  return formatted.length > maxLength ? `${formatted.slice(0, maxLength)}…` : formatted;
}

function escapeHtml(value) {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;');
}

export function highlightJson(value) {
  const escaped = escapeHtml(JSON.stringify(value, null, 2));
  const tokenPattern = /"(?:\\u[\da-fA-F]{4}|\\[^u]|[^\\"])*"|-?\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b|\b(?:true|false|null)\b/g;
  return escaped.replace(tokenPattern, (token, offset, source) => {
    let kind = 'number';
    if (token.startsWith('"')) {
      kind = source.slice(offset + token.length).trimStart().startsWith(':') ? 'key' : 'string';
    } else if (token === 'true' || token === 'false') {
      kind = 'boolean';
    } else if (token === 'null') {
      kind = 'null';
    }
    return `<span class="json-${kind}">${token}</span>`;
  });
}

export function parseParameterValue(text, type) {
  if (type === 'string') return text;
  let value;
  try {
    value = JSON.parse(text);
  } catch {
    if (type === 'number') throw new Error('请输入有效数字');
    if (type === 'boolean') throw new Error('布尔值必须是 true 或 false');
    throw new Error('请输入有效 JSON');
  }
  if (type === 'number' && (typeof value !== 'number' || !Number.isFinite(value))) {
    throw new Error('请输入有效数字');
  }
  if (type === 'boolean' && typeof value !== 'boolean') {
    throw new Error('布尔值必须是 true 或 false');
  }
  return value;
}

export function parseEnumOptions(text) {
  let options;
  try {
    options = JSON.parse(text);
  } catch {
    throw new Error('枚举可选值必须是有效的 JSON 数组');
  }
  if (!Array.isArray(options) || options.length === 0) {
    throw new Error('枚举类型必须提供非空的可选值列表');
  }
  const identities = new Set();
  for (const option of options) {
    const kind = typeof option;
    if ((kind !== 'string' && kind !== 'number') || (kind === 'number' && !Number.isFinite(option))) {
      throw new Error('枚举可选值只能是 string 或 number');
    }
    const identity = `${kind}:${String(option)}`;
    if (identities.has(identity)) throw new Error('枚举可选值不能重复');
    identities.add(identity);
  }
  return options;
}
