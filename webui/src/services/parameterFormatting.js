export function formatParameterValue(value, type, indent = 2) {
  if (type === 'json') return JSON.stringify(value, null, indent);
  if (type === 'number') return String(value);
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
    throw new Error(type === 'number' ? '请输入有效数字' : '请输入有效 JSON');
  }
  if (type === 'number' && (typeof value !== 'number' || !Number.isFinite(value))) {
    throw new Error('请输入有效数字');
  }
  return value;
}
