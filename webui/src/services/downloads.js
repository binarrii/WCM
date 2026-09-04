export const saveBlob = (blob, filename) => {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 0);
};

export const saveJson = (payload, filename = 'analysis.json') => {
  saveBlob(
    new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' }),
    filename
  );
};
