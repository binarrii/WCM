<script setup>
import { AlertTriangle, X } from '@lucide/vue';
import { nextTick, onBeforeUnmount, ref, watch } from 'vue';

const props = defineProps({
  open: { type: Boolean, default: false },
  title: { type: String, default: '确认操作' },
  message: { type: String, default: '' },
  confirmLabel: { type: String, default: '确认' },
  cancelLabel: { type: String, default: '取消' },
  variant: {
    type: String,
    default: 'danger',
    validator: value => ['danger', 'primary'].includes(value)
  },
  busy: { type: Boolean, default: false }
});

const emit = defineEmits(['confirm', 'cancel', 'update:open']);
const closeButton = ref(null);
const cancelButton = ref(null);
const confirmButton = ref(null);
let previousFocus = null;
let previousOverflow = '';

const restorePage = () => {
  if (typeof document === 'undefined') return;
  document.body.style.overflow = previousOverflow;
  previousFocus?.focus?.();
  previousFocus = null;
};

watch(() => props.open, async isOpen => {
  if (!isOpen || typeof document === 'undefined') {
    restorePage();
    return;
  }
  previousFocus = document.activeElement;
  previousOverflow = document.body.style.overflow;
  document.body.style.overflow = 'hidden';
  await nextTick();
  cancelButton.value?.focus();
});

onBeforeUnmount(restorePage);

const cancel = () => {
  if (props.busy) return;
  emit('update:open', false);
  emit('cancel');
};

const confirm = () => {
  if (!props.busy) emit('confirm');
};

const handleKeydown = event => {
  if (event.key === 'Escape') {
    event.preventDefault();
    cancel();
    return;
  }
  if (event.key !== 'Tab') return;

  const focusable = [closeButton.value, cancelButton.value, confirmButton.value].filter(Boolean);
  if (!focusable.length) return;
  event.preventDefault();
  const currentIndex = focusable.indexOf(document.activeElement);
  const nextIndex = event.shiftKey
    ? (currentIndex <= 0 ? focusable.length - 1 : currentIndex - 1)
    : (currentIndex + 1) % focusable.length;
  focusable[nextIndex].focus();
};
</script>

<template>
  <Teleport to="body">
    <Transition name="confirm-dialog">
      <div
        v-if="open"
        class="confirm-dialog-overlay"
        role="presentation"
        @click.self="cancel"
        @keydown="handleKeydown"
      >
        <section
          class="confirm-dialog-card"
          role="alertdialog"
          aria-modal="true"
          aria-labelledby="confirm-dialog-title"
          aria-describedby="confirm-dialog-message"
        >
          <button ref="closeButton" class="confirm-dialog-close" type="button" :disabled="busy" aria-label="关闭确认对话框" @click="cancel">
            <X />
          </button>
          <div :class="['confirm-dialog-icon', variant]" aria-hidden="true">
            <AlertTriangle />
          </div>
          <div class="confirm-dialog-copy">
            <h2 id="confirm-dialog-title">{{ title }}</h2>
            <p id="confirm-dialog-message">{{ message }}</p>
          </div>
          <div v-if="$slots.details" class="confirm-dialog-details">
            <slot name="details" />
          </div>
          <div class="confirm-dialog-actions">
            <button ref="cancelButton" class="confirm-dialog-cancel" type="button" :disabled="busy" @click="cancel">
              {{ cancelLabel }}
            </button>
            <button ref="confirmButton" :class="['confirm-dialog-confirm', variant]" type="button" :disabled="busy" @click="confirm">
              <span v-if="busy" class="confirm-dialog-spinner" aria-hidden="true"></span>
              {{ busy ? '处理中…' : confirmLabel }}
            </button>
          </div>
        </section>
      </div>
    </Transition>
  </Teleport>
</template>

<style scoped>
.confirm-dialog-overlay {
  position: fixed;
  inset: 0;
  z-index: 3000;
  display: grid;
  place-items: center;
  padding: 24px;
  background: var(--modal-overlay-bg);
  backdrop-filter: blur(10px);
}

.confirm-dialog-card {
  position: relative;
  width: min(440px, 100%);
  padding: 28px;
  color: var(--text-primary);
  background: var(--modal-card-bg);
  border: 1px solid var(--border-color);
  border-radius: 20px;
  box-shadow: var(--shadow-lg);
}

.confirm-dialog-close {
  position: absolute;
  top: 16px;
  right: 16px;
  width: 32px;
  height: 32px;
  display: grid;
  place-items: center;
  color: var(--text-muted);
  background: transparent;
  border: 0;
  border-radius: 9px;
  cursor: pointer;
}

.confirm-dialog-close:hover { color: var(--text-primary); background: var(--modal-hover-bg); }
.confirm-dialog-close svg { width: 18px; }

.confirm-dialog-icon {
  width: 48px;
  height: 48px;
  display: grid;
  place-items: center;
  margin-bottom: 18px;
  border: 1px solid;
  border-radius: 15px;
}

.confirm-dialog-icon.danger { color: var(--status-red); background: var(--status-red-bg); border-color: var(--status-red-border); }
.confirm-dialog-icon.primary { color: var(--color-primary); background: var(--color-primary-glow); border-color: var(--border-hover); }
.confirm-dialog-icon svg { width: 23px; }

.confirm-dialog-copy { padding-right: 24px; }
.confirm-dialog-copy h2 { margin: 0 0 8px; font-family: var(--font-display); font-size: 1.25rem; line-height: 1.3; }
.confirm-dialog-copy p { margin: 0; color: var(--text-secondary); font-size: .86rem; line-height: 1.65; }

.confirm-dialog-details {
  display: grid;
  gap: 3px;
  margin-top: 18px;
  padding: 12px 14px;
  color: var(--text-secondary);
  background: var(--modal-hover-bg);
  border: 1px solid var(--border-color);
  border-radius: 11px;
  font-size: .72rem;
  overflow-wrap: anywhere;
}

.confirm-dialog-details :deep(strong) { color: var(--text-primary); font-size: .76rem; }
.confirm-dialog-details :deep(span) { color: var(--text-muted); }

.confirm-dialog-actions {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
  margin-top: 24px;
}

.confirm-dialog-actions button {
  min-width: 96px;
  height: 42px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 0 18px;
  border-radius: 11px;
  font: 600 .82rem var(--font-sans);
  cursor: pointer;
  transition: transform .18s ease, border-color .18s ease, background-color .18s ease, box-shadow .18s ease;
}

.confirm-dialog-actions button:hover:not(:disabled) { transform: translateY(-1px); }
.confirm-dialog-actions button:focus-visible { outline: 3px solid var(--color-primary-glow); outline-offset: 2px; }
.confirm-dialog-actions button:disabled { opacity: .65; cursor: wait; }
.confirm-dialog-cancel { color: var(--text-secondary); background: var(--bg-card); border: 1px solid var(--border-color); }
.confirm-dialog-cancel:hover:not(:disabled) { color: var(--text-primary); border-color: var(--border-hover); }
.confirm-dialog-confirm { color: #fff; border: 1px solid transparent; }
.confirm-dialog-confirm.danger { background: var(--status-red); box-shadow: 0 8px 20px rgba(239, 68, 68, .22); }
.confirm-dialog-confirm.primary { background: var(--color-primary); box-shadow: 0 8px 20px var(--color-primary-glow); }

.confirm-dialog-spinner {
  width: 14px;
  height: 14px;
  border: 2px solid rgba(255, 255, 255, .45);
  border-top-color: #fff;
  border-radius: 50%;
  animation: confirm-dialog-spin .7s linear infinite;
}

.confirm-dialog-enter-active, .confirm-dialog-leave-active { transition: opacity .18s ease; }
.confirm-dialog-enter-active .confirm-dialog-card, .confirm-dialog-leave-active .confirm-dialog-card { transition: transform .22s ease, opacity .18s ease; }
.confirm-dialog-enter-from, .confirm-dialog-leave-to { opacity: 0; }
.confirm-dialog-enter-from .confirm-dialog-card, .confirm-dialog-leave-to .confirm-dialog-card { opacity: 0; transform: translateY(10px) scale(.97); }

@keyframes confirm-dialog-spin { to { transform: rotate(360deg); } }

@media (max-width: 520px) {
  .confirm-dialog-overlay { padding: 16px; }
  .confirm-dialog-card { padding: 24px 20px 20px; border-radius: 17px; }
  .confirm-dialog-actions { display: grid; grid-template-columns: 1fr 1fr; }
  .confirm-dialog-actions button { min-width: 0; }
}
</style>
