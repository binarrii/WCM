<script setup>
import { nextTick, onBeforeUnmount, ref, watch } from 'vue';
import { KeyRound, X } from '@lucide/vue';

const props = defineProps({ open: Boolean });
const emit = defineEmits(['confirm', 'cancel']);
const dialog = ref(null);
const password = ref(''); const confirmation = ref(''); const error = ref('');
let previousFocus;
function clear() { password.value = ''; confirmation.value = ''; error.value = ''; }
function cancel() { clear(); emit('cancel'); }
function submit() {
  if (password.value !== confirmation.value) { error.value = '两次输入的新密码不一致'; return; }
  emit('confirm', password.value);
}
watch(() => props.open, async open => {
  clear();
  if (!open) { dialog.value?.close(); previousFocus?.focus?.(); return; }
  previousFocus = document.activeElement;
  await nextTick();
  dialog.value?.showModal();
  dialog.value?.querySelector('input')?.focus();
});
onBeforeUnmount(clear);
</script>

<template>
  <Teleport to="body">
    <dialog ref="dialog" class="reauth-dialog password-dialog" aria-labelledby="password-dialog-title" aria-describedby="password-dialog-description" @cancel.prevent="cancel">
      <button class="reauth-close" type="button" aria-label="关闭修改密码对话框" @click="cancel"><X /></button>
      <div class="reauth-icon"><KeyRound /></div>
      <h2 id="password-dialog-title">修改登录密码</h2>
      <p id="password-dialog-description" class="reauth-description">使用至少 12 个字符的长密码。确认后需验证身份，修改成功后其他设备需要重新登录。</p>
      <form class="auth-form" @submit.prevent="submit"><fieldset>
        <label>新密码<input v-model="password" type="password" autocomplete="new-password" required minlength="12" maxlength="128" /></label>
        <label>确认新密码<input v-model="confirmation" type="password" autocomplete="new-password" required minlength="12" maxlength="128" /></label>
        <p v-if="error" class="auth-error" role="alert">{{ error }}</p>
        <button class="auth-primary" type="submit">验证身份并修改密码</button>
      </fieldset></form>
      <button class="auth-link reauth-cancel" type="button" @click="cancel">取消</button>
    </dialog>
  </Teleport>
</template>
