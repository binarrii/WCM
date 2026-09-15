<script setup>
import { computed, nextTick, onBeforeUnmount, ref, watch } from 'vue';
import { Fingerprint, KeyRound, ShieldCheck, X } from '@lucide/vue';
import api from '../services/api';
import { acceptSession, authError } from '../services/auth';
import { passkeyAvailable, usePasskey } from '../services/passkeys';
import { finishReauthentication, reauthMethods, reauthState } from '../services/reauth';

const dialog = ref(null);
const methods = ref([]); const method = ref('');
const password = ref(''); const code = ref(''); const error = ref('');
const loading = ref(false); const busy = ref(false);
const labels = { passkey: 'Passkey', totp: '2FA 验证码', password: '登录密码' };
const icons = { passkey: Fingerprint, totp: ShieldCheck, password: KeyRound };
const alternatives = computed(() => methods.value.filter(value => value !== method.value));
let controller;
let previousFocus;

async function focusInput() {
  await nextTick();
  dialog.value?.querySelector('input, button[type="submit"]')?.focus();
}
async function loadMethods() {
  const active = controller;
  loading.value = true; error.value = '';
  try {
    const { data } = await api.get('/auth/security', { signal: active.signal });
    if (active.signal.aborted) return;
    methods.value = reauthMethods(data, passkeyAvailable());
    method.value = methods.value[0];
  } catch (reason) {
    if (!active.signal.aborted) error.value = authError(reason);
  } finally {
    if (!active.signal.aborted) { loading.value = false; await focusInput(); }
  }
}
function cancel() {
  controller?.abort();
  password.value = ''; code.value = '';
  finishReauthentication(false);
}
function selectMethod(value) {
  method.value = value; password.value = ''; code.value = ''; error.value = '';
  focusInput();
}
async function submit() {
  if (busy.value || loading.value) return;
  const active = controller;
  busy.value = true; error.value = '';
  try {
    let result;
    const config = { signal: active.signal };
    if (method.value === 'passkey') {
      const { data } = await api.post('/auth/passkeys/reauthenticate/options', {}, config);
      const credential = await usePasskey(data.options, active.signal);
      result = await api.post('/auth/passkeys/reauthenticate/verify', { challenge_id: data.challenge_id, credential }, config);
    } else {
      result = await api.post('/auth/reauthenticate', method.value === 'totp'
        ? { method: 'totp', code: code.value }
        : { method: 'password', password: password.value }, config);
    }
    if (active.signal.aborted) return;
    acceptSession(result.data);
    finishReauthentication(true);
  } catch (reason) {
    if (!active.signal.aborted) error.value = authError(reason);
  } finally {
    if (controller === active) {
      busy.value = false; password.value = ''; code.value = '';
      if (reauthState.open && error.value && !active.signal.aborted) await focusInput();
    }
  }
}
watch(() => reauthState.open, async open => {
  if (!open) {
    controller?.abort(); dialog.value?.close(); previousFocus?.focus?.();
    return;
  }
  controller = new AbortController();
  previousFocus = document.activeElement;
  methods.value = []; method.value = ''; password.value = ''; code.value = ''; busy.value = false;
  await nextTick();
  dialog.value?.showModal();
  await loadMethods();
}, { flush: 'post' });
onBeforeUnmount(cancel);
</script>

<template>
  <Teleport to="body">
    <dialog ref="dialog" class="reauth-dialog" aria-labelledby="reauth-title" aria-describedby="reauth-description" @cancel.prevent="cancel">
      <button class="reauth-close" type="button" aria-label="取消身份验证" @click="cancel"><X /></button>
      <div class="reauth-icon"><component :is="icons[method] || ShieldCheck" /></div>
      <h2 id="reauth-title">验证身份</h2>
      <p id="reauth-description" class="reauth-description">为保障账户安全，请先完成验证。验证成功后继续当前操作，5 分钟内无需重复验证。</p>
      <p v-if="loading" class="auth-muted" role="status">正在加载验证方式…</p>
      <form v-else-if="method" class="auth-form" @submit.prevent="submit"><fieldset :disabled="busy">
        <p v-if="method === 'passkey'" class="reauth-method-description">使用此账户的 Passkey，通过指纹、面容或设备 PIN 验证。</p>
        <label v-else-if="method === 'totp'">2FA 验证码<input v-model="code" autocomplete="one-time-code" autofocus required minlength="6" maxlength="64" placeholder="输入验证码或一次性恢复码" /><small class="auth-hint">打开已绑定的验证器，输入当前验证码；也可使用一次性恢复码。</small></label>
        <label v-else>当前密码<input v-model="password" type="password" autocomplete="current-password" autofocus required maxlength="128" placeholder="输入当前登录密码" /></label>
        <p v-if="error" class="auth-error" role="alert">{{ error }}</p>
        <button class="auth-primary" type="submit">{{ busy ? '验证中…' : method === 'passkey' ? '使用 Passkey 验证' : '验证并继续' }}</button>
        <div v-if="alternatives.length" class="reauth-alternatives"><span>其他验证方式</span><button v-for="value in alternatives" :key="value" class="auth-link" type="button" @click="selectMethod(value)"><component :is="icons[value]" />使用{{ labels[value] }}</button></div>
      </fieldset></form>
      <template v-else-if="error"><p class="auth-error" role="alert">{{ error }}</p><button class="auth-secondary" @click="loadMethods">重新加载</button></template>
      <button class="auth-link reauth-cancel" type="button" @click="cancel">取消操作</button>
    </dialog>
  </Teleport>
</template>
