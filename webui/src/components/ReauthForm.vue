<script setup>
import { ref } from 'vue';
import api from '../services/api';
import { auth, acceptSession, authError } from '../services/auth';
const emit = defineEmits(['verified']);
const password = ref(''); const code = ref(''); const busy = ref(false); const error = ref(''); const verified = ref(false);
async function submit() {
  busy.value = true; error.value = ''; verified.value = false;
  try { acceptSession((await api.post('/auth/reauthenticate', { password: password.value, code: code.value })).data); verified.value = true; emit('verified'); }
  catch (reason) { error.value = authError(reason); }
  finally { busy.value = false; password.value = ''; code.value = ''; }
}
</script>
<template>
  <details class="auth-reauth">
    <summary>重新验证身份 <span>敏感操作前验证，有效期 5 分钟</span></summary>
    <form class="auth-form" @submit.prevent="submit"><fieldset :disabled="busy">
      <label>当前密码<input v-model="password" type="password" autocomplete="current-password" required maxlength="128" /></label>
      <label v-if="auth.user?.totp_enabled">验证码或恢复码<input v-model="code" autocomplete="one-time-code" required maxlength="64" /></label>
      <p v-if="error" class="auth-error" role="alert">{{ error }}</p><p v-if="verified" class="auth-success" role="status">身份已验证，可以继续操作。</p>
      <button class="auth-secondary" type="submit">{{ busy ? '验证中…' : '验证身份' }}</button>
    </fieldset></form>
  </details>
</template>
