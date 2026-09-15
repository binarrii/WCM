<script setup>
import { onMounted, ref } from 'vue';
import { Fingerprint, ShieldCheck } from '@lucide/vue';
import api from '../services/api';
import { acceptSession, authError } from '../services/auth';
import { passkeyAvailable, usePasskey } from '../services/passkeys';
import './auth.css';

const mode = ref('login');
const username = ref('');
const displayName = ref('');
const password = ref('');
const confirmPassword = ref('');
const code = ref('');
const challenge = ref('');
const busy = ref(false);
const error = ref('');
const firstRegistration = ref(false);
const passkeySupported = passkeyAvailable();
const switchMode = next => { mode.value = next; error.value = ''; password.value = ''; confirmPassword.value = ''; challenge.value = ''; code.value = ''; };
async function submit() {
  if (mode.value === 'register' && password.value !== confirmPassword.value) { error.value = '两次输入的密码不一致'; return; }
  busy.value = true; error.value = '';
  try {
    const path = challenge.value ? '/auth/login/2fa' : `/auth/${mode.value}`;
    const payload = challenge.value ? { challenge_id: challenge.value, code: code.value } : {
      username: username.value.trim(), password: password.value,
      ...(mode.value === 'register' ? { display_name: displayName.value.trim() } : {})
    };
    const { data } = await api.post(path, payload);
    password.value = ''; confirmPassword.value = ''; code.value = '';
    if (data.mfa_required) challenge.value = data.challenge_id;
    else acceptSession(data);
  } catch (reason) {
    error.value = authError(reason);
    if (challenge.value) { challenge.value = ''; code.value = ''; }
  } finally { busy.value = false; }
}
async function passkeyLogin() {
  busy.value = true; error.value = '';
  try {
    const { data } = await api.post('/auth/passkeys/login/options');
    const credential = await usePasskey(data.options);
    acceptSession((await api.post('/auth/passkeys/login/verify', { challenge_id: data.challenge_id, credential })).data);
  } catch (reason) { error.value = authError(reason); }
  finally { busy.value = false; }
}
onMounted(async () => {
  try { firstRegistration.value = (await api.get('/auth/config')).data.first_registration; }
  catch (reason) { error.value = authError(reason); }
});
</script>

<template>
  <main class="auth-screen">
    <section class="auth-intro">
      <div class="auth-brand"><span class="brand-orb"></span> WCM Core</div>
      <span class="auth-eyebrow">智能内容审核工作台</span>
      <h1>每一次审核，<br>从可信身份开始。</h1>
      <p>人物识别、视频审核与团队协作，<br>在统一的工作空间中完成。</p>
      <div class="auth-trust"><ShieldCheck /><span>密码 · Passkey · 双重验证</span></div>
    </section>
    <section class="auth-card" aria-label="账户登录与注册">
      <div class="auth-tabs" v-if="!challenge"><button :class="{ active: mode === 'login' }" :disabled="busy" @click="switchMode('login')">登录</button><button :class="{ active: mode === 'register' }" :disabled="busy" @click="switchMode('register')">创建账户</button></div>
      <h2>{{ challenge ? '双重验证' : mode === 'register' ? '创建你的 WCM 账户' : '欢迎回来' }}</h2>
      <p class="auth-muted">{{ challenge ? '输入验证器中的 6 位验证码，或使用一次性恢复码。' : mode === 'register' ? '注册后即可进入工作台。' : '登录以继续使用内容审核工作台。' }}</p>
      <p v-if="mode === 'register'" class="auth-note">{{ firstRegistration ? '系统尚无用户，首个成功注册的账户将成为超级管理员。' : '新注册账户默认为普通用户，管理员角色由超级管理员授予。' }}</p>
      <form @submit.prevent="submit" class="auth-form">
        <fieldset :disabled="busy">
          <template v-if="!challenge">
            <label>用户名<input v-model="username" name="username" autocomplete="username" required minlength="3" maxlength="64" pattern="[A-Za-z0-9][A-Za-z0-9_.@-]*" placeholder="字母、数字、下划线或邮箱格式" /></label>
            <label v-if="mode === 'register'">显示名称<input v-model="displayName" name="display_name" autocomplete="nickname" required maxlength="80" placeholder="团队中显示的名字" /></label>
            <label>密码<input v-model="password" name="password" type="password" :autocomplete="mode === 'register' ? 'new-password' : 'current-password'" required :minlength="mode === 'register' ? 12 : 1" maxlength="128" :placeholder="mode === 'register' ? '至少 12 个字符，建议使用长密码' : '输入密码'" /></label>
            <label v-if="mode === 'register'">确认密码<input v-model="confirmPassword" type="password" autocomplete="new-password" required minlength="12" maxlength="128" placeholder="再次输入密码" /></label>
          </template>
          <label v-else>验证码或恢复码<input v-model="code" name="code" autocomplete="one-time-code" required minlength="6" maxlength="64" autofocus placeholder="6 位验证码 / 恢复码" /></label>
          <p v-if="error" class="auth-error" role="alert">{{ error }}</p>
          <button type="submit" class="auth-primary">{{ busy ? '正在验证…' : challenge ? '验证并登录' : mode === 'register' ? '创建账户' : '登录' }}</button>
        </fieldset>
      </form>
      <template v-if="mode === 'login' && !challenge">
        <div class="auth-divider">或</div>
        <button class="auth-secondary auth-full" :disabled="busy || !passkeySupported" @click="passkeyLogin"><Fingerprint />使用 Passkey 登录</button>
        <p class="auth-hint">{{ passkeySupported ? '使用已绑定的指纹、面容或安全密钥。' : '当前连接不支持 Passkey，请使用 HTTPS 域名访问。' }}</p>
      </template>
      <button v-if="challenge" class="auth-link" :disabled="busy" @click="switchMode('login')">返回登录</button>
    </section>
  </main>
</template>
