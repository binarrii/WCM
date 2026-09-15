<script setup>
import { onMounted, ref } from 'vue';
import { Fingerprint, KeyRound, ShieldCheck } from '@lucide/vue';
import api from '../services/api';
import { auth, roleNames, acceptSession, authError } from '../services/auth';
import { createPasskey, passkeyAvailable } from '../services/passkeys';
import ReauthForm from '../components/ReauthForm.vue';
import ConfirmDialog from '../components/ConfirmDialog.vue';
import './auth.css';

const security = ref({ passkeys: [], totp_enabled: false, recovery_codes_remaining: 0 });
const setup = ref(null); const code = ref(''); const recoveryCodes = ref([]);
const passkeyName = ref('我的设备'); const passkeySupported = passkeyAvailable();
const busy = ref(false); const error = ref(''); const notice = ref(''); const pendingKey = ref(null);
const action = ref(''); const password = ref(''); const factorCode = ref('');
const newPassword = ref(''); const confirmPassword = ref('');
const load = async () => { security.value = (await api.get('/auth/security')).data; };
async function perform(operation, message = '') {
  busy.value = true; error.value = ''; notice.value = '';
  try { await operation(); await load(); notice.value = message; }
  catch (reason) { error.value = authError(reason); }
  finally { busy.value = false; }
}
async function beginSetup() {
  recoveryCodes.value = []; code.value = '';
  await perform(async () => { setup.value = (await api.post('/auth/2fa/setup')).data; });
}
async function confirmSetup() {
  const pending = setup.value;
  await perform(async () => {
    try {
      const { data } = await api.post('/auth/2fa/confirm', { challenge_id: pending.challenge_id, code: code.value });
      acceptSession(data); recoveryCodes.value = data.recovery_codes;
    } finally { setup.value = null; code.value = ''; }
  }, '验证器已绑定。请妥善保存恢复码。');
}
async function bindPasskey() {
  await perform(async () => {
    const { data } = await api.post('/auth/passkeys/register/options');
    const credential = await createPasskey(data.options);
    await api.post('/auth/passkeys/register/verify', { challenge_id: data.challenge_id, credential, name: passkeyName.value || '我的设备' });
  }, 'Passkey 已绑定，下次可使用指纹、面容或设备 PIN 登录。');
}
async function removePasskey() {
  await perform(async () => { await api.delete(`/auth/passkeys/${pendingKey.value.id}`); pendingKey.value = null; }, 'Passkey 已移除，其他设备的登录已退出。');
}
function chooseAction(next) { action.value = next; password.value = ''; factorCode.value = ''; newPassword.value = ''; confirmPassword.value = ''; error.value = ''; }
async function sensitiveAction() {
  if (action.value === 'password' && newPassword.value !== confirmPassword.value) { error.value = '两次输入的新密码不一致'; return; }
  const selected = action.value;
  await perform(async () => {
    const payload = { password: password.value, code: factorCode.value };
    if (selected === 'password') payload.new_password = newPassword.value;
    const { data } = await api.post(`/auth/${selected === 'password' ? 'password' : `2fa/${selected}`}`, payload);
    if (data.user) acceptSession(data);
    recoveryCodes.value = data.recovery_codes || [];
    chooseAction('');
  }, selected === 'password' ? '密码已更新，其他登录已退出。' : selected === 'disable' ? '双重验证已关闭，其他登录已退出。' : '恢复码已更新，旧恢复码已失效。');
  password.value = ''; factorCode.value = ''; newPassword.value = ''; confirmPassword.value = '';
}
function downloadCodes() {
  const url = URL.createObjectURL(new Blob([`WCM ${auth.user.username} 一次性恢复码\n每个恢复码仅能使用一次，请离线妥善保存。\n\n${recoveryCodes.value.join('\n')}\n`], { type: 'text/plain;charset=utf-8' }));
  const anchor = document.createElement('a'); anchor.href = url; anchor.download = 'wcm-recovery-codes.txt'; anchor.click(); URL.revokeObjectURL(url);
}
onMounted(() => perform(load));
</script>

<template>
  <main class="account-page">
    <div class="account-summary"><div class="account-avatar">{{ auth.user.display_name.slice(0, 1) }}</div><div><h2>{{ auth.user.display_name }}</h2><p>@{{ auth.user.username }} <span class="role-badge">{{ roleNames[auth.user.role] }}</span></p></div></div>
    <ReauthForm @verified="load" />
    <p v-if="error" class="auth-error" role="alert">{{ error }}</p><p v-if="notice" class="auth-success" role="status">{{ notice }}</p>
    <section v-if="recoveryCodes.length" class="security-card recovery-panel">
      <h2>保存一次性恢复码</h2><p>验证器不可用时，用恢复码完成登录。每个码只能使用一次，关闭后无法再次查看。</p>
      <div class="recovery-codes"><code v-for="item in recoveryCodes" :key="item">{{ item }}</code></div>
      <div class="security-actions"><button class="auth-primary" @click="downloadCodes">下载恢复码</button><button class="auth-secondary" @click="recoveryCodes = []">已保存，关闭</button></div>
    </section>
    <div class="security-grid">
      <section class="security-card">
        <div class="security-title"><ShieldCheck /><h2>双重验证 · 2FA</h2><span :class="['security-state', { enabled: security.totp_enabled }]">{{ security.totp_enabled ? '已启用' : '未启用' }}</span></div>
        <p>绑定 Google Authenticator、Microsoft Authenticator 等验证器。使用密码登录时，需要额外输入验证码。</p>
        <template v-if="!security.totp_enabled">
          <button v-if="!setup" class="auth-primary" :disabled="busy" @click="beginSetup">绑定验证器</button>
          <form v-else class="auth-form" @submit.prevent="confirmSetup"><fieldset :disabled="busy">
            <p>1. 在验证器中扫描二维码，或手动输入密钥。</p><img class="totp-qr" :src="setup.qr_code" alt="验证器绑定二维码" /><code class="totp-secret">{{ setup.secret }}</code>
            <label>2. 输入 6 位验证码<input v-model="code" autocomplete="one-time-code" inputmode="numeric" pattern="[0-9]{6}" required maxlength="6" /></label>
            <div class="security-actions"><button class="auth-primary" type="submit">验证并启用</button><button type="button" class="auth-secondary" @click="setup = null">取消</button></div>
          </fieldset></form>
        </template>
        <template v-else><p>剩余恢复码：{{ security.recovery_codes_remaining }} 个</p><div class="security-actions"><button class="auth-secondary" :disabled="busy" @click="chooseAction('recovery-codes')">重新生成恢复码</button><button class="auth-secondary danger" :disabled="busy" @click="chooseAction('disable')">关闭 2FA</button></div></template>
      </section>
      <section class="security-card">
        <div class="security-title"><Fingerprint /><h2>Passkey</h2><span class="security-state">{{ security.passkeys.length }} 个</span></div>
        <p>使用指纹、面容或设备 PIN 登录。Passkey 已包含设备持有与身份验证，登录时无需再输入动态验证码。</p>
        <p v-if="!passkeySupported" class="auth-note">当前为非安全连接，Passkey 需要通过 HTTPS 域名访问后绑定和使用。</p>
        <ul class="passkey-list"><li v-for="key in security.passkeys" :key="key.id"><div><strong>{{ key.name }}</strong><small>{{ new Date(key.created_at * 1000).toLocaleDateString('zh-CN') }}</small></div><button class="auth-link danger" :disabled="busy" @click="pendingKey = key">移除</button></li></ul>
        <form class="auth-form" @submit.prevent="bindPasskey"><fieldset :disabled="busy || !passkeySupported"><label>设备名称<input v-model="passkeyName" maxlength="80" required placeholder="例如：办公电脑" /></label><button class="auth-secondary" type="submit">添加 Passkey</button></fieldset></form>
      </section>
      <section class="security-card"><div class="security-title"><KeyRound /><h2>登录密码</h2></div><p>使用至少 12 个字符的长密码。修改后其他设备需要重新登录。</p><button class="auth-secondary" :disabled="busy" @click="chooseAction('password')">修改密码</button></section>
    </div>
    <section v-if="action" class="security-card sensitive-panel">
      <h2>{{ action === 'password' ? '修改登录密码' : action === 'disable' ? '确认关闭双重验证' : '重新生成恢复码' }}</h2>
      <p v-if="action === 'disable'">关闭后，密码登录将不再要求验证器验证码。</p><p v-if="action === 'recovery-codes'">生成后，所有旧恢复码立即失效。</p>
      <form class="auth-form" @submit.prevent="sensitiveAction"><fieldset :disabled="busy"><label>当前密码<input v-model="password" type="password" autocomplete="current-password" required maxlength="128" /></label><label v-if="security.totp_enabled">验证码或恢复码<input v-model="factorCode" autocomplete="one-time-code" required maxlength="64" /></label><template v-if="action === 'password'"><label>新密码<input v-model="newPassword" type="password" autocomplete="new-password" required minlength="12" maxlength="128" /></label><label>确认新密码<input v-model="confirmPassword" type="password" autocomplete="new-password" required minlength="12" maxlength="128" /></label></template><div class="security-actions"><button class="auth-primary" type="submit">{{ busy ? '处理中…' : '确认' }}</button><button class="auth-secondary" type="button" @click="chooseAction('')">取消</button></div></fieldset></form>
    </section>
    <ConfirmDialog :open="Boolean(pendingKey)" title="移除 Passkey" :message="`移除后将无法使用 ${pendingKey?.name || ''} 登录，其他设备的会话也会退出。`" :busy="busy" @confirm="removePasskey" @cancel="pendingKey = null" />
  </main>
</template>
