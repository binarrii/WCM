<script setup>
import { computed, onMounted, ref } from 'vue';
import api from '../services/api';
import { roleNames, authError } from '../services/auth';
import ReauthForm from '../components/ReauthForm.vue';
import ConfirmDialog from '../components/ConfirmDialog.vue';
import './auth.css';
const users = ref([]); const total = ref(0); const page = ref(1); const query = ref('');
const policy = ref(null); const busy = ref(false); const error = ref(''); const notice = ref(''); const pending = ref(null);
const pageCount = computed(() => Math.max(1, Math.ceil(total.value / 30)));
const load = async () => {
  const [userResult, rolesResult] = await Promise.all([api.get('/auth/users', { params: { page: page.value, query: query.value } }), api.get('/auth/roles')]);
  users.value = userResult.data.items; total.value = userResult.data.total; policy.value = rolesResult.data;
};
async function perform(operation, message = '') {
  busy.value = true; error.value = ''; notice.value = '';
  try { await operation(); notice.value = message; }
  catch (reason) { error.value = authError(reason); }
  finally { busy.value = false; }
}
async function search() { page.value = 1; await perform(load); }
async function changePage(value) { page.value = value; await perform(load); }
function confirmRole(user, event) {
  const role = event.target.value; event.target.value = user.role;
  if (role !== user.role) pending.value = { title: '修改用户角色', message: `将 ${user.display_name} 设置为${roleNames[role]}。该用户需要重新登录。`, path: `/auth/users/${user.id}/role`, payload: { role } };
}
function confirmActive(user) {
  pending.value = { title: user.active ? '停用用户' : '启用用户', message: `${user.active ? '停用后将立即退出该用户的所有登录，并禁止继续登录。' : '启用后该用户可以重新登录。'} 用户：${user.username}`, path: `/auth/users/${user.id}/active`, payload: { active: !user.active } };
}
function confirmPolicy(role) {
  pending.value = { title: '保存角色权限', message: `${roleNames[role]}的操作权限将立即更新。`, path: `/auth/roles/${role}`, payload: { permissions: [...policy.value.roles[role]] } };
}
async function save() {
  await perform(async () => { await api.put(pending.value.path, pending.value.payload); pending.value = null; await load(); }, '修改已保存。');
}
onMounted(() => perform(load));
</script>
<template>
  <main class="account-page users-page">
    <ReauthForm />
    <p v-if="error" class="auth-error" role="alert">{{ error }}</p><p v-if="notice" class="auth-success" role="status">{{ notice }}</p>
    <section class="security-card">
      <div class="users-toolbar"><div><h2>用户账户 <span class="security-state">{{ total }} 人</span></h2><p>用户自行注册；管理员角色仅由超级管理员授予。</p></div><form class="user-search" @submit.prevent="search"><input v-model="query" maxlength="64" placeholder="搜索用户名" aria-label="搜索用户名" /><button class="auth-secondary" :disabled="busy">搜索</button></form></div>
      <div class="users-table-scroll"><table class="users-table"><thead><tr><th>用户</th><th>角色</th><th>双重验证</th><th>状态</th><th>注册时间</th><th>操作</th></tr></thead><tbody><tr v-for="user in users" :key="user.id"><td><strong>{{ user.display_name }}</strong><small>@{{ user.username }}</small></td><td><span v-if="user.role === 'superadmin'" class="role-badge">超级管理员</span><select v-else :value="user.role" :disabled="busy" :aria-label="`${user.username} 的角色`" @change="confirmRole(user, $event)"><option value="user">普通用户</option><option value="admin">普通管理员</option></select></td><td>{{ user.totp_enabled ? '已绑定' : '未绑定' }}</td><td>{{ user.active ? '正常' : '已停用' }}</td><td>{{ new Date(user.created_at * 1000).toLocaleDateString('zh-CN') }}</td><td><button v-if="user.role !== 'superadmin'" class="auth-link" :class="{ danger: user.active }" :disabled="busy" @click="confirmActive(user)">{{ user.active ? '停用' : '启用' }}</button><span v-else class="auth-muted">受保护</span></td></tr></tbody></table><p v-if="!users.length" class="auth-muted">{{ busy ? '正在加载…' : '没有匹配的用户' }}</p></div>
      <div class="users-pagination"><button class="auth-secondary" :disabled="busy || page <= 1" @click="changePage(page - 1)">上一页</button><span>{{ page }} / {{ pageCount }}</span><button class="auth-secondary" :disabled="busy || page >= pageCount" @click="changePage(page + 1)">下一页</button></div>
    </section>
    <section v-if="policy" class="security-card"><h2>按角色分配操作权限</h2><p>超级管理员始终拥有全部权限。系统管理与参数配置只允许管理员角色；用户管理始终仅限超级管理员。</p><div class="role-permissions"><fieldset v-for="role in ['user', 'admin']" :key="role" :disabled="busy"><legend>{{ roleNames[role] }}</legend><label v-for="(label, key) in policy.permissions" :key="key" :class="{ 'permission-locked': role === 'user' && policy.admin_only.includes(key) }"><input v-model="policy.roles[role]" type="checkbox" :value="key" :disabled="role === 'user' && policy.admin_only.includes(key)" />{{ label }}<small v-if="role === 'user' && policy.admin_only.includes(key)">仅管理员</small></label><button class="auth-secondary" @click="confirmPolicy(role)">保存{{ roleNames[role] }}权限</button></fieldset></div><p class="auth-hint">人物与审核记录在团队内共享，是否可查看或操作由上方权限决定。授予维护或审核提交权限时，会同时授予对应查看权限。</p></section>
    <ConfirmDialog :open="Boolean(pending)" :title="pending?.title || ''" :message="pending?.message || ''" :busy="busy" variant="primary" confirm-label="确认修改" @confirm="save" @cancel="pending = null" />
  </main>
</template>
