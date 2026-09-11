<script setup>
import { computed, onMounted, ref, watch } from 'vue';
import {
  AlertCircle,
  CheckCircle2,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  DatabaseZap,
  Edit3,
  Plus,
  RefreshCw,
  Search,
  Trash2,
  X
} from '@lucide/vue';
import ConfirmDialog from '../components/ConfirmDialog.vue';
import {
  formatParameterValue,
  highlightJson,
  parseParameterValue,
  summarizeParameterValue
} from '../services/parameterFormatting';
import { parameterService } from '../services/parameterService';
import './parameter-config.css';

const parameters = ref([]);
const loading = ref(false);
const saving = ref(false);
const deleting = ref(false);
const error = ref('');
const notice = ref('');
const query = ref('');
const selectedGroup = ref('');
const page = ref(1);
const pageSize = 30;
const expandedKeys = ref(new Set());
const editingKey = ref(null);
const deleteTarget = ref(null);
const formError = ref('');
const form = ref({ key: '', valueText: '', type: 'string', group: 'default', builtIn: false, secret: false });

const groups = computed(() => [...new Set(parameters.value.map(item => item.group))].sort());
const filteredParameters = computed(() => {
  const term = query.value.trim().toLocaleLowerCase();
  return parameters.value.filter(item => {
    if (selectedGroup.value && item.group !== selectedGroup.value) return false;
    const searchableValue = item.secret ? '' : summarizeParameterValue(item.value, item.type, 500);
    return !term || `${item.key} ${item.group} ${searchableValue}`
      .toLocaleLowerCase()
      .includes(term);
  });
});
const pageCount = computed(() => Math.max(1, Math.ceil(filteredParameters.value.length / pageSize)));
const visibleParameters = computed(() => {
  const start = (page.value - 1) * pageSize;
  return filteredParameters.value.slice(start, start + pageSize);
});
const visibleStart = computed(() => filteredParameters.value.length ? (page.value - 1) * pageSize + 1 : 0);
const visibleEnd = computed(() => Math.min(page.value * pageSize, filteredParameters.value.length));

const sortParameters = items => [...items].sort((left, right) => (
  left.group.localeCompare(right.group, 'zh-CN') || left.key.localeCompare(right.key, 'en')
));
const requestError = (reason, fallback) => reason.response?.data?.detail || reason.message || fallback;

const loadParameters = async () => {
  loading.value = true;
  error.value = '';
  try {
    const payload = await parameterService.list();
    parameters.value = sortParameters(payload.items);
  } catch (reason) {
    error.value = requestError(reason, '参数配置加载失败');
  } finally {
    loading.value = false;
  }
};

const toggleExpanded = key => {
  const next = new Set(expandedKeys.value);
  if (next.has(key)) next.delete(key);
  else next.add(key);
  expandedKeys.value = next;
};

const changePage = value => {
  page.value = Math.min(pageCount.value, Math.max(1, value));
};

const openCreate = () => {
  editingKey.value = '';
  form.value = {
    key: '',
    valueText: '',
    type: 'string',
    group: selectedGroup.value || 'default',
    builtIn: false,
    secret: false
  };
  formError.value = '';
};

const openEdit = item => {
  editingKey.value = item.key;
  form.value = {
    key: item.key,
    valueText: item.secret ? '' : formatParameterValue(item.value, item.type),
    type: item.type,
    group: item.group,
    builtIn: Boolean(item.built_in),
    secret: Boolean(item.secret)
  };
  formError.value = '';
};

const closeEditor = () => {
  if (!saving.value) editingKey.value = null;
};

const saveParameter = async () => {
  formError.value = '';
  const key = form.value.key.trim();
  const group = form.value.group.trim();
  if (!/^[A-Za-z][A-Za-z0-9_.-]*$/.test(key)) {
    formError.value = 'key 必须以字母开头，且只能包含字母、数字、点、横线和下划线';
    return;
  }
  if (!group) {
    formError.value = '参数分组不能为空';
    return;
  }
  let value;
  try {
    value = parseParameterValue(form.value.valueText, form.value.type);
  } catch (reason) {
    formError.value = reason.message;
    return;
  }

  saving.value = true;
  error.value = '';
  notice.value = '';
  try {
    const payload = { value, type: form.value.type, group };
    const saved = editingKey.value
      ? await parameterService.update(editingKey.value, payload)
      : await parameterService.create({ key, ...payload });
    parameters.value = sortParameters([
      ...parameters.value.filter(item => item.key !== saved.key),
      saved
    ]);
    const savedIndex = filteredParameters.value.findIndex(item => item.key === saved.key);
    if (savedIndex >= 0) page.value = Math.floor(savedIndex / pageSize) + 1;
    expandedKeys.value = new Set([...expandedKeys.value, saved.key]);
    notice.value = editingKey.value ? `参数 ${saved.key} 已更新并刷新内存` : `参数 ${saved.key} 已创建并载入内存`;
    editingKey.value = null;
  } catch (reason) {
    formError.value = requestError(reason, '参数保存失败');
  } finally {
    saving.value = false;
  }
};

const confirmDelete = async () => {
  if (!deleteTarget.value) return;
  const target = deleteTarget.value;
  deleting.value = true;
  error.value = '';
  notice.value = '';
  try {
    await parameterService.delete(target.key);
    parameters.value = parameters.value.filter(item => item.key !== target.key);
    if (page.value > pageCount.value) page.value = pageCount.value;
    const next = new Set(expandedKeys.value);
    next.delete(target.key);
    expandedKeys.value = next;
    notice.value = `参数 ${target.key} 已删除`;
    deleteTarget.value = null;
  } catch (reason) {
    error.value = requestError(reason, '参数删除失败');
  } finally {
    deleting.value = false;
  }
};

watch([query, selectedGroup], () => { page.value = 1; });
watch(pageCount, count => { if (page.value > count) page.value = count; });
onMounted(loadParameters);
</script>

<template>
  <main class="parameter-config animate-fade-in">
    <section class="parameter-toolbar">
      <form class="parameter-search" @submit.prevent>
        <Search />
        <input v-model="query" type="search" placeholder="检索 key、分组或值" aria-label="检索参数配置" />
      </form>
      <label class="parameter-group-filter">
        <span>参数分组</span>
        <select v-model="selectedGroup">
          <option value="">全部分组</option>
          <option v-for="group in groups" :key="group" :value="group">{{ group }}</option>
        </select>
      </label>
      <button class="parameter-refresh" type="button" :disabled="loading" @click="loadParameters">
        <RefreshCw :class="{ spinner: loading }" />刷新
      </button>
      <button class="parameter-create" type="button" @click="openCreate"><Plus />新增参数</button>
    </section>

    <p v-if="error" class="parameter-message error" role="alert"><AlertCircle />{{ error }}</p>
    <p v-if="notice" class="parameter-message notice" role="status"><CheckCircle2 />{{ notice }}</p>

    <section class="parameter-table-card">
      <header>
        <div><h2>通用参数</h2><span>显示 {{ visibleStart }}-{{ visibleEnd }} / {{ filteredParameters.length }} 条</span></div>
        <small>点击行可展开查看完整值</small>
      </header>
      <div class="parameter-table-scroll">
        <table>
          <thead><tr><th aria-label="展开状态"></th><th>Key</th><th>分组</th><th>类型</th><th>值</th><th aria-label="操作"></th></tr></thead>
          <tbody>
            <template v-for="item in visibleParameters" :key="item.key">
              <tr class="parameter-row" tabindex="0" :aria-expanded="expandedKeys.has(item.key)" @click="toggleExpanded(item.key)" @keydown.enter="toggleExpanded(item.key)">
                <td class="parameter-expand"><ChevronDown v-if="expandedKeys.has(item.key)" /><ChevronRight v-else /></td>
                <td class="parameter-key"><strong>{{ item.key }}</strong></td>
                <td><span class="parameter-group">{{ item.group }}</span></td>
                <td><span :class="['parameter-type', item.type]">{{ item.type }}</span></td>
                <td v-if="item.secret" class="parameter-preview parameter-secret">{{ item.has_value ? '••••••••（已配置）' : '（未配置）' }}</td>
                <td v-else class="parameter-preview" :title="summarizeParameterValue(item.value, item.type, 500)">{{ summarizeParameterValue(item.value, item.type) || '（空字符串）' }}</td>
                <td class="parameter-actions" @click.stop @keydown.enter.stop>
                  <button type="button" title="编辑参数" @click="openEdit(item)"><Edit3 /></button>
                  <button class="danger" type="button" :disabled="item.built_in" :title="item.built_in ? '内置业务参数不能删除' : '删除参数'" @click="deleteTarget = item"><Trash2 /></button>
                </td>
              </tr>
              <tr v-if="expandedKeys.has(item.key)" class="parameter-detail-row">
                <td colspan="6">
                  <div class="parameter-detail-header"><span>{{ item.key }}</span><small>{{ item.type === 'json' ? '格式化 JSON' : '完整值' }}</small></div>
                  <div v-if="item.secret" class="secret-value">敏感值不会通过接口或页面回显</div>
                  <pre v-else-if="item.type === 'json'" class="json-viewer" v-html="highlightJson(item.value)"></pre>
                  <pre v-else class="plain-value">{{ formatParameterValue(item.value, item.type) }}</pre>
                </td>
              </tr>
            </template>
          </tbody>
        </table>
        <div v-if="loading && !parameters.length" class="parameter-empty"><RefreshCw class="spinner" /><p>正在加载参数配置…</p></div>
        <div v-else-if="!visibleParameters.length" class="parameter-empty"><DatabaseZap /><p>{{ parameters.length ? '没有匹配的参数' : '暂无参数配置' }}</p><small>{{ parameters.length ? '调整检索条件后重试' : '点击“新增参数”创建第一条配置' }}</small></div>
      </div>
      <footer class="parameter-pagination">
        <span>第 {{ page }} / {{ pageCount }} 页</span>
        <button type="button" :disabled="page <= 1 || loading" @click="changePage(page - 1)"><ChevronLeft />上一页</button>
        <button type="button" :disabled="page >= pageCount || loading" @click="changePage(page + 1)">下一页<ChevronRight /></button>
      </footer>
    </section>

    <Teleport to="body">
      <Transition name="parameter-modal">
        <div v-if="editingKey !== null" class="parameter-modal-overlay" @click.self="closeEditor">
          <form class="parameter-modal" @submit.prevent="saveParameter">
            <header><div><h2>{{ editingKey ? '编辑参数' : '新增参数' }}</h2><p>保存后会立即更新数据库与内存快照</p></div><button type="button" :disabled="saving" aria-label="关闭" @click="closeEditor"><X /></button></header>
            <label><span>Key</span><input v-model="form.key" :disabled="Boolean(editingKey) || saving" maxlength="191" placeholder="例如 review.max_retries" autocomplete="off" /></label>
            <div class="parameter-form-grid">
              <label><span>类型</span><select v-model="form.type" :disabled="saving || form.builtIn"><option value="string">string</option><option value="number">number</option><option value="json">json</option></select></label>
              <label><span>分组</span><input v-model="form.group" :disabled="saving || form.builtIn" maxlength="100" placeholder="default" autocomplete="off" /></label>
            </div>
            <label><span>{{ form.secret ? '新密钥' : '值' }}</span><textarea v-model="form.valueText" :disabled="saving" rows="12" :placeholder="form.secret ? '敏感值不会回显；输入新值后保存，留空会清空当前密钥' : form.type === 'json' ? '{\n  &quot;enabled&quot;: true\n}' : form.type === 'number' ? '10' : '请输入字符串'"></textarea></label>
            <p v-if="formError" class="parameter-form-error" role="alert"><AlertCircle />{{ formError }}</p>
            <footer><button class="secondary" type="button" :disabled="saving" @click="closeEditor">取消</button><button class="primary" type="submit" :disabled="saving">{{ saving ? '保存中…' : '保存参数' }}</button></footer>
          </form>
        </div>
      </Transition>
    </Teleport>

    <ConfirmDialog
      :open="Boolean(deleteTarget)"
      title="删除参数配置"
      :message="deleteTarget ? `确定删除 ${deleteTarget.key}？删除后会同步刷新内存快照。` : ''"
      confirm-label="删除参数"
      :busy="deleting"
      @confirm="confirmDelete"
      @cancel="deleteTarget = null"
      @update:open="value => { if (!value && !deleting) deleteTarget = null; }"
    />
  </main>
</template>
