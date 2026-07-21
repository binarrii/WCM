<script setup>
import { ref, onMounted, onUnmounted, computed, watch, nextTick } from 'vue';
import Cropper from 'cropperjs';
import 'cropperjs/dist/cropper.css';
import { 
  UserPlus, 
  Search, 
  Trash2, 
  Edit3, 
  Users, 
  UserCheck, 
  ShieldAlert, 
  UploadCloud, 
  X, 
  Image as ImageIcon,
  CheckCircle, 
  AlertTriangle,
  AlertCircle,
  FileText,
  Briefcase,
  Layers,
  Settings,
  ArrowUp,
  Sun,
  Moon,
  Monitor
} from '@lucide/vue';
import { faceService } from '../services/faceService';
import { IMAGE_BASE } from '../services/api';

// State
const records = ref([]);
const loading = ref(false);
const loadingMore = ref(false);
const submitting = ref(false);
const searchQuery = ref('');
const filterType = ref('All');

// Pagination state
const page = ref(1);
const limit = ref(12); // Load 12 items at a time
const hasMore = ref(true);

// Stats state
const stats = ref({
  total: 0,
  badArtists: 0,
  political: 0,
  officials: 0
});

// Modal state
const isModalOpen = ref(false);
const isEditMode = ref(false);
const editingRecordId = ref(null);

// Form state
const form = ref({
  name: '',
  occupation: '',
  type: '劣迹艺人',
  remarks: '',
  file: null
});
const imagePreviewUrl = ref(null);
const fileInputRef = ref(null);

// Image Search state
const isImageSearchModalOpen = ref(false);
const imageSearchFile = ref(null);
const imageSearchPreviewUrl = ref(null);
const imageSearchFileInputRef = ref(null);
const imageSearching = ref(false);
const isImageSearchActive = ref(false);
const imageSearchResults = ref([]);

// Full image preview state
const isImagePreviewOpen = ref(false);
const previewImageUrl = ref('');
const openImagePreview = (url) => {
  previewImageUrl.value = url;
  isImagePreviewOpen.value = true;
};
const closeImagePreview = () => {
  isImagePreviewOpen.value = false;
  previewImageUrl.value = '';
};

// Toast state
const toasts = ref([]);

// Fetch records from backend (paginated)
const fetchRecords = async (reset = false) => {
  if (reset) {
    page.value = 1;
    records.value = [];
    hasMore.value = true;
  }
  
  if (!hasMore.value || loading.value || loadingMore.value) return;
  
  if (page.value === 1) {
    loading.value = true;
  } else {
    loadingMore.value = true;
  }
  
  try {
    const data = await faceService.getRecords({
      page: page.value,
      limit: limit.value,
      search: searchQuery.value,
      type: filterType.value
    });
    
    records.value = [...records.value, ...data.items];
    hasMore.value = data.has_more;
    page.value += 1;
  } catch (error) {
    showToast('获取人脸记录失败', 'error');
    console.error(error);
  } finally {
    loading.value = false;
    loadingMore.value = false;
  }
};

// Fetch global stats
const fetchStats = async () => {
  try {
    const data = await faceService.getStats();
    stats.value = {
      total: data.total,
      badArtists: data.bad_artists,
      political: data.political,
      officials: data.officials
    };
  } catch (error) {
    console.error('获取统计数据失败', error);
  }
};

// Toast notification helper
const showToast = (message, type = 'success') => {
  const id = Date.now();
  toasts.value.push({ id, message, type });
  setTimeout(() => {
    toasts.value = toasts.value.filter(t => t.id !== id);
  }, 4000);
};

// Filtered records
const filteredRecords = computed(() => {
  if (isImageSearchActive.value && imageSearchResults.value) {
    return imageSearchResults.value
      .filter(item => (1 - (item.distance || 0)) >= 0.9)
      .map(item => {
        // Map file_path to image_url
        let imageUrl = null;
        if (item.file_path) {
          const prefix = "/tmp/wcm";
          if (item.file_path.startsWith(prefix)) {
            imageUrl = "/images" + item.file_path.substring(prefix.length);
          } else {
            imageUrl = item.file_path;
          }
        }
        
        return {
          id: item.id,
          name: item.name,
          image_url: imageUrl,
          created_at: item.created_at,
          person: {
            name: item.person_name,
            occupation: item.occupation,
            type: item.type,
            remarks: item.remarks
          },
          searchDistance: item.distance,
          searchSimilarity: (1 - item.distance).toFixed(2)
        };
      });
  }
  return records.value;
});

// Cropper state
const rawFile = ref(null);
const rawImageSrc = ref('');
const showCropperModal = ref(false);
const cropperInstance = ref(null);
const cropperImageRef = ref(null);

const openCropper = (file) => {
  rawFile.value = file;
  rawImageSrc.value = URL.createObjectURL(file);
  showCropperModal.value = true;
};

const closeCropper = () => {
  showCropperModal.value = false;
  if (cropperInstance.value) {
    cropperInstance.value.destroy();
    cropperInstance.value = null;
  }
  if (rawImageSrc.value) {
    URL.revokeObjectURL(rawImageSrc.value);
    rawImageSrc.value = '';
  }
  rawFile.value = null;
  
  const fileInput = document.getElementById('file-input');
  if (fileInput) fileInput.value = '';
};

const initCropper = () => {
  if (cropperInstance.value) {
    cropperInstance.value.destroy();
  }
  nextTick(() => {
    if (!cropperImageRef.value) return;
    cropperInstance.value = new Cropper(cropperImageRef.value, {
      viewMode: 1,
      dragMode: 'move',
      autoCropArea: 1,
      restore: false,
      guides: true,
      center: true,
      highlight: false,
      cropBoxMovable: true,
      cropBoxResizable: true,
      toggleDragModeOnDblclick: false,
    });
  });
};

const confirmCrop = () => {
  if (!cropperInstance.value) return;
  
  const canvas = cropperInstance.value.getCroppedCanvas({
    maxWidth: 1024,
    maxHeight: 1024,
    imageSmoothingEnabled: true,
    imageSmoothingQuality: 'high',
  });
  
  if (!canvas) {
    showToast('裁剪失败，请重试', 'error');
    return;
  }
  
  canvas.toBlob((blob) => {
    if (!blob) {
      showToast('获取裁剪图片失败', 'error');
      return;
    }
    
    const originalName = rawFile.value?.name || 'cropped.jpg';
    const croppedFile = new File([blob], originalName, { type: 'image/jpeg' });
    
    form.value.file = croppedFile;
    imagePreviewUrl.value = URL.createObjectURL(croppedFile);
    
    closeCropper();
  }, 'image/jpeg', 0.95);
};

// Handle local image file selection for preview
const handleFileChange = (e) => {
  const file = e.target.files[0];
  if (!file) return;
  
  openCropper(file);
};

// Drag and drop handlers
const handleDrop = (e) => {
  const file = e.dataTransfer.files[0];
  if (!file) return;
  
  if (!file.type.startsWith('image/')) {
    showToast('只能上传图片文件', 'error');
    return;
  }
  
  openCropper(file);
};

// Open modal for creating
const openCreateModal = () => {
  isEditMode.value = false;
  editingRecordId.value = null;
  form.value = {
    name: '',
    occupation: '',
    type: '劣迹艺人',
    remarks: '',
    file: null
  };
  imagePreviewUrl.value = null;
  isModalOpen.value = true;
};

// Open modal for editing
const openEditModal = (record) => {
  isEditMode.value = true;
  editingRecordId.value = record.id;
  form.value = {
    name: record.name,
    occupation: record.person?.occupation || '',
    type: record.person?.type || '劣迹艺人',
    remarks: record.person?.remarks || '',
    file: null
  };
  imagePreviewUrl.value = record.image_url ? `${IMAGE_BASE}${record.image_url}` : null;
  isModalOpen.value = true;
};

// Close modal
const closeModal = () => {
  isModalOpen.value = false;
  if (imagePreviewUrl.value && !isEditMode.value) {
    URL.revokeObjectURL(imagePreviewUrl.value);
  }
};

// Submit form (Save / Update)
const handleSubmit = async () => {
  if (!form.value.name.trim()) {
    showToast('请输入姓名', 'error');
    return;
  }
  
  if (!isEditMode.value && !form.value.file) {
    showToast('请上传人脸图片', 'error');
    return;
  }

  submitting.value = true;
  
  try {
    if (isEditMode.value) {
      await faceService.updateRecord(editingRecordId.value, {
        name: form.value.name,
        occupation: form.value.occupation,
        type: form.value.type,
        remarks: form.value.remarks
      });
      showToast('人脸及人物记录更新成功');
      closeModal();
      
      // Update image search results dynamically if active
      if (isImageSearchActive.value && imageSearchResults.value) {
        imageSearchResults.value = imageSearchResults.value.map(item => {
          if (item.id === editingRecordId.value) {
            return {
              ...item,
              name: form.value.name,
              person_name: form.value.name,
              occupation: form.value.occupation,
              type: form.value.type,
              remarks: form.value.remarks
            };
          }
          return item;
        });
      }
      
      // Update local records array in place so the scroll position remains identical
      records.value = records.value.map(item => {
        if (item.id === editingRecordId.value) {
          return {
            ...item,
            name: form.value.name,
            person: {
              ...item.person,
              name: form.value.name,
              occupation: form.value.occupation,
              type: form.value.type,
              remarks: form.value.remarks
            }
          };
        }
        return item;
      });
      
      fetchStats();
    } else {
      const formData = new FormData();
      formData.append('name', form.value.name);
      formData.append('occupation', form.value.occupation);
      formData.append('type', form.value.type);
      formData.append('remarks', form.value.remarks);
      formData.append('file', form.value.file);
      
      await faceService.createRecord(formData);
      
      showToast('人脸及人物档案注册成功！');
      closeModal();
      fetchRecords(true);
      fetchStats();
    }
  } catch (error) {
    const errorMsg = error.response?.data?.detail || '操作失败';
    showToast(errorMsg, 'error');
    console.error(error);
  } finally {
    submitting.value = false;
  }
};

// Delete record
const handleDelete = async (record) => {
  if (!confirm(`确定要删除 ${record.name} 的人脸及记录吗？此操作不可逆。`)) return;
  
  try {
    await faceService.deleteRecord(record.id);
    showToast('记录删除成功');
    
    // Filter out deleted record from active search results
    if (isImageSearchActive.value && imageSearchResults.value) {
      imageSearchResults.value = imageSearchResults.value.filter(r => r.id !== record.id);
    }
    
    fetchRecords(true);
    fetchStats();
  } catch (error) {
    showToast('删除记录失败', 'error');
    console.error(error);
  }
};

// Image Search functions
const openImageSearchModal = () => {
  isImageSearchModalOpen.value = true;
  imageSearchFile.value = null;
  imageSearchPreviewUrl.value = null;
};

const closeImageSearchModal = () => {
  isImageSearchModalOpen.value = false;
  clearImageSearchPreview();
};

const handleImageSearchFileChange = (e) => {
  const file = e.target.files[0];
  if (!file) return;
  imageSearchFile.value = file;
  imageSearchPreviewUrl.value = URL.createObjectURL(file);
};

const handleImageSearchDrop = (e) => {
  const file = e.dataTransfer.files[0];
  if (!file) return;
  if (!file.type.startsWith('image/')) {
    showToast('只能上传图片文件', 'error');
    return;
  }
  imageSearchFile.value = file;
  imageSearchPreviewUrl.value = URL.createObjectURL(file);
};

const clearImageSearchPreview = () => {
  if (imageSearchPreviewUrl.value) {
    URL.revokeObjectURL(imageSearchPreviewUrl.value);
    imageSearchPreviewUrl.value = null;
  }
  imageSearchFile.value = null;
};

const executeImageSearch = async () => {
  if (!imageSearchFile.value) return;
  imageSearching.value = true;
  try {
    const formData = new FormData();
    formData.append('file', imageSearchFile.value);
    formData.append('top_k', '10');
    formData.append('threshold', '0.4');
    
    const data = await faceService.searchFaces(formData);
    imageSearchResults.value = data.results || [];
    isImageSearchActive.value = true;
    const matchCount = imageSearchResults.value.filter(r => (1 - (r.distance || 0)) >= 0.9).length;
    showToast(`检索成功，共找到 ${matchCount} 个相似人脸`);
    closeImageSearchModal();
  } catch (error) {
    const errorMsg = error.response?.data?.detail || '以图搜图失败';
    showToast(errorMsg, 'error');
    console.error(error);
  } finally {
    imageSearching.value = false;
  }
};

const clearImageSearch = () => {
  isImageSearchActive.value = false;
  imageSearchResults.value = [];
  fetchRecords(true);
};

// Watchers for filtering and searching (with debounce)
let searchTimeout = null;
watch(searchQuery, () => {
  if (isImageSearchActive.value) {
    isImageSearchActive.value = false;
    imageSearchResults.value = [];
  }
  if (searchTimeout) clearTimeout(searchTimeout);
  searchTimeout = setTimeout(() => {
    fetchRecords(true);
  }, 400);
});

watch(filterType, () => {
  if (isImageSearchActive.value) {
    isImageSearchActive.value = false;
    imageSearchResults.value = [];
  }
  fetchRecords(true);
});

// State for back to top button
const showBackToTop = ref(false);

const scrollToTop = () => {
  window.scrollTo({
    top: 0,
    behavior: 'smooth'
  });
};

// Scroll listener for infinite scroll (cross-browser compatible)
const handleScroll = () => {
  const scrollTop = window.scrollY || window.pageYOffset || document.documentElement.scrollTop;
  const scrollHeight = document.documentElement.scrollHeight || document.body.scrollHeight;
  const clientHeight = window.innerHeight || document.documentElement.clientHeight;
  
  // Show button when scrolled down more than 300px
  showBackToTop.value = scrollTop > 300;
  
  // Trigger when scrolled to 150px from bottom (disabled during image search)
  if (scrollHeight - scrollTop - clientHeight < 150 && !isImageSearchActive.value) {
    fetchRecords(false);
  }
};

// Theme state & handlers
const currentTheme = ref('system');

const applyTheme = () => {
  const theme = currentTheme.value;
  let activeTheme = theme;
  if (theme === 'system') {
    activeTheme = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }
  document.documentElement.setAttribute('data-theme', activeTheme);
};

const setTheme = (themeName) => {
  currentTheme.value = themeName;
  localStorage.setItem('theme', themeName);
};

watch(currentTheme, applyTheme);

// Handle system theme changes dynamically
const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
const handleSystemThemeChange = () => {
  if (currentTheme.value === 'system') {
    applyTheme();
  }
};

const handleGlobalPaste = (e) => {
  if (!isModalOpen.value && !isImageSearchModalOpen.value) return;
  const items = e.clipboardData?.items;
  if (!items) return;
  
  let imageFile = null;
  for (let i = 0; i < items.length; i++) {
    if (items[i].type.indexOf('image') !== -1) {
      imageFile = items[i].getAsFile();
      break;
    }
  }
  
  if (!imageFile) return;
  
  e.preventDefault();
  
  if (isImageSearchModalOpen.value) {
    imageSearchFile.value = imageFile;
    if (imageSearchPreviewUrl.value) {
      URL.revokeObjectURL(imageSearchPreviewUrl.value);
    }
    imageSearchPreviewUrl.value = URL.createObjectURL(imageFile);
    showToast('已从剪贴板粘贴图片到搜图', 'success');
  } else if (isModalOpen.value) {
    openCropper(imageFile);
    showToast('已从剪贴板粘贴图片并打开裁剪器', 'success');
  }
};

const handleGlobalKeyDown = (e) => {
  if (e.key === 'Escape' || e.keyCode === 27) {
    if (isImagePreviewOpen.value) {
      closeImagePreview();
    } else if (showCropperModal.value) {
      closeCropper();
    } else if (isImageSearchModalOpen.value) {
      closeImageSearchModal();
    } else if (isModalOpen.value) {
      closeModal();
    }
  }
};

onMounted(() => {
  // Load saved theme
  const savedTheme = localStorage.getItem('theme') || 'system';
  currentTheme.value = savedTheme;
  applyTheme();
  
  fetchRecords(true);
  fetchStats();
  window.addEventListener('scroll', handleScroll);
  window.addEventListener('paste', handleGlobalPaste);
  window.addEventListener('keydown', handleGlobalKeyDown);
  mediaQuery.addEventListener('change', handleSystemThemeChange);
});

onUnmounted(() => {
  window.removeEventListener('scroll', handleScroll);
  window.removeEventListener('paste', handleGlobalPaste);
  window.removeEventListener('keydown', handleGlobalKeyDown);
  mediaQuery.removeEventListener('change', handleSystemThemeChange);
});
</script>

<template>
  <div class="app-container">
    <!-- Header -->
    <header class="app-header">
      <div class="logo-area">
        <div class="glowing-orb"></div>
        <div class="brand">
          <span class="brand-text">WCM Core</span>
          <span class="sub-brand">智能内容审核库</span>
        </div>
      </div>
      <div class="header-right">
        <!-- Theme Switcher -->
        <div class="theme-switcher">
          <button 
            v-for="t in ['light', 'dark', 'system']" 
            :key="t"
            @click="setTheme(t)"
            :class="['theme-btn', { active: currentTheme === t }]"
            :title="t === 'light' ? '浅色模式' : t === 'dark' ? '深色模式' : '跟随系统'"
          >
            <Sun v-if="t === 'light'" class="theme-icon" />
            <Moon v-if="t === 'dark'" class="theme-icon" />
            <Monitor v-if="t === 'system'" class="theme-icon" />
          </button>
        </div>
        
        <div class="system-status">
          <span class="status-indicator"></span>
          <span class="status-text">系统服务正常</span>
        </div>
      </div>
    </header>

    <main class="app-main animate-fade-in">
      <!-- Stats Dashboard -->
      <section class="stats-grid">
        <div class="stat-card">
          <div class="stat-icon-wrapper blue">
            <Users class="stat-icon" />
          </div>
          <div class="stat-content">
            <h3 class="stat-label">库总容量</h3>
            <p class="stat-value">{{ stats.total }} <span class="unit">人</span></p>
          </div>
        </div>

        <div class="stat-card">
          <div class="stat-icon-wrapper red">
            <ShieldAlert class="stat-icon" />
          </div>
          <div class="stat-content">
            <h3 class="stat-label">劣迹艺人</h3>
            <p class="stat-value">{{ stats.badArtists }} <span class="unit">人</span></p>
          </div>
        </div>

        <div class="stat-card">
          <div class="stat-icon-wrapper yellow">
            <AlertTriangle class="stat-icon" />
          </div>
          <div class="stat-content">
            <h3 class="stat-label">时政敏感</h3>
            <p class="stat-value">{{ stats.political }} <span class="unit">人</span></p>
          </div>
        </div>

        <div class="stat-card">
          <div class="stat-icon-wrapper green">
            <UserCheck class="stat-icon" />
          </div>
          <div class="stat-content">
            <h3 class="stat-label">落马官员</h3>
            <p class="stat-value">{{ stats.officials }} <span class="unit">人</span></p>
          </div>
        </div>
      </section>

      <!-- Controls Panel -->
      <section class="controls-panel">
        <div class="search-box">
          <Search class="search-icon" />
          <input 
            type="text" 
            v-model="searchQuery" 
            placeholder="搜索姓名或描述信息..." 
            class="search-input"
          />
          <button 
            v-if="searchQuery" 
            class="search-clear-btn" 
            @click="searchQuery = ''"
            title="清除搜索内容"
          >
            <X class="clear-icon" />
          </button>
          <button 
            type="button"
            class="search-image-btn" 
            @click="openImageSearchModal"
            title="以图搜图"
          >
            <ImageIcon class="image-icon" />
          </button>
        </div>

        <div class="filter-actions">
          <div class="filter-group">
            <button 
              v-for="type in ['All', '劣迹艺人', '时政敏感', '落马官员', '其它']" 
              :key="type"
              @click="filterType = type"
              :class="['filter-btn', { active: filterType === type }]"
            >
              {{ type === 'All' ? '全部类别' : type }}
            </button>
          </div>

          <button class="add-btn" @click="openCreateModal">
            <UserPlus class="btn-icon" />
            <span>新增人脸</span>
          </button>
        </div>
      </section>

      <!-- Records Display Grid -->
      <section class="records-section">
        <!-- Image Search Active Banner -->
        <div v-if="isImageSearchActive" class="image-search-banner">
          <div class="banner-info">
            <UserCheck class="banner-icon" />
            <span>以图搜图结果：找到 {{ filteredRecords.length }} 个相似人脸</span>
          </div>
          <button class="clear-image-search-btn" @click="clearImageSearch">
            <X class="btn-icon-small" />
            <span>清除搜图</span>
          </button>
        </div>

        <div v-if="loading" class="loading-state">
          <Settings class="spinner loading-icon" />
          <p>正在加载敏感人脸数据库...</p>
        </div>

        <div v-else-if="filteredRecords.length === 0" class="empty-state">
          <AlertCircle class="empty-icon" />
          <p>暂无符合筛选条件的人脸记录</p>
          <button v-if="searchQuery || filterType !== 'All'" @click="searchQuery = ''; filterType = 'All'" class="reset-filter-btn">
            重置筛选条件
          </button>
        </div>

        <div v-else class="records-grid">
          <div 
            v-for="record in filteredRecords" 
            :key="record.id" 
            class="record-card"
          >
            <!-- Card Image -->
            <div class="card-image-container" @click="record.image_url && openImagePreview(`${IMAGE_BASE}${record.image_url}`)">
              <img 
                v-if="record.image_url" 
                :src="`${IMAGE_BASE}${record.image_url}`" 
                :alt="record.name" 
                class="card-image"
                loading="lazy"
              />
              <div v-else class="card-image-fallback">
                <AlertTriangle class="fallback-icon" />
                <span>暂无预览图片</span>
              </div>
              <!-- Category Badge -->
              <span :class="['category-badge', record.person?.type || '其它']">
                {{ record.person?.type || '其它' }}
              </span>
              <!-- Similarity Badge -->
              <span v-if="record.searchSimilarity" class="similarity-badge">
                相似度: {{ (record.searchSimilarity * 100).toFixed(0) }}%
              </span>
            </div>

            <!-- Card Body -->
            <div class="card-body">
              <div class="card-header-row">
                <h4 class="card-title">{{ record.name }}</h4>
              </div>
              
              <div class="card-info-item remarks-item">
                <FileText class="info-icon" />
                <p class="remarks-text" :title="record.person?.remarks">
                  {{ record.person?.remarks || '暂无描述信息' }}
                </p>
              </div>

              <div class="card-footer-row">
                <span class="date-text">时间: {{ new Date(record.created_at).toLocaleDateString() }}</span>
                <div class="action-buttons">
                  <button class="icon-btn edit" @click="openEditModal(record)" title="编辑信息">
                    <Edit3 class="icon-btn-svg" />
                  </button>
                  <button class="icon-btn delete" @click="handleDelete(record)" title="删除">
                    <Trash2 class="icon-btn-svg" />
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
        <!-- Infinite Scroll Loading States -->
        <div v-if="loadingMore" class="loading-more-state">
          <Settings class="spinner loading-more-icon" />
          <span>正在加载更多人脸数据...</span>
        </div>
        <div v-else-if="!hasMore && filteredRecords.length > 0" class="no-more-state">
          <span>— 已加载全部数据 —</span>
        </div>
      </section>
    </main>

    <!-- Glassmorphic Add/Edit Modal -->
    <div v-if="isModalOpen" class="modal-overlay" @click.self="closeModal">
      <div class="modal-card animate-fade-in">
        <div class="modal-header">
          <h3 class="modal-title">
            {{ isEditMode ? '修改人脸档案' : '新增人脸' }}
          </h3>
          <button class="close-btn" @click="closeModal">
            <X class="close-icon" />
          </button>
        </div>

        <form @submit.prevent="handleSubmit" class="modal-form">
          <!-- Image upload zone -->
          <div class="upload-section">
            <label class="form-label">人脸照片 (只能包含单张人脸，禁止合照及无人脸图)</label>
            
            <div 
              v-if="!imagePreviewUrl" 
              class="upload-dropzone"
              @click="fileInputRef.click()"
              @dragover.prevent
              @drop.prevent="handleDrop"
            >
              <UploadCloud class="upload-icon" />
              <p class="upload-text">点击或拖拽图片文件到此区域上传</p>
              <p class="upload-hint">支持 JPG、PNG 格式，图片大小不超 100MB</p>
            </div>

            <div v-else class="upload-preview-container">
              <img :src="imagePreviewUrl" alt="预览" class="upload-preview" />
              <button 
                v-if="!isEditMode" 
                type="button" 
                class="remove-preview-btn" 
                @click="imagePreviewUrl = null; form.file = null"
              >
                <X class="remove-icon" />
              </button>
            </div>

            <input 
              type="file" 
              ref="fileInputRef" 
              class="hidden-input" 
              accept="image/*" 
              @change="handleFileChange"
            />
          </div>

          <!-- Text Info Fields -->
          <div class="form-fields">
            <div class="form-group">
              <label for="name" class="form-label required">姓名</label>
              <input 
                type="text" 
                id="name" 
                v-model="form.name" 
                class="form-input" 
                placeholder="请输入涉政敏感/劣迹人员真实姓名"
                required
              />
            </div>

            <div class="form-group">
              <label for="type" class="form-label">类别</label>
              <select id="type" v-model="form.type" class="form-select">
                <option value="劣迹艺人">劣迹艺人</option>
                <option value="时政敏感">时政敏感</option>
                <option value="落马官员">落马官员</option>
                <option value="其它">其它</option>
              </select>
            </div>

            <div class="form-group">
              <label for="remarks" class="form-label">违规情况备注</label>
              <textarea 
                id="remarks" 
                v-model="form.remarks" 
                class="form-textarea" 
                placeholder="请输入详细违规备注，以便审核模型定位关键特征..."
                rows="3"
              ></textarea>
            </div>
          </div>

          <!-- Action Row -->
          <div class="modal-actions">
            <button type="button" class="btn-secondary" @click="closeModal" :disabled="submitting">
              取消
            </button>
            <button type="submit" class="btn-primary" :disabled="submitting">
              <Settings v-if="submitting" class="spinner btn-spinner" />
              <span>{{ submitting ? '人脸特征检测与注册中...' : '确认提交' }}</span>
            </button>
          </div>
        </form>
      </div>
    </div>

    <!-- Toast Notifications -->
    <div class="toast-container">
      <div 
        v-for="toast in toasts" 
        :key="toast.id" 
        :class="['toast-card', toast.type]"
      >
        <CheckCircle v-if="toast.type === 'success'" class="toast-icon" />
        <AlertCircle v-else class="toast-icon" />
        <span class="toast-message">{{ toast.message }}</span>
      </div>
    </div>

    <!-- Back to Top Button -->
    <button 
      v-if="showBackToTop" 
      class="back-to-top-btn animate-fade-in" 
      @click="scrollToTop"
      title="返回顶部"
    >
      <ArrowUp class="back-to-top-icon" />
    </button>

    <!-- Image Search Modal -->
    <div v-if="isImageSearchModalOpen" class="modal-overlay" @click.self="closeImageSearchModal">
      <div class="modal-card animate-fade-in">
        <div class="modal-header">
          <h3 class="modal-title">以图搜图</h3>
          <button class="close-btn" @click="closeImageSearchModal">
            <X class="close-icon" />
          </button>
        </div>

        <div class="cropper-content-wrapper">
          <div class="upload-section">
            <label class="form-label">上传待检索人脸照片</label>
            
            <div 
              v-if="!imageSearchPreviewUrl" 
              class="upload-dropzone"
              @click="imageSearchFileInputRef.click()"
              @dragover.prevent
              @drop.prevent="handleImageSearchDrop"
            >
              <UploadCloud class="upload-icon" />
              <p class="upload-text">点击或拖拽图片文件到此区域上传</p>
              <p class="upload-hint">支持 JPG、PNG 格式，系统将检索相似的人员档案</p>
            </div>

            <div v-else class="upload-preview-container">
              <img :src="imageSearchPreviewUrl" alt="待搜索预览" class="upload-preview" />
              <button 
                type="button" 
                class="remove-preview-btn" 
                @click="clearImageSearchPreview"
              >
                <X class="remove-icon" />
              </button>
            </div>

            <input 
              type="file" 
              ref="imageSearchFileInputRef" 
              class="hidden-input" 
              accept="image/*" 
              @change="handleImageSearchFileChange"
            />
          </div>

          <div class="modal-actions" style="margin-top: 24px;">
            <button 
              type="button" 
              class="btn-secondary" 
              @click="closeImageSearchModal"
              :disabled="imageSearching"
            >
              取消
            </button>
            <button 
              type="button" 
              class="btn-primary" 
              @click="executeImageSearch" 
              :disabled="!imageSearchFile || imageSearching"
            >
              <Settings v-if="imageSearching" class="spinner btn-spinner" />
              <span>{{ imageSearching ? '人脸检索中...' : '搜索' }}</span>
            </button>
          </div>
        </div>
      </div>
    </div>

    <!-- Cropper Modal -->
    <div v-if="showCropperModal" class="modal-overlay cropper-overlay" @click.self="closeCropper">
      <div class="modal-card cropper-card animate-fade-in">
        <div class="modal-header">
          <h3 class="modal-title">裁剪人脸图片</h3>
          <button class="close-btn" @click="closeCropper">
            <X class="close-icon" />
          </button>
        </div>
        
        <div class="cropper-content-wrapper">
          <div class="cropper-body">
            <div class="cropper-container">
              <img 
                ref="cropperImageRef" 
                :src="rawImageSrc" 
                class="cropper-raw-image" 
                @load="initCropper"
              />
            </div>
          </div>
          
          <div class="modal-actions">
            <button type="button" class="btn-secondary" @click="closeCropper">
              取消
            </button>
            <button type="button" class="btn-primary cropper-confirm-btn" @click="confirmCrop">
              确认裁剪
            </button>
          </div>
        </div>
      </div>
    </div>

    <!-- Image Preview Modal -->
    <div v-if="isImagePreviewOpen" class="modal-overlay preview-overlay" @click.self="closeImagePreview">
      <div class="preview-close-btn" @click="closeImagePreview" title="关闭">
        <X class="close-icon" />
      </div>
      <img :src="previewImageUrl" alt="完整大图" class="preview-image-large animate-zoom-in" />
    </div>
  </div>
</template>

<style scoped>
/* App Layout Container */
.app-container {
  max-width: 1400px;
  margin: 0 auto;
  padding: 40px 20px;
  display: flex;
  flex-direction: column;
  gap: 30px;
}

/* App Main Content Area */
.app-main {
  display: flex;
  flex-direction: column;
  gap: 24px;
}

/* Header styling */
.app-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding-bottom: 20px;
  border-bottom: 1px solid var(--border-color);
}

.header-right {
  display: flex;
  align-items: center;
  gap: 20px;
}

.theme-switcher {
  display: flex;
  background: var(--bg-primary);
  border: 1px solid var(--border-color);
  border-radius: 20px;
  padding: 3px;
  gap: 2px;
  align-items: center;
}

.theme-btn {
  background: transparent;
  border: none;
  width: 32px;
  height: 32px;
  border-radius: 50%;
  color: var(--text-secondary);
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
}

.theme-btn:hover {
  color: var(--text-primary);
}

.theme-btn.active {
  background: var(--bg-card);
  color: var(--color-primary);
  box-shadow: var(--shadow-sm);
}

.theme-icon {
  width: 16px;
  height: 16px;
}

.logo-area {
  display: flex;
  align-items: center;
  gap: 15px;
  position: relative;
}

.glowing-orb {
  width: 16px;
  height: 16px;
  background-color: var(--color-primary);
  border-radius: 50%;
  box-shadow: 0 0 16px var(--color-primary);
}

.brand {
  display: flex;
  flex-direction: column;
}

.brand-text {
  font-family: var(--font-display);
  font-weight: 700;
  font-size: 1.5rem;
  letter-spacing: -0.5px;
  background: linear-gradient(135deg, var(--brand-gradient-start) 0%, var(--brand-gradient-end) 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.sub-brand {
  font-size: 0.75rem;
  color: var(--text-secondary);
}

.system-status {
  display: flex;
  align-items: center;
  gap: 8px;
  background-color: var(--bg-secondary);
  padding: 6px 12px;
  border-radius: 20px;
  border: 1px solid var(--border-color);
}

.status-indicator {
  width: 8px;
  height: 8px;
  background-color: var(--status-green);
  border-radius: 50%;
  box-shadow: 0 0 8px var(--status-green);
}

.status-text {
  font-size: 0.8rem;
  color: var(--text-secondary);
}

/* Stats dashboard */
.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 20px;
}

.stat-card {
  background-color: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: 16px;
  padding: 24px;
  display: flex;
  align-items: center;
  gap: 20px;
  box-shadow: var(--shadow-sm);
  transition: transform 0.3s ease, border-color 0.3s ease;
}

.stat-card:hover {
  transform: translateY(-2px);
  border-color: var(--border-hover);
}

.stat-icon-wrapper {
  width: 52px;
  height: 52px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.stat-icon-wrapper.blue {
  background-color: var(--status-blue-bg);
  border: 1px solid var(--status-blue-border);
  color: var(--status-blue);
}

.stat-icon-wrapper.red {
  background-color: var(--status-red-bg);
  border: 1px solid var(--status-red-border);
  color: var(--status-red);
}

.stat-icon-wrapper.yellow {
  background-color: var(--status-yellow-bg);
  border: 1px solid var(--status-yellow-border);
  color: var(--status-yellow);
}

.stat-icon-wrapper.green {
  background-color: var(--status-green-bg);
  border: 1px solid var(--status-green-border);
  color: var(--status-green);
}

.stat-icon {
  width: 24px;
  height: 24px;
}

.stat-content {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.stat-label {
  font-size: 0.85rem;
  color: var(--text-secondary);
  font-weight: 500;
}

.stat-value {
  font-family: var(--font-display);
  font-size: 1.8rem;
  font-weight: 700;
}

.stat-value .unit {
  font-size: 0.9rem;
  font-weight: 500;
  color: var(--text-muted);
}

/* Controls styling */
.controls-panel {
  position: sticky;
  top: 16px;
  z-index: 10;
  background-color: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: 16px;
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 20px;
  box-shadow: var(--shadow-md);
}

@media (min-width: 900px) {
  .controls-panel {
    flex-direction: row;
    justify-content: space-between;
    align-items: center;
  }
}

.search-box {
  display: flex;
  align-items: center;
  gap: 12px;
  background-color: var(--bg-primary);
  border: 1px solid var(--border-color);
  border-radius: 12px;
  padding: 10px 16px;
  width: 100%;
  max-width: 480px;
  transition: border-color 0.3s ease, box-shadow 0.3s ease;
}

.search-box:focus-within {
  border-color: var(--color-primary);
  box-shadow: 0 0 0 3px var(--color-primary-glow);
}

.search-icon {
  width: 20px;
  height: 20px;
  color: var(--text-muted);
}

.search-input {
  flex: 1;
  background: transparent;
  border: none;
  color: var(--text-primary);
  font-size: 0.95rem;
  outline: none;
  min-width: 0;
}

.search-clear-btn {
  background: transparent;
  border: none;
  color: var(--text-muted);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 4px;
  border-radius: 50%;
  transition: all 0.2s ease;
  margin-left: 4px;
  flex-shrink: 0;
}

.search-clear-btn:hover {
  background: var(--bg-card);
  color: var(--text-primary);
}

.search-clear-btn .clear-icon {
  width: 16px;
  height: 16px;
}

.search-image-btn {
  background: transparent;
  border: none;
  color: var(--text-muted);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 4px;
  border-radius: 50%;
  transition: all 0.2s ease;
  flex-shrink: 0;
}

.search-image-btn:hover {
  background: var(--bg-card);
  color: var(--color-primary);
}

.search-image-btn .image-icon {
  width: 18px;
  height: 18px;
}

.image-search-banner {
  display: flex;
  justify-content: space-between;
  align-items: center;
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  border-radius: 12px;
  padding: 12px 20px;
  margin-bottom: 20px;
  box-shadow: var(--shadow-sm);
  animation: fadeIn 0.3s ease;
}

.banner-info {
  display: flex;
  align-items: center;
  gap: 10px;
  color: var(--text-primary);
  font-weight: 500;
}

.banner-icon {
  width: 20px;
  height: 20px;
  color: var(--status-green);
}

.clear-image-search-btn {
  display: flex;
  align-items: center;
  gap: 6px;
  background: var(--bg-primary);
  border: 1px solid var(--border-color);
  border-radius: 8px;
  padding: 6px 12px;
  color: var(--text-secondary);
  font-size: 0.85rem;
  cursor: pointer;
  transition: all 0.2s ease;
}

.clear-image-search-btn:hover {
  background: var(--status-red-bg);
  border-color: var(--status-red-border);
  color: var(--status-red);
}

.btn-icon-small {
  width: 14px;
  height: 14px;
}

.similarity-badge {
  position: absolute;
  top: 16px;
  left: 16px;
  background-color: var(--status-green);
  color: white;
  padding: 6px 12px;
  border-radius: 20px;
  font-size: 0.75rem;
  font-weight: 600;
  letter-spacing: 0.5px;
  box-shadow: var(--shadow-sm);
  z-index: 10;
}

.filter-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
  align-items: center;
}

.filter-group {
  display: flex;
  background-color: var(--bg-primary);
  padding: 4px;
  border-radius: 10px;
  border: 1px solid var(--border-color);
}

.filter-btn {
  background: transparent;
  border: none;
  color: var(--text-secondary);
  padding: 8px 16px;
  border-radius: 8px;
  font-size: 0.85rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.25s ease;
}

.filter-btn.active {
  background-color: var(--bg-card);
  color: var(--text-primary);
  box-shadow: var(--shadow-sm);
}

.add-btn {
  background-color: var(--color-primary);
  color: #ffffff;
  border: none;
  border-radius: 12px;
  padding: 12px 20px;
  font-weight: 600;
  font-size: 0.9rem;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 8px;
  transition: background-color 0.25s ease, box-shadow 0.25s ease;
  box-shadow: 0 4px 14px var(--color-primary-glow);
}

.add-btn:hover {
  background-color: var(--color-primary-hover);
  box-shadow: 0 6px 20px rgba(92, 103, 242, 0.4);
}

.btn-icon {
  width: 18px;
  height: 18px;
}

/* Records display and grids */
.records-section {
  min-height: 400px;
}

.loading-state, .empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 16px;
  height: 400px;
  color: var(--text-secondary);
}

.loading-icon {
  width: 48px;
  height: 48px;
  color: var(--color-primary);
}

.empty-icon {
  width: 48px;
  height: 48px;
  color: var(--text-muted);
}

.reset-filter-btn {
  background-color: var(--bg-secondary);
  color: var(--text-primary);
  border: 1px solid var(--border-color);
  padding: 8px 16px;
  border-radius: 8px;
  cursor: pointer;
  margin-top: 10px;
  font-weight: 500;
}

.reset-filter-btn:hover {
  border-color: var(--color-primary);
}

.records-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 24px;
}

.record-card {
  background-color: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: 16px;
  overflow: hidden;
  box-shadow: var(--shadow-sm);
  display: flex;
  flex-direction: column;
  transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
}

.record-card:hover {
  transform: translateY(-4px);
  border-color: var(--border-hover);
  box-shadow: var(--shadow-lg);
}

/* Card images */
.card-image-container {
  height: 240px;
  position: relative;
  background-color: var(--bg-primary);
  overflow: hidden;
  cursor: zoom-in;
}

.card-image {
  width: 100%;
  height: 100%;
  object-fit: cover;
  transition: transform 0.5s ease;
}

.record-card:hover .card-image {
  transform: scale(1.05);
}

.card-image-fallback {
  height: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 12px;
  color: var(--text-muted);
  font-size: 0.85rem;
}

.fallback-icon {
  width: 36px;
  height: 36px;
}

/* Badges */
.category-badge {
  position: absolute;
  top: 16px;
  right: 16px;
  padding: 6px 12px;
  border-radius: 20px;
  font-size: 0.75rem;
  font-weight: 600;
  letter-spacing: 0.5px;
  box-shadow: var(--shadow-sm);
}

.category-badge.劣迹艺人 {
  background-color: var(--status-red-bg);
  border: 1px solid var(--status-red-border);
  color: var(--status-red);
}

.category-badge.时政敏感 {
  background-color: var(--status-yellow-bg);
  border: 1px solid var(--status-yellow-border);
  color: var(--status-yellow);
}

.category-badge.落马官员 {
  background-color: var(--status-green-bg);
  border: 1px solid var(--status-green-border);
  color: var(--status-green);
}

.category-badge.其它 {
  background-color: var(--status-blue-bg);
  border: 1px solid var(--status-blue-border);
  color: var(--status-blue);
}

/* Card body details */
.card-body {
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 14px;
  flex-grow: 1;
}

.card-header-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.card-title {
  font-family: var(--font-display);
  font-size: 1.15rem;
  font-weight: 600;
  color: var(--text-primary);
}

.card-info-item {
  display: flex;
  align-items: center;
  gap: 10px;
  color: var(--text-secondary);
  font-size: 0.85rem;
}

.info-icon {
  width: 16px;
  height: 16px;
  color: var(--text-muted);
}

.info-value {
  color: var(--text-primary);
  font-weight: 500;
}

.remarks-item {
  align-items: flex-start;
  background-color: var(--bg-primary);
  padding: 10px 12px;
  border-radius: 8px;
  border: 1px solid var(--border-color);
}

.remarks-text {
  font-size: 0.8rem;
  color: var(--text-secondary);
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
  line-height: 1.4;
}

/* Card footer actions */
.card-footer-row {
  margin-top: auto;
  padding-top: 14px;
  border-top: 1px solid var(--border-color);
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.date-text {
  font-size: 0.75rem;
  color: var(--text-muted);
}

.action-buttons {
  display: flex;
  gap: 8px;
}

.icon-btn {
  width: 32px;
  height: 32px;
  border-radius: 8px;
  border: 1px solid var(--border-color);
  background-color: var(--bg-primary);
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all 0.25s ease;
}

.icon-btn-svg {
  width: 14px;
  height: 14px;
}

.icon-btn.edit {
  color: var(--text-secondary);
}

.icon-btn.edit:hover {
  border-color: var(--status-blue);
  background-color: var(--status-blue-bg);
  color: var(--status-blue);
}

.icon-btn.delete {
  color: var(--text-secondary);
}

.icon-btn.delete:hover {
  border-color: var(--status-red);
  background-color: var(--status-red-bg);
  color: var(--status-red);
}

/* Modals overlays & glassmorphism */
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  background-color: var(--modal-overlay-bg);
  backdrop-filter: blur(12px);
  z-index: 100;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 20px;
}

.modal-card {
  background-color: var(--modal-card-bg);
  border: 1px solid var(--border-color);
  border-radius: 24px;
  width: 100%;
  max-width: 580px;
  box-shadow: var(--shadow-lg);
  overflow: hidden;
}

.modal-header {
  padding: 24px;
  border-bottom: 1px solid var(--border-color);
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.modal-title {
  font-family: var(--font-display);
  font-size: 1.25rem;
  font-weight: 600;
  color: var(--text-primary);
}

.close-btn {
  background: transparent;
  border: none;
  color: var(--text-muted);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 32px;
  height: 32px;
  border-radius: 50%;
  transition: all 0.25s ease;
}

.close-btn:hover {
  background-color: var(--modal-hover-bg);
  color: var(--text-primary);
}

.close-icon {
  width: 20px;
  height: 20px;
}

.modal-form {
  padding: 24px;
  display: flex;
  flex-direction: column;
  gap: 20px;
}

/* Form Upload fields */
.upload-section {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.upload-dropzone {
  border: 2px dashed var(--modal-dropzone-border);
  border-radius: 16px;
  padding: 30px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 12px;
  cursor: pointer;
  transition: all 0.25s ease;
}

.upload-dropzone:hover {
  border-color: var(--color-primary);
  background-color: var(--color-primary-glow);
}

.upload-icon {
  width: 42px;
  height: 42px;
  color: var(--color-primary);
}

.upload-text {
  font-size: 0.9rem;
  font-weight: 500;
  color: var(--text-primary);
}

.upload-hint {
  font-size: 0.75rem;
  color: var(--text-muted);
}

.upload-preview-container {
  height: 180px;
  border-radius: 16px;
  overflow: hidden;
  position: relative;
  border: 1px solid var(--border-color);
}

.upload-preview {
  width: 100%;
  height: 100%;
  object-fit: contain;
  background-color: var(--bg-primary);
}

.remove-preview-btn {
  position: absolute;
  top: 12px;
  right: 12px;
  background-color: var(--modal-card-bg);
  border: 1px solid var(--border-color);
  width: 32px;
  height: 32px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--text-primary);
  cursor: pointer;
  transition: background-color 0.2s ease;
}

.remove-preview-btn:hover {
  background-color: var(--status-red);
  border-color: var(--status-red);
}

.remove-icon {
  width: 16px;
  height: 16px;
}

.hidden-input {
  display: none;
}

/* Fields design */
.form-fields {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.form-group {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.form-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}

.form-label {
  font-size: 0.8rem;
  font-weight: 600;
  color: var(--text-secondary);
}

.form-label.required::after {
  content: ' *';
  color: var(--status-red);
}

.form-input, .form-select, .form-textarea {
  background-color: var(--bg-primary);
  border: 1px solid var(--border-color);
  color: var(--text-primary);
  border-radius: 10px;
  padding: 10px 14px;
  font-size: 0.9rem;
  outline: none;
  font-family: var(--font-sans);
  transition: border-color 0.25s ease, box-shadow 0.25s ease;
}

.form-input:focus, .form-select:focus, .form-textarea:focus {
  border-color: var(--color-primary);
  box-shadow: 0 0 0 3px var(--color-primary-glow);
}

.form-textarea {
  resize: vertical;
}

/* Actions */
.modal-actions {
  margin-top: 10px;
  display: flex;
  justify-content: flex-end;
  gap: 12px;
}

.btn-primary {
  background-color: var(--color-primary);
  color: #ffffff;
  border: none;
  border-radius: 10px;
  padding: 12px 24px;
  font-weight: 600;
  font-size: 0.9rem;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 8px;
  box-shadow: 0 4px 12px var(--color-primary-glow);
  transition: all 0.25s ease;
}

.btn-primary:hover:not(:disabled) {
  background-color: var(--color-primary-hover);
  box-shadow: 0 6px 18px rgba(92, 103, 242, 0.35);
}

.btn-primary:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.btn-secondary {
  background-color: transparent;
  border: 1px solid var(--border-color);
  color: var(--text-secondary);
  border-radius: 10px;
  padding: 12px 24px;
  font-weight: 600;
  font-size: 0.9rem;
  cursor: pointer;
  transition: all 0.25s ease;
}

.btn-secondary:hover:not(:disabled) {
  background-color: var(--modal-hover-bg);
  color: var(--text-primary);
}

.btn-secondary:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.btn-spinner {
  width: 16px;
  height: 16px;
}

/* Toast Notifications styling */
.toast-container {
  position: fixed;
  bottom: 24px;
  right: 24px;
  display: flex;
  flex-direction: column;
  gap: 10px;
  z-index: 200;
}

.toast-card {
  background-color: var(--bg-card);
  border: 1px solid var(--border-color);
  border-radius: 12px;
  padding: 14px 20px;
  display: flex;
  align-items: center;
  gap: 12px;
  box-shadow: var(--shadow-md);
  min-width: 280px;
  max-width: 420px;
  animation: fadeIn 0.3s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}

.toast-card.success {
  border-left: 4px solid var(--status-green);
}

.toast-card.success .toast-icon {
  color: var(--status-green);
}

.toast-card.error {
  border-left: 4px solid var(--status-red);
}

.toast-card.error .toast-icon {
  color: var(--status-red);
}

.toast-icon {
  width: 20px;
  height: 20px;
  flex-shrink: 0;
}

.toast-message {
  font-size: 0.85rem;
  font-weight: 500;
  color: var(--text-primary);
}

/* Infinite Scroll styling */
.loading-more-state {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  padding: 24px 0;
  color: var(--text-secondary);
  font-size: 0.9rem;
}

.loading-more-icon {
  width: 20px;
  height: 20px;
  color: var(--color-primary);
}

.no-more-state {
  text-align: center;
  padding: 24px 0;
  color: var(--text-muted);
  font-size: 0.85rem;
  letter-spacing: 1px;
}

/* Floating Back to Top Button styling */
.back-to-top-btn {
  position: fixed;
  bottom: 30px;
  right: 30px;
  width: 48px;
  height: 48px;
  border-radius: 50%;
  background: rgba(30, 41, 59, 0.7);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border: 1px solid rgba(255, 255, 255, 0.1);
  color: var(--text-primary);
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
  transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
  z-index: 99;
}

.back-to-top-btn:hover {
  background: var(--color-primary);
  border-color: var(--color-primary);
  transform: translateY(-4px);
  box-shadow: 0 8px 30px rgba(59, 130, 246, 0.4);
}

.back-to-top-icon {
  width: 20px;
  height: 20px;
}

/* Cropper Modal Specific Styles */
.cropper-overlay {
  z-index: 101 !important;
}

.cropper-card {
  max-width: 600px;
  width: 90%;
}

.cropper-body {
  padding: 10px 0;
  display: flex;
  justify-content: center;
  align-items: center;
}

.cropper-container {
  width: 100%;
  max-height: 360px;
  height: 360px;
  background: #000;
  border-radius: 8px;
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
}

.cropper-raw-image {
  max-width: 100%;
  max-height: 100%;
  display: block;
}

.cropper-view-box {
  outline: 2px solid var(--color-primary) !important;
  outline-color: var(--color-primary) !important;
}

.cropper-line {
  background-color: var(--color-primary) !important;
}

.cropper-point {
  background-color: var(--color-primary) !important;
  width: 8px !important;
  height: 8px !important;
}

.cropper-point.point-se {
  width: 12px !important;
  height: 12px !important;
  background-color: var(--color-primary) !important;
  opacity: 1 !important;
}

.cropper-content-wrapper {
  padding: 24px;
  display: flex;
  flex-direction: column;
}

/* Image Preview Modal */
.preview-overlay {
  background-color: var(--preview-overlay-bg) !important;
  z-index: 200;
}

.preview-image-large {
  max-width: 90vw;
  max-height: 85vh;
  object-fit: contain;
  border-radius: 12px;
  box-shadow: var(--shadow-lg);
  border: 1px solid var(--border-color);
  background-color: var(--bg-secondary);
}

.preview-close-btn {
  position: fixed;
  top: 24px;
  right: 24px;
  background-color: var(--preview-close-bg);
  border: 1px solid var(--border-color);
  width: 44px;
  height: 44px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--preview-close-color);
  cursor: pointer;
  z-index: 201;
  transition: all 0.25s ease;
}

.preview-close-btn:hover {
  background-color: var(--status-red);
  border-color: var(--status-red);
  color: #ffffff;
  transform: rotate(90deg);
}

.animate-zoom-in {
  animation: zoomIn 0.3s cubic-bezier(0.34, 1.56, 0.64, 1) forwards;
}

@keyframes zoomIn {
  from {
    opacity: 0;
    transform: scale(0.92);
  }
  to {
    opacity: 1;
    transform: scale(1);
  }
}
</style>
