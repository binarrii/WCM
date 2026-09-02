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
  ChevronLeft,
  ChevronRight,
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
const nextCursor = ref(null);
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
const DEFAULT_IMAGE_SEARCH_SIMILARITY = 0.8;
const IMAGE_SEARCH_SIMILARITY_TICKS = [30, 40, 50, 60, 70, 80, 90, 100];
const imageSearchSimilarityThreshold = ref(DEFAULT_IMAGE_SEARCH_SIMILARITY);
const activeImageSearchSimilarityThreshold = ref(DEFAULT_IMAGE_SEARCH_SIMILARITY);

// Full image preview state
const isImagePreviewOpen = ref(false);
const previewImageUrls = ref([]);
const previewImageIndex = ref(0);
const previewImageUrl = computed(() => previewImageUrls.value[previewImageIndex.value] || '');
const openImagePreview = (urls, initialIndex = 0) => {
  const availableUrls = (Array.isArray(urls) ? urls : [urls]).filter(Boolean);
  if (availableUrls.length === 0) return;
  previewImageUrls.value = availableUrls;
  previewImageIndex.value = Math.min(Math.max(initialIndex, 0), availableUrls.length - 1);
  isImagePreviewOpen.value = true;
};
const closeImagePreview = () => {
  isImagePreviewOpen.value = false;
  previewImageUrls.value = [];
  previewImageIndex.value = 0;
};
const changePreviewImage = (direction) => {
  if (previewImageUrls.value.length < 2) return;
  previewImageIndex.value = (
    previewImageIndex.value + direction + previewImageUrls.value.length
  ) % previewImageUrls.value.length;
};
const activeCardImageIndexes = ref({});
const getRecordImages = (record) => {
  if (Array.isArray(record.image_urls) && record.image_urls.length > 0) {
    return record.image_urls;
  }
  return record.image_url ? [record.image_url] : [];
};
const getActiveCardImageIndex = (record) => {
  const images = getRecordImages(record);
  if (images.length === 0) return 0;
  return Math.min(activeCardImageIndexes.value[record.id] || 0, images.length - 1);
};
const getActiveCardImage = (record) => (
  getRecordImages(record)[getActiveCardImageIndex(record)] || null
);
const changeCardImage = (record, direction) => {
  const images = getRecordImages(record);
  if (images.length < 2) return;
  const current = getActiveCardImageIndex(record);
  activeCardImageIndexes.value[record.id] = (current + direction + images.length) % images.length;
};

// Toast state
const toasts = ref([]);

// Fetch records from backend (paginated)
const fetchRecords = async (reset = false) => {
  if (reset) {
    nextCursor.value = null;
    records.value = [];
    hasMore.value = true;
  }
  
  if (!hasMore.value || loading.value || loadingMore.value) return;
  
  if (records.value.length === 0) {
    loading.value = true;
  } else {
    loadingMore.value = true;
  }
  
  try {
    const data = await faceService.getRecords({
      cursor: nextCursor.value,
      limit: limit.value,
      search: searchQuery.value,
      type: filterType.value
    });
    
    records.value = [...records.value, ...data.items];
    nextCursor.value = data.next_cursor;
    hasMore.value = Boolean(data.has_more && data.next_cursor);
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
const getSearchSimilarity = (item) => Number(
  item.similarity ?? (1 - (item.distance ?? 1))
);

const filteredRecords = computed(() => {
  if (isImageSearchActive.value && imageSearchResults.value) {
    return imageSearchResults.value
      .filter(item => getSearchSimilarity(item) >= activeImageSearchSimilarityThreshold.value)
      .map(item => {
        const similarity = getSearchSimilarity(item);
        return {
          id: item.id,
          name: item.name,
          image_url: item.image_url,
          created_at: item.created_at,
          person: {
            name: item.name,
            occupation: item.occupation,
            type: item.type,
            remarks: item.remarks
          },
          searchDistance: item.distance,
          searchSimilarity: similarity.toFixed(2)
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
      formData.append('category', form.value.type);
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
    formData.append('threshold', String(1 - imageSearchSimilarityThreshold.value));
    
    const data = await faceService.searchFaces(formData);
    imageSearchResults.value = data.results || [];
    activeImageSearchSimilarityThreshold.value = imageSearchSimilarityThreshold.value;
    isImageSearchActive.value = true;
    const matchCount = imageSearchResults.value.filter(
      item => getSearchSimilarity(item) >= activeImageSearchSimilarityThreshold.value
    ).length;
    showToast(`检索成功，共找到 ${matchCount} 个相似人脸`);
    closeImageSearchModal();
  } catch (error) {
    const errorMsg = error.response?.data?.detail || '人脸检索失败';
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
  if (isImagePreviewOpen.value && e.key === 'ArrowLeft') {
    e.preventDefault();
    changePreviewImage(-1);
    return;
  }
  if (isImagePreviewOpen.value && e.key === 'ArrowRight') {
    e.preventDefault();
    changePreviewImage(1);
    return;
  }
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
            title="人脸检索"
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
            <span>人脸检索结果：找到 {{ filteredRecords.length }} 个相似人脸</span>
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
            <div class="card-image-container" @click="getActiveCardImage(record) && openImagePreview(getRecordImages(record).map(url => `${IMAGE_BASE}${url}`), getActiveCardImageIndex(record))">
              <img 
                v-if="getActiveCardImage(record)" 
                :src="`${IMAGE_BASE}${getActiveCardImage(record)}`" 
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
              <template v-if="getRecordImages(record).length > 1">
                <button class="image-nav-button image-nav-previous" type="button" title="上一张" @click.stop="changeCardImage(record, -1)">
                  <ChevronLeft />
                </button>
                <button class="image-nav-button image-nav-next" type="button" title="下一张" @click.stop="changeCardImage(record, 1)">
                  <ChevronRight />
                </button>
                <span class="image-count-badge">
                  {{ getActiveCardImageIndex(record) + 1 }} / {{ getRecordImages(record).length }}
                </span>
              </template>
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
          <h3 class="modal-title">人脸检索</h3>
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

          <div class="similarity-threshold-field">
            <div class="similarity-threshold-header">
              <label for="image-search-threshold" class="form-label">最低相似度</label>
              <output for="image-search-threshold" class="similarity-threshold-value">
                {{ Math.round(imageSearchSimilarityThreshold * 100) }}%
              </output>
            </div>
            <div
              class="similarity-threshold-control"
              :style="{
                '--threshold-progress': `${((imageSearchSimilarityThreshold - 0.3) / 0.7) * 100}%`
              }"
            >
              <div class="similarity-threshold-track" aria-hidden="true">
                <span
                  v-for="tick in IMAGE_SEARCH_SIMILARITY_TICKS"
                  :key="tick"
                  :class="['similarity-threshold-tick', { active: tick <= imageSearchSimilarityThreshold * 100 }]"
                  :style="{ left: `${((tick - 30) / 70) * 100}%` }"
                ></span>
              </div>
              <input
                id="image-search-threshold"
                v-model.number="imageSearchSimilarityThreshold"
                class="similarity-threshold-range"
                type="range"
                min="0.3"
                max="1"
                step="0.01"
              />
            </div>
            <div class="similarity-threshold-scale" aria-hidden="true">
              <span v-for="tick in IMAGE_SEARCH_SIMILARITY_TICKS" :key="tick">{{ tick }}%</span>
            </div>
            <p class="similarity-threshold-hint">默认 80%，数值越高，检索结果越严格</p>
          </div>

          <div class="modal-actions image-search-actions">
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
      <button
        v-if="previewImageUrls.length > 1"
        type="button"
        class="preview-nav-button preview-nav-previous"
        title="上一张"
        aria-label="上一张完整大图"
        @click.stop="changePreviewImage(-1)"
      >
        <ChevronLeft />
      </button>
      <img :key="previewImageUrl" :src="previewImageUrl" alt="完整大图" class="preview-image-large animate-zoom-in" />
      <button
        v-if="previewImageUrls.length > 1"
        type="button"
        class="preview-nav-button preview-nav-next"
        title="下一张"
        aria-label="下一张完整大图"
        @click.stop="changePreviewImage(1)"
      >
        <ChevronRight />
      </button>
      <span v-if="previewImageUrls.length > 1" class="preview-image-count">
        {{ previewImageIndex + 1 }} / {{ previewImageUrls.length }}
      </span>
    </div>
  </div>
</template>

<style scoped src="./face-dashboard.css"></style>
