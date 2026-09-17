<script setup>
import { ArrowUpRight, LockKeyhole } from '@lucide/vue';
import { can } from '../services/auth';
import peopleArtwork from '../assets/home/people.svg';
import videoArtwork from '../assets/home/video.svg';
import tasksArtwork from '../assets/home/tasks.svg';
import mediaArtwork from '../assets/home/media.svg';
import './home.css';

const entries = [
  { route: 'people', title: '人物库管理', label: 'PEOPLE', description: '维护人物档案，让每一次识别有据可依。', artwork: peopleArtwork, permission: 'people.read' },
  { route: 'video', title: '视频审核', label: 'REVIEW', description: '识别视频人物与违规内容，定位关键画面。', artwork: videoArtwork, permission: 'review.read' },
  { route: 'tasks', title: '审核任务', label: 'TASKS', description: '追踪审核进度，回看结果与复核记录。', artwork: tasksArtwork, permission: 'review.read' },
  { route: 'media', title: '媒资库', label: 'MEDIA', description: '为每一份素材，准备一个新空间。', artwork: mediaArtwork, upcoming: true }
];
const accessible = entry => !entry.permission || can(entry.permission);
</script>

<template>
  <main class="home-page" aria-labelledby="home-title">
    <div class="home-inner">
      <header class="home-intro">
        <div>
          <p class="home-eyebrow"><span></span> WCM CORE <span class="home-eyebrow-divider">/</span> YOUR WORKSPACE</p>
          <h2 id="home-title">看清每一帧，<span>掌握每一步。</span></h2>
          <p class="home-description">人物识别、内容审核与媒体管理，从这里有序展开。</p>
        </div>
        <span class="home-intro-mark" aria-hidden="true">一站协同 <span>·</span> 从容审核</span>
      </header>

      <nav class="home-gallery" aria-label="工作台快捷入口">
        <a v-for="(entry, index) in entries" :key="entry.route"
          :class="['home-card', `home-card--${entry.route}`, { 'home-card--unavailable': !accessible(entry) }]"
          :href="accessible(entry) ? `#/${entry.route}` : undefined"
          :aria-disabled="!accessible(entry) || undefined"
          :tabindex="accessible(entry) ? undefined : -1"
          :aria-label="`${entry.title}${accessible(entry) ? '' : '，暂无访问权限'}`">
          <div class="home-card-top"><span class="home-card-index">0{{ index + 1 }}</span><span>{{ entry.label }}</span><span v-if="entry.upcoming" class="home-card-status">建设中</span></div>
          <img class="home-card-art" :src="entry.artwork" width="520" height="360" alt="" aria-hidden="true" />
          <div class="home-card-copy"><h3>{{ entry.title }}</h3><p>{{ entry.description }}</p><span v-if="!accessible(entry)" class="home-card-permission">暂无访问权限</span></div>
          <span class="home-card-arrow" aria-hidden="true"><ArrowUpRight v-if="accessible(entry)" /><LockKeyhole v-else /></span>
        </a>
        <div class="home-gallery-caption" aria-hidden="true"><span></span>让复杂的内容，回归清晰。</div>
      </nav>
    </div>
  </main>
</template>
