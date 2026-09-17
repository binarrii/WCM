<script setup>
import mediaLibraryArtwork from '../assets/media-library-coming-soon.svg';
</script>

<template>
  <section class="media-library-page" aria-labelledby="media-library-title">
    <div class="media-library-stage">
      <div class="media-library-eyebrow" aria-hidden="true">WCM <span>/</span> MEDIA LIBRARY</div>
      <img class="media-library-artwork" :src="mediaLibraryArtwork" width="1000" height="560" alt="视频、图片与音频素材环绕着蓝紫色的媒体文件夹" />
      <div class="media-library-copy">
        <span class="media-library-status"><span aria-hidden="true"></span>COMING SOON</span>
        <h2 id="media-library-title">媒资库<span>，建设中</span></h2>
        <p>全新的媒体资源管理空间正在准备中，敬请期待。</p>
      </div>
      <div class="media-library-footnote" aria-hidden="true"><span></span>为每一份素材，准备一个新空间<span></span></div>
    </div>
  </section>
</template>

<style scoped>
.media-library-page { --media-accent: color-mix(in srgb, var(--color-primary) 75%, var(--text-primary)); padding: 24px var(--workspace-page-gutter); }
.media-library-stage { position: relative; isolation: isolate; overflow: hidden; min-height: calc(100dvh - 156px); display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 40px 24px; border: 1px solid var(--border-color); border-radius: 24px; background: radial-gradient(ellipse at 50% 36%, color-mix(in srgb, var(--color-primary) 9%, transparent), transparent 66%), var(--bg-card); }
.media-library-stage::before { content: ''; position: absolute; z-index: -1; width: 620px; height: 620px; top: 50%; left: 50%; transform: translate(-50%, -60%); border: 1px solid color-mix(in srgb, var(--color-primary) 6%, transparent); border-radius: 50%; box-shadow: 0 0 0 70px color-mix(in srgb, var(--color-primary) 2%, transparent); pointer-events: none; }
.media-library-eyebrow { color: var(--text-muted); font: 600 .65rem/1.5 var(--font-sans); letter-spacing: .2em; }
.media-library-eyebrow span { margin: 0 10px; opacity: .5; }
.media-library-artwork { display: block; width: min(100%, 720px); height: clamp(220px, 40dvh, 360px); margin: 6px 0 4px; object-fit: contain; }
.media-library-copy { text-align: center; }
.media-library-status { display: inline-flex; align-items: center; gap: 7px; padding: 6px 12px; border: 1px solid color-mix(in srgb, var(--color-primary) 14%, transparent); border-radius: 30px; background: color-mix(in srgb, var(--color-primary) 5%, transparent); color: var(--media-accent); font: 600 .6rem/1.3 var(--font-sans); letter-spacing: .16em; }
.media-library-status > span { width: 5px; height: 5px; border-radius: 50%; background: currentColor; }
.media-library-copy h2 { margin: 17px 0 12px; color: var(--text-primary); font: 700 clamp(1.6rem, 2.5vw, 2.25rem)/1.4 var(--font-display); letter-spacing: .025em; }
.media-library-copy h2 > span { color: var(--media-accent); }
.media-library-copy p { margin: 0; color: var(--text-secondary); font-size: .85rem; line-height: 1.9; }
.media-library-footnote { display: flex; align-items: center; gap: 12px; margin-top: 38px; color: var(--text-muted); font-size: .65rem; letter-spacing: .08em; }
.media-library-footnote > span { width: 28px; height: 1px; background: var(--border-color); }
@media (max-width: 800px) { .media-library-stage { padding: 32px 20px; } }
@media (max-width: 520px) {
  .media-library-page { padding-top: 20px; padding-bottom: 20px; }
  .media-library-stage { min-height: calc(100dvh - 200px); padding: 30px 16px; border-radius: 18px; }
  .media-library-artwork { width: calc(100% + 24px); max-width: none; height: auto; margin: 24px -12px 20px; }
  .media-library-copy p { max-width: 230px; margin-inline: auto; font-size: .8rem; }
  .media-library-footnote { margin-top: 32px; gap: 8px; font-size: .58rem; letter-spacing: 0; }
  .media-library-footnote > span { width: 16px; }
}
</style>
