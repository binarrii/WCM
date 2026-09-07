<script setup>
import { ref } from 'vue';
import FaceOverlay from './FaceOverlay.vue';
const selected = ref('');
    const mode = ref('corners');
    const faces = [
      { key: 'a', box: { x: .20, y: .19, w: .18, h: .32 }, candidates: [{ markerId: 'a', name: '示例人物 A', similarity: .92 }] },
      { key: 'b', box: { x: .62, y: .15, w: .18, h: .32 }, candidates: [{ markerId: 'b', name: '示例人物 B', similarity: .87 }] }
    ];

</script>
<template><main><header><small>WCM Core · 交互预览</small><h1>让每一条命中都有明确位置</h1><p>示意画面与模拟候选，仅用于体验标记交互。</p></header>
  <section><div><div class="stage">
    <svg width="800" height="450" viewBox="0 0 800 450" aria-label="两位示例人物的示意画面">
      <defs><linearGradient id="bg"><stop stop-color="#30384d"/><stop offset="1" stop-color="#151d2e"/></linearGradient></defs>
      <rect width="800" height="450" fill="url(#bg)"/>
      <path d="M95 450 Q105 240 230 234 Q365 235 380 450" fill="#757e98"/>
      <ellipse cx="232" cy="158" rx="56" ry="69" fill="#d7bca4"/>
      <path d="M174 161 Q156 66 231 78 Q300 75 290 164 L272 125 Q208 145 190 116Z" fill="#292732"/>
      <path d="M440 450 Q454 233 568 216 Q703 234 738 450" fill="#596d79"/>
      <ellipse cx="568" cy="139" rx="57" ry="68" fill="#cbb09a"/>
      <path d="M510 119 Q509 62 568 65 Q628 63 626 120 Q564 98 510 119" fill="#252634"/>
      <text x="24" y="424" fill="#c5cddd" font-size="13">Ⅱ  00:01:19 / 00:09:49</text>
    </svg>
    <FaceOverlay :faces="faces" :rect="{left:0,top:0,width:800,height:450}" :selected="selected" :mode="mode" @select="selected = $event"/>
  </div><footer><label>人脸标记 <select v-model="mode"><option value="corners">四角框</option><option value="boxes">完整框</option><option value="hidden">隐藏</option></select></label><span>悬浮查看候选 · 点击联动记录</span></footer></div>
  <aside><h2>2 条命中记录</h2><button v-for="face in faces" :class="{selected: selected === face.key}" @click="selected = face.key"><strong>00:01:19.000～00:01:21.000</strong><span>{{face.candidates[0].name}}</span><small>点击定位并显示姓名</small></button><p>实际审核页会暂停到采样帧，播放时自动隐藏位置标记。</p></aside></section></main></template>
