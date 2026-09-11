import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { taskProgress, sampleTime } from '../src/services/reviewProgress.js';
import { reviewSocketUrl, submitReview } from '../src/services/reviewSubmission.js';

const progressComponent = readFileSync(new URL('../src/components/ReviewProgress.vue', import.meta.url), 'utf8');

test('unknown progress is indeterminate; only finished work displays 100%', () => {
  assert.equal(taskProgress({status:'processing'}).percent, null);
  assert.equal(taskProgress({status:'processing', progress:{percent:100}}).percent,99);
  assert.equal(taskProgress({status:'completed'}).percent,100);
  assert.equal(taskProgress({status:'failed',progress:{phase:'failed',percent:20}}).percent,20);
  assert.equal(taskProgress({status:'failed',progress:{phase:'finished',percent:100}}).percent,100);
});

test('each concurrent window exposes its selected samples and current module', () => {
  const view=taskProgress({status:'processing',progress:{percent:42.3,active_windows:[
    {index:8,start_seconds:21,end_seconds:23,sample_timestamps:[21,22,23],stages:{visual:[21,22,23],ocr:[22]}},
    {index:9,start_seconds:24,end_seconds:26,sample_timestamps:[24,25,26],stages:{face:[24]}}
  ]}});
  assert.match(view.windows[0].title,/窗口 #8/);
  assert.match(view.windows[0].stages,/文字：00:00:22.000/);
  assert.match(view.windows[1].stages,/人脸：00:00:24.000/);
  assert.equal(sampleTime(8.9999),'00:00:09.000');
});

test('window details use the smaller type size and reserve two running-stage lines', () => {
  assert.match(progressComponent, /\.active-window small\s*\{[^}]*font-size:\s*10px[^}]*\}/);
  assert.match(progressComponent, /\.running-stage\s*\{[^}]*min-height:\s*3em[^}]*line-height:\s*1\.5[^}]*\}/);
});

test('difficult face resampling exposes a dedicated bounded sub-progress', () => {
  const view=taskProgress({status:'processing',progress:{phase:'resampling',percent:99,sub_progress:{
    stage:'face_resampling',completed:7,total:10,percent:70
  }}});
  assert.equal(view.label,'审核收尾');
  assert.equal(view.percent,99);
  assert.deepEqual(view.subProgress,{label:'困难帧补采',percent:70,details:'7 / 10 帧'});
  assert.equal(taskProgress({status:'completed',progress:{phase:'finished',sub_progress:{stage:'face_resampling',completed:10,total:10}}}).subProgress,null);
});

test('submission returns the task id before results and never resubmits on disconnect', async () => {
  let socket={send:()=>{},close:()=>{}};let id;
  const response=submitReview('ws://test',{},value=>id=value,()=>socket);
  socket.onopen();socket.onmessage({data:JSON.stringify({status:'accepted',taskId:'task'})});
  assert.equal(id,'task');
  socket.onmessage({data:JSON.stringify({status:'completed',results:[]})});
  assert.deepEqual(await response,[]);
  const failed=submitReview('ws://test',{},()=>{},()=>socket={send:()=>{},close:()=>{}});
  socket.onmessage({data:JSON.stringify({status:'accepted',taskId:'second'})});
  socket.onclose();await assert.rejects(failed,error=>error.taskId==='second');
  assert.equal(reviewSocketUrl('/api/v1','https://example.com/#/video'),'wss://example.com/api/v1/ws/analyze_media');
});
