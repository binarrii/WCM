import test from 'node:test';
import assert from 'node:assert/strict';
import { taskProgress, sampleTime, createTaskPoller } from '../src/services/reviewProgress.js';
import { reviewSocketUrl, submitReview } from '../src/services/reviewSubmission.js';

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

test('stopping a poller ignores late results and does not queue another poll', async () => {
  let done; const seen=[];
  const poller=createTaskPoller({getTask:()=>new Promise(resolve=>done=resolve),onTask:task=>seen.push(task)});
  poller.start('task');poller.stop();done({status:'processing'});
  await Promise.resolve();await Promise.resolve();
  assert.equal(seen.length,0);
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
