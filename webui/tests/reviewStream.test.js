import test from 'node:test';
import assert from 'node:assert/strict';
import { createReviewStream, mergeReviewTask, reviewStreamUrl } from '../src/services/reviewStream.js';
import { submitReview } from '../src/services/reviewSubmission.js';

const flush = async () => { await Promise.resolve(); await Promise.resolve(); };
const mockSockets = () => {
  const sockets = [];
  return { sockets, factory: () => {
    const socket = { sent: [], close() { this.closed = true; }, send(data) { this.sent.push(JSON.parse(data)); }, message(data) { this.onmessage({data:JSON.stringify(data)}); } };
    sockets.push(socket); return socket;
  }};
};

test('subscription uses the submission endpoint and only sends a subscribe message', async () => {
  const { sockets, factory } = mockSockets(); const events = [];
  const stream = createReviewStream({url:'ws://test',socketFactory:factory,onEvent:event=>events.push(event)});
  stream.start(['task'], {watchList:true}); sockets[0].onopen();
  assert.deepEqual(sockets[0].sent,[{type:'subscribe',task_ids:['task'],watch_list:true}]);
  sockets[0].message({type:'snapshot',tasks:[]}); sockets[0].message({type:'progress',task_id:'task',progress:{sequence:1}});
  await flush(); assert.equal(events.length,2);
  stream.start(['task'], {watchList:true}); assert.equal(sockets.length,1);
  stream.stop(); assert.equal(sockets[0].closed,true);
  assert.equal(reviewStreamUrl('/api/v1','https://example.com/'),'wss://example.com/api/v1/ws/analyze_media');
});

test('disconnect reconnects with a subscription and stop invalidates late socket callbacks', async t => {
  t.mock.timers.enable({apis:['setTimeout']});
  const { sockets, factory } = mockSockets(); const events=[];
  const stream=createReviewStream({url:'ws://test',socketFactory:factory,onEvent:event=>events.push(event),retryDelay:100});
  stream.start(['first']); sockets[0].onopen();
  const late=sockets[0].onmessage;
  sockets[0].onclose({code:1006});t.mock.timers.tick(100);
  assert.equal(sockets.length,2);sockets[1].onopen();
  assert.equal(sockets[1].sent[0].type,'subscribe');
  stream.stop();late({data:JSON.stringify({type:'progress'})});await flush();
  t.mock.timers.tick(100000);assert.equal(sockets.length,2);assert.equal(events.length,0);
});

test('heartbeat timeout reconnects without any HTTP polling', t => {
  t.mock.timers.enable({apis:['setTimeout']});
  const { sockets, factory }=mockSockets();
  const stream=createReviewStream({url:'ws://test',socketFactory:factory,onEvent:()=>{},retryDelay:100,heartbeatTimeout:1000});
  stream.start(['task']);sockets[0].onopen();
  t.mock.timers.tick(900);sockets[0].message({type:'heartbeat'});t.mock.timers.tick(900);
  assert.equal(sockets.length,1);
  t.mock.timers.tick(100);t.mock.timers.tick(100);assert.equal(sockets.length,2);stream.stop();
});

test('out-of-order progress cannot move a task backwards or revive a completed task', () => {
  const current={id:'task',status:'processing',progress:{sequence:3,percent:40}};
  assert.equal(mergeReviewTask(current,{status:'processing',progress:{sequence:2,percent:20}}).progress.percent,40);
  const completed=mergeReviewTask(current,{status:'completed',progress:{percent:100}});
  assert.equal(mergeReviewTask(completed,current).status,'completed');
});

test('one submission socket carries progress and final task metadata without another subscription', async () => {
  const { sockets, factory }=mockSockets();const events=[];let accepted;
  const result=submitReview('ws://test',{url:'https://fixture/video.mp4'},id=>accepted=id,factory,event=>events.push(event));
  sockets[0].onopen();sockets[0].message({status:'accepted',taskId:'task'});
  sockets[0].message({type:'progress',task_id:'task',progress:{percent:30}});
  sockets[0].message({status:'completed',taskId:'task',results:[],task:{id:'task',status:'partial'}});
  assert.deepEqual(await result,[]);assert.equal(accepted,'task');assert.equal(sockets.length,1);assert.equal(sockets[0].sent.length,1);
  assert.equal(events[0].progress.percent,30);assert.equal(events[1].task.status,'partial');
});

test('leaving the page closes its submission socket without sending a task cancellation', async () => {
  const {sockets,factory}=mockSockets();const controller=new AbortController();
  const result=submitReview('ws://test',{url:'https://fixture/video.mp4'},()=>{},factory,()=>{},controller.signal);
  sockets[0].onopen();sockets[0].message({status:'accepted',taskId:'task'});controller.abort();
  await assert.rejects(result,error=>error.taskId==='task');
  assert.equal(sockets[0].closed,true);assert.equal(sockets[0].sent.length,1);
});
