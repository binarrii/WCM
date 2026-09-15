import test from 'node:test';
import assert from 'node:assert/strict';
import { createReplicationPull, replicationPullUrl } from '../src/services/replicationPull.js';

const fixture = t => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const sockets = [], values = [], states = [], errors = [], pending = [];
  const client = createReplicationPull({
    url: 'ws://test', retryDelay: 100, requestTimeout: 20000,
    onData: value => values.push(value), onState: value => states.push(value),
    onError: error => errors.push(error.message), onPending: value => pending.push(value),
    socketFactory: () => {
      const socket = {
        sent: [], close() { this.closed = true; },
        send(raw) { this.sent.push(JSON.parse(raw)); },
        reply(id = this.sent.at(-1).request_id, data = { enabled: true }) {
          this.onmessage({ data: JSON.stringify({ type: 'status', request_id: id, data }) });
        }
      };
      sockets.push(socket);
      return socket;
    }
  });
  t.after(() => client.stop());
  client.start();
  return { client, sockets, values, states, errors, pending };
};

test('pull URL supports same-origin, a path prefix, and external HTTPS API', () => {
  assert.equal(replicationPullUrl('/api/v1', 'http://host/#/system'), 'ws://host/api/v1/insightface/replication/ws');
  assert.equal(replicationPullUrl('/wcm/api/v1/', 'https://host/wcm/'), 'wss://host/wcm/api/v1/insightface/replication/ws');
  assert.equal(replicationPullUrl('https://api.host/v1', 'http://host'), 'wss://api.host/v1/insightface/replication/ws');
});

test('first pull on open, then another five seconds after each response', t => {
  const { sockets, values, client } = fixture(t);
  const socket = sockets[0];
  assert.deepEqual(socket.sent, []);
  socket.onopen();
  assert.deepEqual(socket.sent, [{ type: 'status', request_id: 1 }]);
  socket.reply();
  t.mock.timers.tick(4999); assert.equal(socket.sent.length, 1);
  t.mock.timers.tick(1); assert.equal(socket.sent.length, 2);
  socket.reply(); assert.equal(values.length, 2);
  client.start(); assert.equal(sockets.length, 1);
});

test('slow requests never overlap and repeated manual refreshes coalesce', t => {
  const { client, sockets } = fixture(t);
  const socket = sockets[0]; socket.onopen();
  t.mock.timers.tick(15000);
  client.refresh(); client.refresh(); client.refresh();
  assert.equal(socket.sent.length, 1);
  socket.reply(); assert.equal(socket.sent.length, 2);
  socket.reply(); t.mock.timers.tick(5000); assert.equal(socket.sent.length, 3);
});

test('manual refresh uses the same socket and resets the periodic timer', t => {
  const { client, sockets } = fixture(t);
  const socket = sockets[0]; socket.onopen(); socket.reply();
  t.mock.timers.tick(4000); client.refresh(); socket.reply();
  t.mock.timers.tick(1000); assert.equal(socket.sent.length, 2);
  t.mock.timers.tick(4000); assert.equal(socket.sent.length, 3);
  assert.equal(sockets.length, 1);
});

test('responses started before a manual write can be drained without restoring stale state', t => {
  const { client, sockets, values } = fixture(t);
  const socket = sockets[0]; socket.onopen();
  client.refresh({ discardPending: true });
  socket.reply(1, { enabled: true, committed_sequence: 1 });
  assert.equal(values.length, 0); assert.equal(socket.sent.length, 2);
  socket.reply(2, { enabled: true, committed_sequence: 2 });
  assert.equal(values[0].committed_sequence, 2);
});

test('disconnect reconnects, old callbacks and unsolicited responses are ignored', t => {
  const { sockets, values, states } = fixture(t);
  sockets[0].onopen(); const late = sockets[0].onmessage;
  sockets[0].onclose(); assert.equal(states.at(-1), 'reconnecting');
  t.mock.timers.tick(100); sockets[1].onopen();
  late({ data: JSON.stringify({ type: 'status', request_id: 1, data: { enabled: true } }) });
  sockets[1].reply(1); assert.equal(values.length, 0);
  sockets[1].reply(); assert.equal(values.length, 1);
  sockets[1].reply(); assert.equal(values.length, 1);
});

test('opening and response deadlines both recover a stuck connection', t => {
  const { sockets } = fixture(t);
  t.mock.timers.tick(20000); assert.equal(sockets[0].closed, true);
  t.mock.timers.tick(100); sockets[1].onopen();
  t.mock.timers.tick(20000); assert.equal(sockets[1].closed, true);
  t.mock.timers.tick(200); assert.equal(sockets.length, 3);
});

test('database errors preserve the socket and retry using a new request ID', t => {
  const { sockets, errors, values } = fixture(t);
  const socket = sockets[0]; socket.onopen();
  socket.onmessage({ data: JSON.stringify({ type: 'error', request_id: 1, error: '暂时无法读取同步状态' }) });
  assert.deepEqual(errors, ['暂时无法读取同步状态']); assert.equal(values.length, 0);
  t.mock.timers.tick(5000); socket.reply();
  assert.equal(values.length, 1); assert.equal(sockets.length, 1);
});

test('hide or unmount releases timers; returning immediately queries fresh state', t => {
  const { client, sockets, values } = fixture(t);
  sockets[0].onopen(); const late = sockets[0].onmessage;
  client.stop(); t.mock.timers.tick(100000);
  late({ data: JSON.stringify({ type: 'status', request_id: 1, data: { enabled: true } }) });
  assert.equal(sockets.length, 1); assert.equal(values.length, 0);
  client.start(); sockets[1].onopen(); assert.equal(sockets[1].sent.length, 1);
});

test('malformed status responses reconnect without marking data as fresh', t => {
  const { sockets, values } = fixture(t);
  sockets[0].onopen(); sockets[0].reply(1, {});
  assert.equal(values.length, 0); assert.equal(sockets[0].closed, true);
  t.mock.timers.tick(100); assert.equal(sockets.length, 2);
});
