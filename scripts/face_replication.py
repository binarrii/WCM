"""Inspect, seed and recover WCM-managed InsightFace replicas through HTTP.

Seeding verifies an independently restored consistent snapshot; it never copies
or opens a SQLite file. --fenced is an operator assertion, not a fencing action.
"""

import argparse
import asyncio
import json
from contextlib import nullcontext
from uuid import uuid4

from api import parameter_store
from wcm_facerec import face_sync_store as store
from wcm_facerec import person_operations
from wcm_facerec.cluster import cluster_slot, connect, run_sync
from wcm_facerec.config import settings
from wcm_facerec.face_engine import FaceEngine
from wcm_facerec.face_replication import apply_snapshot, manifest, node_adapter


def baseline_manifest(target, cid, item, hashes):
    snapshot = manifest(cid, item["id"], item, hashes=hashes, strict=False)
    if snapshot.get("rebuildable") is False:
        faces, cursor = [], None
        while True:
            page = target._client.list_faces(cid, item["id"], limit=100, cursor=cursor)
            faces.extend(page.faces)
            cursor = page.next_cursor
            if not cursor:
                break
        if len(faces) != item["face_count"] or len({f["id"] for f in faces}) != len(faces):
            raise store.ReplicationUnavailable("历史人物原生人脸清单不完整")
        # Only identical native snapshot copies can seed these legacy records.
        snapshot["native_faces_hash"] = store.digest(sorted(faces, key=lambda face: face["id"]))
    return snapshot


def scan(target, hashes):
    people = {}
    for cid in store.collections():
        cursor = None
        while True:
            items, cursor = target.list_persons(collection_id=cid, limit=100, cursor=cursor)
            for item in items:
                snapshot = baseline_manifest(target, cid, item, hashes)
                people[store.identity(cid, item["id"])] = snapshot
            if not cursor:
                break
    return people


def record_seed(name, snapshots):
    with connect() as db:
        db.begin()
        with db.cursor() as c:
            root = store.control(c, lock=True, require_initialized=False)
            if root["initialized"]:
                c.execute("SELECT identity,payload FROM face_sync_people")
                existing = {
                    row["identity"]: json.loads(row["payload"])
                    for row in c.fetchall()
                    if json.loads(row["payload"])["person"] is not None
                }
                if existing != snapshots:
                    raise store.ReplicationUnavailable(
                        "主节点与已提交清单不一致，可能存在绕过 WCM 的写入；先排查，不能静默重置基线"
                    )
            else:
                c.executemany(
                    "INSERT INTO face_sync_people (identity,seq,payload) VALUES (%s,0,%s)",
                    [(key, store.encode(value)) for key, value in snapshots.items()],
                )
                c.execute("UPDATE face_sync_control SET initialized=TRUE WHERE id=1")
            c.execute("DELETE FROM face_sync_applied WHERE node_id=%s", (name,))
            c.executemany(
                "INSERT INTO face_sync_applied (node_id,identity,manifest_hash,images_hash) VALUES (%s,%s,%s,%s)",
                [
                    (name, key, store.digest(value), store.digest(value["images"]))
                    for key, value in snapshots.items()
                ],
            )
            c.execute(
                "UPDATE face_sync_nodes SET state='ready',applied_seq=%s,owner=NULL,heartbeat=UTC_TIMESTAMP(3),attempts=0,next_retry=NULL,last_error=NULL WHERE id=%s",
                (root["head"], name),
            )
        db.commit()


async def seed(name, *, locked=False):
    async with (
        nullcontext() if locked else cluster_slot("person-library"),
        cluster_slot(f"face-replica-writer:{name}"),
    ):
        pending = await run_sync(person_operations._load)
        if any(row.get("kind") == "transaction" for row in pending):
            raise store.ReplicationUnavailable("先恢复未完成的人物写入，再初始化副本")
        await run_sync(store.quarantine, name, "verifying_seed")
        while not await run_sync(store.drained, name):
            await asyncio.sleep(0.25)
        hashes = {}
        primary = await run_sync(scan, node_adapter(store.source()), hashes)
        replica = await run_sync(scan, node_adapter(settings.insightface_replicas[name]), hashes)
        if primary != replica:
            raise store.ReplicationUnavailable("副本快照与主节点人物、照片数或元数据不一致")
        await run_sync(record_seed, name, primary)
        print(
            json.dumps(
                {
                    "seeded": name,
                    "people_across_collections": len(primary),
                    "native_backup_only_records": sum(
                        value.get("rebuildable") is False for value in primary.values()
                    ),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


async def recover(name):
    async with cluster_slot(f"face-replica-writer:{name}"):
        _, row = await run_sync(store.node, name)
        if row["state"] != "quarantined":
            raise store.ReplicationUnavailable("只有已隔离的副本允许恢复")
        if not await run_sync(store.drained, name):
            raise store.ReplicationUnavailable("仍有读取租约，请等待读取结束")
        await run_sync(node_adapter(settings.insightface_replicas[name]).health)

        def reset():
            with connect() as db:
                db.begin()
                with db.cursor() as c:
                    # The checkpoint only includes fully verified committed batches.
                    # Removing optimizations forces uncertain persons to be rebuilt.
                    c.execute("DELETE FROM face_sync_applied WHERE node_id=%s", (name,))
                    c.execute(
                        "UPDATE face_sync_nodes SET state='retry',owner=NULL,next_retry=NULL,last_error=NULL WHERE id=%s",
                        (name,),
                    )
                db.commit()

        await run_sync(reset)
        print(json.dumps({"resumed": name, "from_sequence": row["applied_seq"]}))


async def replay_primary(after):
    """Replay onto an operator-restored consistent snapshot using HTTP only.

    The persistent recovery flag stays set after any failed check/interruption.
    The supplied sequence must come from that snapshot's backup record.
    """
    name, owner = "__primary_recovery__", uuid4().hex
    async with cluster_slot("person-library"):
        with connect() as db, db.cursor() as c:
            root = store.control(c)
            if after < 0 or after > root["head"]:
                raise ValueError("备份的 sequence 不在当前日志范围内")
            c.execute("UPDATE face_sync_control SET primary_recovering=TRUE WHERE id=1")
            c.execute(
                "INSERT INTO face_sync_nodes (id,url,state,owner) VALUES (%s,%s,'syncing',%s) ON DUPLICATE KEY UPDATE state='syncing',owner=VALUES(owner)",
                (name, store.source(), owner),
            )
            c.execute("DELETE FROM face_sync_applied WHERE node_id=%s", (name,))
        engine = FaceEngine()
        for pending in await run_sync(person_operations._load):
            if pending.get("kind") == "transaction":
                await person_operations.restore(engine, pending)
        latest = {}
        position = after
        while True:
            batch = await run_sync(store.changes, position)
            if not batch:
                break
            for event in batch:
                for snapshot in event["payload"]:
                    latest[store.identity(snapshot["collection"], snapshot["person_id"])] = snapshot
            position = batch[-1]["seq"]
        if position != root["head"]:
            raise store.ReplicationUnavailable("备份之后的同步日志不完整，保持主节点隔离")
        for snapshot in latest.values():
            await apply_snapshot(name, owner, engine._primary_adapter, snapshot)
        actual = await run_sync(scan, engine._primary_adapter, {})
        with connect() as db, db.cursor() as c:
            c.execute("SELECT identity,payload FROM face_sync_people")
            expected = {
                row["identity"]: json.loads(row["payload"])
                for row in c.fetchall()
                if json.loads(row["payload"])["person"] is not None
            }
            if actual != expected:
                raise store.ReplicationUnavailable("恢复后全量清单不一致，保持主节点隔离")
            c.execute("UPDATE face_sync_control SET primary_recovering=FALSE WHERE id=1")
            c.execute("UPDATE face_sync_nodes SET state='disabled',owner=NULL WHERE id=%s", (name,))
        print(
            json.dumps(
                {"primary_restored_to_sequence": position, "people_replayed": len(latest)},
                ensure_ascii=False,
            )
        )


async def run(args):
    await parameter_store.initialize()
    try:
        await run_sync(person_operations.initialize)
        await run_sync(store.initialize)
        if args.action == "status":
            print(
                json.dumps(await run_sync(store.status), ensure_ascii=False, default=str, indent=2)
            )
        elif args.action == "quarantine":
            await run_sync(store.quarantine, args.node, "operator_quarantine")
        elif args.action == "seed":
            await seed(args.node)
        elif args.action == "seed-all":
            async with cluster_slot("person-library"):
                for name in settings.insightface_replicas:
                    await seed(name, locked=True)
        elif args.action == "recover":
            await recover(args.node)
        elif args.action == "recover-primary":
            async with cluster_slot("person-library"):
                engine = FaceEngine()
                for pending in await run_sync(person_operations._load):
                    if pending.get("kind") == "transaction":
                        await person_operations.restore(engine, pending)
                print("Pending primary operations compensated; committed outbox preserved")
        elif args.action == "replay-primary":
            await replay_primary(args.after_sequence)
    finally:
        await parameter_store.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "status",
            "seed",
            "seed-all",
            "quarantine",
            "recover",
            "recover-primary",
            "replay-primary",
        ),
    )
    parser.add_argument("--node")
    parser.add_argument(
        "--after-sequence",
        type=int,
        help="Committed sequence recorded with the restored primary snapshot",
    )
    parser.add_argument(
        "--fenced",
        action="store_true",
        help="Confirm all old writers and in-flight replica requests have been stopped before recovery",
    )
    args = parser.parse_args()
    if (
        args.action in {"seed", "quarantine", "recover"}
        and args.node not in settings.insightface_replicas
    ):
        parser.error("--node must name a configured replica")
    if args.action in {"recover", "recover-primary", "replay-primary"} and not args.fenced:
        parser.error(
            "Recovery requires fencing the old executor and upstream in-flight requests, then --fenced"
        )
    if args.action == "replay-primary" and args.after_sequence is None:
        parser.error("--after-sequence from the backup record is required")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
