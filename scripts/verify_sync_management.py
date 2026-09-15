"""Isolated MySQL + HTTP verification; never calls an InsightFace server."""

import asyncio
from unittest.mock import Mock

import httpx
from fastapi import FastAPI

from api.face_sync_worker import sync_once
from api.insightface_management import insightface_management_bp
from wcm_facerec import face_sync_store as store
from wcm_facerec import person_operations
from wcm_facerec.cluster import connect, run_sync
from wcm_facerec.config import settings


def sql(statement, args=()):
    with connect() as db, db.cursor() as c:
        c.execute(statement, args)
        return c.fetchall()


async def main():
    assert settings.review_tasks_db_name.startswith("wcm_verify_"), "Use an isolated test database"
    assert settings.insightface_replicas == {
        "a": "http://replica-a.invalid",
        "b": "http://replica-b.invalid",
    }
    person_operations.initialize()
    store.initialize()
    assert not store.status()["initialized"], "Use a fresh database"
    app = FastAPI()
    app.include_router(insightface_management_bp, prefix="/api/v1")
    target = Mock()
    endpoint = "/api/v1/insightface/replication"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://test"
    ) as client:

        async def post(node=None):
            return await client.post(endpoint + "/sync", json={"node_id": node})

        async def node(name="a"):
            response = await client.get(endpoint)
            assert response.status_code == 200, response.text
            return next(n for n in response.json()["replicas"] if n["id"] == name)

        assert (await post()).status_code == 409
        sql("UPDATE face_sync_control SET initialized=TRUE")
        assert (await post()).status_code == 409
        sql("UPDATE face_sync_nodes SET state='ready',heartbeat=UTC_TIMESTAMP(3)")
        responses = await asyncio.gather(post("a"), post("a"))
        assert all(r.status_code == 202 for r in responses)
        request_id = responses[0].json()["requested"][0]["request_id"]
        assert responses[1].json()["requested"][0]["request_id"] == request_id
        assert not (await node())["manual_request"]["completed_at"]
        await sync_once("a", target)
        current = await node()
        assert current["manual_request"]["completed_at"] and current["read_eligible"]
        assert current["heartbeat"].endswith(("Z", "+00:00"))
        print(
            "PASS HTTP 202, concurrent coalescing, durable worker acknowledgement, UTC status",
            flush=True,
        )

        sql(
            "UPDATE face_sync_nodes SET state='retry',attempts=3,next_retry=DATE_ADD(UTC_TIMESTAMP(3),INTERVAL 1 HOUR) WHERE id='a'"
        )
        assert (await post("a")).status_code == 202
        current = await node()
        pending = current["manual_request"]["id"]
        assert current["next_retry"] is None and current["attempts"] == 3
        target.health.side_effect = TimeoutError("unreachable before writes")
        await sync_once("a", target)
        current = await node()
        assert current["state"] == "retry" and not current["manual_request"]["completed_at"]
        assert (await post("a")).json()["requested"][0]["request_id"] == pending
        target.health.side_effect = None
        await sync_once("a", target)
        assert (await node())["manual_request"]["completed_at"]
        print(
            "PASS explicit retry bypasses backoff without clearing failure history; health failure stays pending",
            flush=True,
        )

        for state in ("syncing", "draining", "quarantined", "disabled", "new"):
            sql(
                "UPDATE face_sync_nodes SET state=%s,owner='old-owner',last_error='keep-reason',attempts=4 WHERE id='a'",
                (state,),
            )
            before = sql("SELECT * FROM face_sync_nodes WHERE id='a'")[0]
            assert (await post("a")).status_code == 409
            assert sql("SELECT * FROM face_sync_nodes WHERE id='a'")[0] == before
            response = await post()
            assert response.status_code == 202 and response.json()["skipped"] == [
                {"id": "a", "state": state}
            ]
            assert [n["id"] for n in response.json()["requested"]] == ["b"]
        target.reset_mock()
        await sync_once("a", target)
        target.health.assert_not_called()
        await sync_once("b", target)
        print(
            "PASS busy/isolated/uninitialized nodes cannot be changed by single or bulk requests",
            flush=True,
        )

        sql("UPDATE face_sync_nodes SET state='ready',owner=NULL,last_error=NULL WHERE id='a'")
        assert (await post("a")).status_code == 202
        old_id = (await node())["manual_request"]["id"]
        await sync_once("a", target)
        assert (await post("a")).status_code == 202
        newer = (await node())["manual_request"]["id"]
        assert newer != old_id
        await run_sync(store.complete_sync_request, "a", old_id)
        assert not (await node())["manual_request"]["completed_at"]
        await sync_once("a", target)
        assert (await node())["manual_request"]["completed_at"]

        sql("UPDATE face_sync_control SET head=5")
        assert (await post("a")).status_code == 202
        current = await node()
        assert current["lag"] == 5 and not current["read_eligible"]
        await run_sync(store.complete_sync_request, "a", current["manual_request"]["id"])
        assert not (await node())["manual_request"]["completed_at"]
        sql("UPDATE face_sync_nodes SET applied_seq=5 WHERE id='a'")
        await sync_once("a", target)
        assert (await node())["manual_request"]["completed_at"]
        sql(
            "UPDATE face_sync_nodes SET heartbeat=DATE_SUB(UTC_TIMESTAMP(3),INTERVAL 1 HOUR) WHERE id='a'"
        )
        assert not (await node())["read_eligible"]
        sql("UPDATE face_sync_control SET primary_recovering=TRUE")
        assert (await post()).status_code == 409
        assert (await post("missing")).status_code == 404
        print(
            "PASS stale acknowledgements, lag and stale heartbeats never report completion/readiness incorrectly",
            flush=True,
        )
    print("ALL SYNC MANAGEMENT CHECKS PASSED", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
