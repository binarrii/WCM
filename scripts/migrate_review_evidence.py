"""Move legacy embedded result images into S3, preserving task timestamps."""

import json

from api.review_evidence import archive_evidence
from wcm_facerec.cluster import connect
from wcm_facerec.config import settings


def main():
    assert settings.cluster_enabled and settings.image_storage == "s3"
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute("SELECT id, results FROM review_tasks WHERE results IS NOT NULL")
        tasks = cursor.fetchall()
    changed = 0
    for row in tasks:
        results = json.loads(row["results"])
        updated = archive_evidence(row["id"], results)
        if updated == results:
            continue
        with connect() as connection, connection.cursor() as cursor:
            changed += cursor.execute(
                "UPDATE review_tasks SET results = %s, updated_at = updated_at "
                "WHERE id = %s AND status NOT IN ('queued', 'processing', 'cancelling') AND results = CAST(%s AS JSON)",
                (json.dumps(updated, ensure_ascii=False), row["id"], row["results"]),
            )
    print(json.dumps({"review_results_checked": len(tasks), "review_results_updated": changed}))


if __name__ == "__main__":
    main()
