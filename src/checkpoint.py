from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any


class Checkpoint:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("pragma journal_mode=WAL")
        self.conn.execute(
            """
            create table if not exists progress (
              patient_sn text not null,
              trial_id text not null,
              standard_no text not null,
              status text not null,
              error text default '',
              updated_at datetime default current_timestamp,
              primary key (patient_sn, trial_id, standard_no)
            )
            """
        )
        self.conn.execute(
            """
            create table if not exists patient_status (
              patient_sn text primary key,
              source_file text default '',
              status text not null,
              total_rules integer default 0,
              done_rules integer default 0,
              failed_rules integer default 0,
              last_error text default '',
              started_at datetime default current_timestamp,
              elapsed_seconds real default 0,
              updated_at datetime default current_timestamp
            )
            """
        )
        self._ensure_patient_status_columns()
        self.conn.commit()

    def _ensure_patient_status_columns(self) -> None:
        columns = {row[1] for row in self.conn.execute("pragma table_info(patient_status)").fetchall()}
        if "started_at" not in columns:
            self.conn.execute("alter table patient_status add column started_at datetime default ''")
        if "elapsed_seconds" not in columns:
            self.conn.execute("alter table patient_status add column elapsed_seconds real default 0")

    def done(self, patient_sn: str, trial_id: str, standard_no: str) -> bool:
        with self.lock:
            row = self.conn.execute(
                "select status from progress where patient_sn=? and trial_id=? and standard_no=?",
                (patient_sn, trial_id, standard_no),
            ).fetchone()
        return bool(row and row[0] == "done")

    def mark_task(self, patient_sn: str, trial_id: str, standard_no: str, status: str, error: str = "") -> None:
        with self.lock:
            self.conn.execute(
                """
                insert into progress(patient_sn, trial_id, standard_no, status, error, updated_at)
                values (?, ?, ?, ?, ?, current_timestamp)
                on conflict(patient_sn, trial_id, standard_no)
                do update set status=excluded.status, error=excluded.error, updated_at=current_timestamp
                """,
                (patient_sn, trial_id, standard_no, status, error[:1000]),
            )
            self.conn.commit()

    def mark(self, patient_sn: str, trial_id: str, standard_no: str, status: str) -> None:
        self.mark_task(patient_sn, trial_id, standard_no, status)

    def mark_patient(
        self,
        patient_sn: str,
        source_file: str,
        status: str,
        total_rules: int,
        done_rules: int,
        failed_rules: int,
        last_error: str = "",
        elapsed_seconds: float | None = None,
    ) -> None:
        elapsed_value = 0.0 if elapsed_seconds is None else round(float(elapsed_seconds), 3)
        with self.lock:
            self.conn.execute(
                """
                insert into patient_status(patient_sn, source_file, status, total_rules, done_rules, failed_rules, last_error, started_at, elapsed_seconds, updated_at)
                values (?, ?, ?, ?, ?, ?, ?, current_timestamp, ?, current_timestamp)
                on conflict(patient_sn)
                do update set source_file=excluded.source_file,
                              status=excluded.status,
                              total_rules=excluded.total_rules,
                              done_rules=excluded.done_rules,
                              failed_rules=excluded.failed_rules,
                              last_error=excluded.last_error,
                              started_at=case when excluded.status='running' then current_timestamp else patient_status.started_at end,
                              elapsed_seconds=case when excluded.status='running' then 0 else excluded.elapsed_seconds end,
                              updated_at=current_timestamp
                """,
                (patient_sn, source_file, status, total_rules, done_rules, failed_rules, last_error[:1000], elapsed_value),
            )
            self.conn.commit()

    def failed_patient_ids(self) -> set[str]:
        with self.lock:
            rows = self.conn.execute(
                """
                select distinct patient_sn from progress where status='failed'
                union
                select patient_sn from patient_status where status in ('failed', 'partial')
                """
            ).fetchall()
        return {str(row[0]) for row in rows}

    def failed_patient_files(self) -> dict[str, str]:
        with self.lock:
            rows = self.conn.execute(
                """
                select patient_sn, source_file
                from patient_status
                where status in ('failed', 'partial') or failed_rules > 0
                """
            ).fetchall()
        return {str(patient_sn): str(source_file) for patient_sn, source_file in rows if source_file}

    def recover_running_patients(self) -> int:
        with self.lock:
            rows = self.conn.execute(
                """
                select patient_sn, total_rules, done_rules, failed_rules
                from patient_status
                where status='running'
                """
            ).fetchall()
            changed = 0
            for patient_sn, total_rules, done_rules, failed_rules in rows:
                done = int(done_rules or 0)
                failed = int(failed_rules or 0)
                total = int(total_rules or 0)
                if total and done >= total and failed == 0:
                    status = "done"
                elif done > 0 or failed > 0:
                    status = "partial"
                else:
                    status = "failed"
                self.conn.execute(
                    """
                    update patient_status
                    set status=?, last_error=?, updated_at=current_timestamp
                    where patient_sn=? and status='running'
                    """,
                    (status, "recovered stale running status at startup", patient_sn),
                )
                changed += 1
            self.conn.commit()
        return changed

    def task_status_counts(self) -> dict[str, int]:
        with self.lock:
            rows = self.conn.execute("select status, count(*) from progress group by status").fetchall()
        return {str(status): int(count) for status, count in rows}

    def patient_rows(self) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.conn.execute(
                """
                select patient_sn, source_file, status, total_rules, done_rules, failed_rules, last_error, updated_at
                from patient_status
                order by updated_at desc
                """
            ).fetchall()
        keys = ["patient_sn", "source_file", "status", "total_rules", "done_rules", "failed_rules", "last_error", "updated_at"]
        return [dict(zip(keys, row)) for row in rows]

    def failed_task_rows(self) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.conn.execute(
                """
                select patient_sn, trial_id, standard_no, status, error, updated_at
                from progress
                where status='failed'
                order by updated_at desc
                """
            ).fetchall()
        keys = ["patient_sn", "trial_id", "standard_no", "status", "error", "updated_at"]
        return [dict(zip(keys, row)) for row in rows]
