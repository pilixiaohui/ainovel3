from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from app.storage.graph import GraphStorage


def _copy_db(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"db path not found: {src}")
    if dst.exists():
        raise FileExistsError(f"backup path already exists: {dst}")
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def _restore_db(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"backup path not found: {src}")
    if dst.exists():
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def run_backfill_migrations(db_path: Path) -> None:
    storage = GraphStorage(db_path=db_path)
    try:
        storage.run_backfill_migrations()
    finally:
        storage.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run explicit graph migrations or rollback from backup."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    migrate = subparsers.add_parser("migrate", help="Run backfill migrations.")
    migrate.add_argument("--db-path", required=True, help="Path to the Kuzu DB.")
    migrate.add_argument(
        "--backup-path",
        required=False,
        help="Optional backup path before migration.",
    )

    rollback = subparsers.add_parser("rollback", help="Restore DB from backup.")
    rollback.add_argument("--db-path", required=True, help="Path to the Kuzu DB.")
    rollback.add_argument("--backup-path", required=True, help="Path to the backup.")

    args = parser.parse_args()
    db_path = Path(args.db_path)

    if args.command == "migrate":
        backup_path = Path(args.backup_path) if args.backup_path else None
        if backup_path:
            _copy_db(db_path, backup_path)
        run_backfill_migrations(db_path)
        return

    if args.command == "rollback":
        backup_path = Path(args.backup_path)
        _restore_db(backup_path, db_path)
        return

    raise RuntimeError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
