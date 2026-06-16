#!/usr/bin/env bash
# =============================================================================
# データベース初期化スクリプト
#   スキーマ作成 → 競馬場マスタ投入 → (任意) サンプルデータ投入 を行う。
#
# 使い方:
#   ./scripts/init_db.sh                 # keiba.db を作成
#   ./scripts/init_db.sh mydb.db         # ファイル名を指定
#   ./scripts/init_db.sh keiba.db --sample  # サンプルデータも投入
# =============================================================================
set -euo pipefail

DB="${1:-keiba.db}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "==> スキーマ作成: $DB"
sqlite3 "$DB" < "$ROOT/schema/schema.sql"

echo "==> 競馬場マスタ投入"
sqlite3 "$DB" < "$ROOT/schema/seed_venues.sql"

if [[ "${2:-}" == "--sample" ]]; then
    echo "==> サンプルデータ投入"
    sqlite3 "$DB" < "$ROOT/data/sample.sql"
fi

echo "==> 完了。テーブル一覧:"
sqlite3 "$DB" ".tables"
