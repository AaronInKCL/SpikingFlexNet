import sqlite3
import pandas as pd
from pathlib import Path
import json

def extract_sqlite_log_to_csv(log_path: Path, out_csv_path: Path):
    try:
        conn = sqlite3.connect(log_path)
        # 获取第一个表名
        table_names = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
        if table_names.empty:
            print(f"[✗] No tables found in: {log_path}")
            return False

        table = table_names.iloc[0, 0]
        print(f"[✓] Found table: {table} in {log_path.name}")

        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        df.to_csv(out_csv_path, index=False)
        print(f"[✓] Saved CSV to: {out_csv_path}")
        conn.close()
        return True
    except Exception as e:
        print(f"[!] Failed to process {log_path.name}: {e}")
        return False


if __name__ == "__main__":
    input_root = Path("checkpoints")
    output_root = Path("data_csv/training_curve")
    output_root.mkdir(parents=True, exist_ok=True)

    registry = {}

    for i, run_folder in enumerate(sorted(input_root.iterdir())):
        log_file = run_folder / "logs"
        if log_file.exists():
            key = f"{i:06d}"
            model_name = run_folder.name
            out_csv_path = output_root / f"{key}.csv"

            success = extract_sqlite_log_to_csv(log_file, out_csv_path)
            if success:
                registry[key] = model_name
            else:
                print(f"[!] Skipped: {run_folder.name}")

    # 保存模型名与编号映射
    registry_path = output_root / "registry.json"
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)

    print(f"[✓] Done. Extracted {len(registry)} runs.")
    print(f"[✓] Registry saved to: {registry_path}")
