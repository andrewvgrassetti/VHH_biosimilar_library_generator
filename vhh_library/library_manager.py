from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd


class LibraryManager:
    def __init__(self) -> None:
        date_str = datetime.now().strftime("%Y%m%d")
        hex_str = uuid.uuid4().hex[:6]
        self.session_id: str = f"{date_str}_{hex_str}"

    def create_variant_id(self, index: int) -> str:
        return f"VHH-{self.session_id[:8]}-{index:06d}"

    def save_session(self, data: dict, output_dir: str = "sessions") -> dict:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        serializable: dict = {}
        for key, value in data.items():
            if isinstance(value, pd.DataFrame):
                serializable[key] = value.to_dict(orient="records")
            else:
                serializable[key] = value

        filepath = out_path / f"{self.session_id}.json"
        filepath.write_text(json.dumps(serializable, indent=2))
        return {"json": str(filepath)}

    def load_session(self, filepath: str) -> dict:
        return json.loads(Path(filepath).read_text())

    def export_fasta(
        self,
        df: pd.DataFrame,
        filepath: str,
        id_column: str = "variant_id",
        seq_column: str = "aa_sequence",
    ) -> None:
        with open(filepath, "w") as fh:
            for _, row in df.iterrows():
                fh.write(f">{row[id_column]}\n{row[seq_column]}\n")
