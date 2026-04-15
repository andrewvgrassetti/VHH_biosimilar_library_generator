"""HTML visualizations for VHH sequences in the Streamlit app."""

from __future__ import annotations

import html

from vhh_library.sequence import IMGT_REGIONS, VHHSequence


class SequenceVisualizer:

    def render_score_bar(self, score: float, label: str, color: str) -> str:
        pct = score * 100
        safe_label = html.escape(label)
        safe_color = html.escape(color)
        return (
            f'<div style="margin:4px 0;font-family:Arial,sans-serif;font-size:13px;">'
            f'<div style="display:flex;justify-content:space-between;">'
            f"<span>{safe_label}</span><span>{score:.3f}</span>"
            f"</div>"
            f'<div style="background:#eee;border-radius:4px;height:14px;overflow:hidden;">'
            f'<div style="width:{pct}%;background:{safe_color};height:100%;border-radius:4px;"></div>'
            f"</div>"
            f"</div>"
        )

    def render_region_track(self, vhh: VHHSequence) -> str:
        blocks: list[str] = []
        for name, (start, end) in IMGT_REGIONS.items():
            bg = "#E3F2FD" if name.startswith("FR") else "#FFCDD2"
            safe_name = html.escape(name)
            subseq = "".join(
                vhh.imgt_numbered.get(str(pos), "") for pos in range(start, end + 1)
            )
            len(subseq)
            blocks.append(
                f'<span style="display:inline-block;background:{bg};'
                f"padding:2px 4px;margin:1px;border-radius:3px;"
                f'font-family:Arial,sans-serif;font-size:12px;"'
                f' title="positions {start}-{end}">'
                f"{safe_name}<br>"
                f'<span style="font-family:Courier New,monospace;font-size:11px;">'
                f"{html.escape(subseq)}</span>"
                f"</span>"
            )
        return (
            f'<div style="font-family:Arial,sans-serif;white-space:nowrap;'
            f'overflow-x:auto;">{"".join(blocks)}</div>'
        )

    def render_alignment(
        self,
        original: VHHSequence,
        mutant_seq: str,
        mutation_info: dict[int, str],
    ) -> str:
        orig_seq = original.sequence
        max_len = max(len(orig_seq), len(mutant_seq))

        orig_chars: list[str] = []
        mut_chars: list[str] = []
        for i in range(max_len):
            o_aa = orig_seq[i] if i < len(orig_seq) else "-"
            m_aa = mutant_seq[i] if i < len(mutant_seq) else "-"
            differs = o_aa != m_aa
            bg = "background:#FFCDD2;" if differs else ""
            orig_chars.append(
                f'<span style="{bg}font-family:Courier New,monospace;">'
                f"{html.escape(o_aa)}</span>"
            )
            mut_chars.append(
                f'<span style="{bg}font-family:Courier New,monospace;">'
                f"{html.escape(m_aa)}</span>"
            )

        return (
            f'<div style="font-family:Courier New,monospace;font-size:13px;'
            f'line-height:1.6;white-space:nowrap;overflow-x:auto;">'
            f'<div><strong>Original:</strong> {"".join(orig_chars)}</div>'
            f'<div><strong>Mutant:&nbsp;&nbsp;</strong> {"".join(mut_chars)}</div>'
            f"</div>"
        )
