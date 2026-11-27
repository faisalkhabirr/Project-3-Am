"""Offline multi-channel dispatch notices.

Pure ``string``/HTML templates; no LLM, no network, no templating engine. The
same values produce the same SMS, WhatsApp message and printable brief, so a
judge can unplug the network and still export a complete commissioner packet.
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path

from .. import config

SUPPLY_KEYS = ("saline_bags_needed", "diagnostic_kits_needed", "hospital_beds_required")
SHORT = {"saline_bags_needed": "Saline", "diagnostic_kits_needed": "Kits", "hospital_beds_required": "Beds"}
SMS_SHORT = {"saline_bags_needed": "S", "diagnostic_kits_needed": "K", "hospital_beds_required": "B"}


@dataclass(frozen=True)
class DispatchNotice:
    ward_id: str
    city: str
    target_date: str
    predicted_cases: float
    tier: str
    required: dict[str, int]
    allocated: dict[str, int]
    crews_required: int
    crews_allocated: int
    overflow: dict[str, int]
    sms: str
    whatsapp: str
    html: str
    preview: str

    @property
    def deficits(self) -> dict[str, int]:
        return {key: max(0, self.required[key] - self.allocated[key]) for key in SUPPLY_KEYS}


def _fmt(value: float) -> str:
    return f"{value:.1f}" if abs(value - round(value)) > 0.05 else str(int(round(value)))


def _deficit_flag(required: int, allocated: int) -> str:
    return "DEFICIT" if allocated < required else "OK"


def _sms(
    *,
    ward_id: str,
    predicted_cases: float,
    required: dict[str, int],
    allocated: dict[str, int],
    crews_required: int,
    crews_allocated: int,
    overflow: dict[str, int],
    **_: object,
) -> str:
    short_ward = str(ward_id).replace("_Ward_", " W").replace("Ward_", "W")
    parts = [f"{short_ward} {_fmt(predicted_cases)}c"]
    for key in SUPPLY_KEYS:
        parts.append(f"{SMS_SHORT[key]}{allocated[key]}/{required[key]}")
    parts.append(f"C{crews_allocated}/{crews_required}")
    deficits = [SHORT[k] for k in SUPPLY_KEYS if allocated[k] < required[k]]
    if crews_allocated < crews_required:
        deficits.append("Crews")
    if deficits:
        parts.append("DEF:" + "/".join(deficits))
    if overflow:
        parts.append("OVF:" + str(sum(overflow.values())))
    message = " | ".join(parts)
    if len(message) > 160:
        # Keep the clinical essentials; drop the overflow tail first.
        message = " | ".join(parts[:-1]) if overflow else message
        message = message[:157].rstrip(" |") + "..."
    return message


def _whatsapp(
    *,
    ward_id: str,
    city: str,
    target_date: str,
    predicted_cases: float,
    tier: str,
    required: dict[str, int],
    allocated: dict[str, int],
    crews_required: int,
    crews_allocated: int,
    overflow: dict[str, int],
    notes: str,
) -> str:
    lines = [
        f"🦟 *Dengue dispatch — {city} · {str(ward_id).replace('_', ' ')}*",
        f"Week of {target_date} · *{tier}* · {_fmt(predicted_cases)} expected cases",
        "",
    ]
    for key in SUPPLY_KEYS:
        gap = required[key] - allocated[key]
        flag = f"⚠️ deficit {gap}" if gap > 0 else "✅"
        lines.append(f"{SHORT[key]}: {allocated[key]} / {required[key]} {flag}")
    lines.append(
        f"Crews: {crews_allocated} / {crews_required} assigned"
        + (" ⚠️ deficit" if crews_allocated < crews_required else " ✅")
    )
    if overflow:
        routes = ", ".join(f"{patients} → {ward.replace('_', ' ')}" for ward, patients in overflow.items())
        lines.append(f"Overflow: {routes}")
    else:
        lines.append("Overflow: none · city headroom available")
    if notes:
        lines.extend(["", notes])
    lines.extend(["", "Reply ACK to confirm receipt. Offline-generated · Aedes Sentinel."])
    return "\n".join(lines)


def _html(
    *,
    ward_id: str,
    city: str,
    target_date: str,
    predicted_cases: float,
    tier: str,
    required: dict[str, int],
    allocated: dict[str, int],
    crews_required: int,
    crews_allocated: int,
    overflow: dict[str, int],
    notes: str,
) -> str:
    e = html.escape
    rows = []
    for key in SUPPLY_KEYS:
        gap = required[key] - allocated[key]
        status = f"Deficit {gap}" if gap > 0 else "Covered"
        colour = "#b3261e" if gap > 0 else "#2f7d4a"
        rows.append(
            "<tr>"
            f"<td>{e(SHORT[key])}</td>"
            f"<td style='text-align:right'>{allocated[key]}</td>"
            f"<td style='text-align:right'>{required[key]}</td>"
            f"<td style='color:{colour};font-weight:600'>{e(status)}</td>"
            "</tr>"
        )
    overflow_html = (
        "<ul>"
        + "".join(
            f"<li>{patients} patient(s) → {e(ward.replace('_', ' '))}</li>"
            for ward, patients in overflow.items()
        )
        + "</ul>"
        if overflow
        else "<p>No overflow required within the city pool.</p>"
    )
    tier_colour = config.TIER_COLORS.get(tier, "#334155")
    note_html = f"<p><strong>Commissioner note:</strong> {e(notes)}</p>" if notes else ""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Dengue dispatch — {e(str(ward_id))}</title>
<style>
  @page {{ size: A4; margin: 16mm; }}
  body {{ font-family: Georgia, 'Times New Roman', serif; color: #17221c; line-height: 1.45; }}
  .kicker {{ font-family: Arial, sans-serif; letter-spacing: .14em; text-transform: uppercase;
             font-size: 11px; color: #5b6b60; }}
  h1 {{ font-size: 26px; margin: 4px 0 2px; }}
  .tier {{ display: inline-block; padding: 4px 10px; border-radius: 999px; color: white;
           background: {tier_colour}; font-family: Arial, sans-serif; font-size: 12px; }}
  .grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin: 18px 0; }}
  .card {{ border: 1px solid #d8e0da; border-radius: 8px; padding: 10px 12px; }}
  .label {{ font-family: Arial, sans-serif; font-size: 10px; letter-spacing: .08em;
            text-transform: uppercase; color: #5b6b60; }}
  .value {{ font-size: 20px; font-weight: 700; }}
  table {{ width: 100%; border-collapse: collapse; margin: 10px 0 18px; }}
  th, td {{ border-bottom: 1px solid #e2e8e3; padding: 7px 6px; font-family: Arial, sans-serif;
            font-size: 13px; }}
  th {{ text-align: left; color: #5b6b60; font-size: 11px; text-transform: uppercase; }}
  .footer {{ margin-top: 24px; padding-top: 10px; border-top: 1px solid #d8e0da;
             font-family: Arial, sans-serif; font-size: 11px; color: #5b6b60; }}
  @media print {{ .no-print {{ display: none; }} }}
</style>
</head>
<body>
  <div class="kicker">Aedes Sentinel · Team 3AM · field dispatch</div>
  <h1>{e(city)} · {e(str(ward_id).replace('_', ' '))}</h1>
  <p>Forecast week of <strong>{e(target_date)}</strong> &nbsp; <span class="tier">{e(tier)}</span></p>

  <div class="grid">
    <div class="card"><div class="label">Predicted cases</div>
      <div class="value">{_fmt(predicted_cases)}</div></div>
    <div class="card"><div class="label">Crews assigned</div>
      <div class="value">{crews_allocated} / {crews_required}</div></div>
    <div class="card"><div class="label">Bed supply gap</div>
      <div class="value">{max(0, required['hospital_beds_required'] - allocated['hospital_beds_required'])}</div></div>
  </div>

  <h2 style="font-size:17px;margin-bottom:4px">Supply position</h2>
  <table>
    <thead><tr><th>Commodity</th><th style="text-align:right">Allocated</th>
      <th style="text-align:right">Required</th><th>Status</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>

  <h2 style="font-size:17px;margin-bottom:4px">Hospital overflow</h2>
  {overflow_html}
  {note_html}

  <p class="footer">Generated locally by Aedes Sentinel · no external API · decision support only.
  Verify against the ward register before dispatch.</p>
  <button class="no-print" onclick="window.print()">Print this brief</button>
</body>
</html>"""


def _preview(
    *,
    ward_id: str,
    city: str,
    target_date: str,
    predicted_cases: float,
    tier: str,
    required: dict[str, int],
    allocated: dict[str, int],
    crews_required: int,
    crews_allocated: int,
    overflow: dict[str, int],
    notes: str,
) -> str:
    """Inline-styled preview fragment for st.html (no global CSS, no iframe)."""
    e = html.escape
    tier_colour = config.TIER_COLORS.get(tier, "#334155")
    rows = []
    for key in SUPPLY_KEYS:
        gap = required[key] - allocated[key]
        status = f"Deficit {gap}" if gap > 0 else "Covered"
        colour = "#b3261e" if gap > 0 else "#2f7d4a"
        rows.append(
            "<tr>"
            f"<td style='padding:6px 4px;border-bottom:1px solid #e2e8e3'>{e(SHORT[key])}</td>"
            f"<td style='padding:6px 4px;border-bottom:1px solid #e2e8e3;text-align:right'>"
            f"{allocated[key]} / {required[key]}</td>"
            f"<td style='padding:6px 4px;border-bottom:1px solid #e2e8e3;color:{colour};"
            f"font-weight:600'>{e(status)}</td>"
            "</tr>"
        )
    overflow_text = (
        ", ".join(
            f"{patients} → {ward.replace('_', ' ')}"
            for ward, patients in overflow.items()
        )
        if overflow
        else "None · city headroom available"
    )
    note = (
        f"<p style='margin:10px 0 0;font-size:12px;color:#5b6b60'><strong>Note:</strong> "
        f"{e(notes)}</p>"
        if notes
        else ""
    )
    return f"""
<div style="border:1px solid #d8e0da;border-radius:14px;padding:18px 20px;background:#fff;
            color:#17221c;font-family:Inter,Arial,sans-serif;line-height:1.45">
  <div style="font-size:10px;letter-spacing:.14em;text-transform:uppercase;color:#5b6b60;
              font-weight:700">Aedes Sentinel · brief preview</div>
  <div style="font-size:22px;font-weight:800;margin:4px 0 2px">
    {e(city)} · {e(str(ward_id).replace('_', ' '))}
  </div>
  <div style="font-size:13px;color:#5b6b60">
    Week of <strong>{e(target_date)}</strong> ·
    <span style="background:{tier_colour};color:#fff;border-radius:999px;padding:2px 9px;
                 font-size:11px;font-weight:700">{e(tier)}</span> ·
    <strong>{_fmt(predicted_cases)}</strong> expected cases
  </div>
  <table style="width:100%;border-collapse:collapse;margin:14px 0 8px;font-size:13px">
    <thead><tr>
      <th style="text-align:left;font-size:10px;text-transform:uppercase;color:#5b6b60;
                 padding:4px">Commodity</th>
      <th style="text-align:right;font-size:10px;text-transform:uppercase;color:#5b6b60;
                 padding:4px">Got / need</th>
      <th style="text-align:left;font-size:10px;text-transform:uppercase;color:#5b6b60;
                 padding:4px">Status</th>
    </tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:12px 0">
    <div style="border:1px solid #e2e8e3;border-radius:8px;padding:8px 10px">
      <div style="font-size:10px;text-transform:uppercase;color:#5b6b60">Crews</div>
      <div style="font-size:18px;font-weight:800">{crews_allocated} / {crews_required}</div>
    </div>
    <div style="border:1px solid #e2e8e3;border-radius:8px;padding:8px 10px">
      <div style="font-size:10px;text-transform:uppercase;color:#5b6b60">Overflow</div>
      <div style="font-size:12px;font-weight:600">{e(overflow_text)}</div>
    </div>
  </div>
  {note}
  <p style="margin:12px 0 0;font-size:11px;color:#5b6b60">
    Use <strong>Download printable brief</strong> for the print / PDF packet.
  </p>
</div>
"""


def build_notice(
    *,
    ward_id: str,
    city: str,
    target_date: str,
    predicted_cases: float,
    tier: str,
    required: dict[str, int],
    allocated: dict[str, int],
    crews_required: int,
    crews_allocated: int,
    overflow: dict[str, int] | None = None,
    notes: str = "",
) -> DispatchNotice:
    """Build all three channels from one validated set of numbers."""
    overflow = {str(k): int(v) for k, v in (overflow or {}).items() if int(v) > 0}
    required = {key: int(required[key]) for key in SUPPLY_KEYS}
    allocated = {key: int(allocated[key]) for key in SUPPLY_KEYS}
    common = {
        "ward_id": str(ward_id),
        "city": str(city),
        "target_date": str(target_date),
        "predicted_cases": float(predicted_cases),
        "tier": str(tier),
        "required": required,
        "allocated": allocated,
        "crews_required": int(crews_required),
        "crews_allocated": int(crews_allocated),
        "overflow": overflow,
        "notes": notes,
    }
    return DispatchNotice(
        **{key: common[key] for key in (
            "ward_id", "city", "target_date", "predicted_cases", "tier",
            "required", "allocated", "crews_required", "crews_allocated", "overflow",
        )},
        sms=_sms(**common),
        whatsapp=_whatsapp(**common),
        html=_html(**common),
        preview=_preview(**common),
    )


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-") or "ward"


def write_exports(notices: list[DispatchNotice], out_dir: Path) -> list[Path]:
    """Write .txt/.html packet files; return the created paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for notice in notices:
        stem = f"{_slug(notice.ward_id)}_{_slug(notice.target_date)}"
        sms_path = out_dir / f"{stem}_sms.txt"
        wa_path = out_dir / f"{stem}_whatsapp.txt"
        html_path = out_dir / f"{stem}_brief.html"
        sms_path.write_text(notice.sms + "\n", encoding="utf-8")
        wa_path.write_text(notice.whatsapp + "\n", encoding="utf-8")
        html_path.write_text(notice.html, encoding="utf-8")
        written.extend([sms_path, wa_path, html_path])
    return written
