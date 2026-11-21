"""Shared visual theme for Aedes Sentinel / Team 3AM.

All colours live here. The CSS is injected once from ``app.py``; ``src/charts.py``
imports ``PALETTE`` so Plotly and the HTML shell stay in sync.
"""
from __future__ import annotations

import streamlit as st

PALETTE = {
    "bg": "#F4F4F6",
    "bg_end": "#FFF3B8",
    "yellow": "#FFD84D",
    "yellow_soft": "#FFF0B3",
    "charcoal": "#222222",
    "ink": "#1A1A1A",
    "muted": "#8A8A8A",
    "line": "rgba(34, 34, 34, 0.08)",
    "card": "rgba(255, 255, 255, 0.70)",
    "card_strong": "rgba(255, 255, 255, 0.86)",
    "low": "#2F7D4A",
    "medium": "#C98A00",
    "high": "#D96A1A",
    "critical": "#B3261E",
}

CSS = r"""
:root {
  --bg: #F4F4F6;
  --bg-warm: #FFF3B8;
  --yellow: #FFD84D;
  --yellow-soft: #FFF0B3;
  --charcoal: #222222;
  --ink: #1A1A1A;
  --muted: #8A8A8A;
  --line: rgba(34, 34, 34, 0.08);
  --card: rgba(255, 255, 255, 0.70);
  --card-strong: rgba(255, 255, 255, 0.86);
  --low: #2F7D4A;
  --medium: #C98A00;
  --high: #D96A1A;
  --critical: #B3261E;
  --shadow: 0 18px 50px rgba(34, 34, 34, 0.10);
  --shadow-soft: 0 8px 26px rgba(34, 34, 34, 0.06);
  --radius: 24px;
}

html, body, [class*="css"] {
  font-family: Inter, "Segoe UI", Roboto, Arial, sans-serif;
}

[data-testid="stAppViewContainer"] {
  background: #E9E9EC;
}

.block-container {
  max-width: 1400px;
  margin: 18px auto 28px;
  padding: 92px 30px 34px;
  background: linear-gradient(155deg, #F7F7F8 0%, #FBF8E8 55%, #FFF2B8 100%);
  border: 1px solid var(--line);
  border-radius: 30px;
  box-shadow: var(--shadow);
  min-height: calc(100vh - 46px);
}

header[data-testid="stHeader"], #MainMenu, footer {
  display: none !important;
}

/* ---------------------------------------------------------------- nav */
.st-key-topnav {
  position: fixed;
  top: 18px;
  left: 50%;
  transform: translateX(-50%);
  width: min(1340px, calc(100vw - 48px));
  z-index: 9999;
  background: rgba(255, 255, 255, 0.94);
  backdrop-filter: blur(18px);
  -webkit-backdrop-filter: blur(18px);
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 7px 10px;
  box-shadow: var(--shadow-soft);
}

.st-key-topnav [data-testid="stHorizontalBlock"] {
  align-items: center;
  gap: 4px;
}

.st-key-topnav [data-testid="stColumn"] {
  display: flex;
  align-items: center;
  justify-content: center;
}

.st-key-topnav [data-testid="stColumn"]:first-child { justify-content: flex-start; }
.st-key-topnav [data-testid="stColumn"]:last-child { justify-content: flex-end; }

.st-key-topnav button[kind="primary"],
.st-key-topnav button[kind="secondary"] {
  border: none !important;
  box-shadow: none !important;
  border-radius: 999px !important;
  padding: 7px 14px !important;
  min-height: 0 !important;
  font-size: 0.8rem !important;
  font-weight: 700 !important;
  transition: background 0.25s ease, color 0.25s ease, transform 0.25s ease;
}

.st-key-topnav button[kind="primary"] {
  background: var(--charcoal) !important;
  color: #fff !important;
}

.st-key-topnav button[kind="secondary"] {
  background: transparent !important;
  color: var(--muted) !important;
}

.st-key-topnav button[kind="primary"] *,
.st-key-topnav button[kind="secondary"] * {
  color: inherit !important;
}

.st-key-topnav button[kind="secondary"]:hover {
  color: var(--ink) !important;
  background: rgba(34, 34, 34, 0.05) !important;
}

.brand {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  font-weight: 850;
  font-size: 1.05rem;
  color: var(--charcoal);
  letter-spacing: -0.03em;
  text-decoration: none !important;
  padding-left: 4px;
  gap: 4px;
}

.brand:hover { color: var(--charcoal); }

.brand-mark {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  background: var(--yellow);
  color: var(--charcoal);
  border-radius: 9px;
  padding: 2px 7px;
  font-weight: 850;
}

.nav-note {
  display: flex;
  justify-content: flex-end;
  align-items: center;
  gap: 7px;
  color: var(--muted);
  font-size: 0.72rem;
  font-weight: 700;
  white-space: nowrap;
  padding-right: 6px;
}

.status-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--low);
  box-shadow: 0 0 0 4px rgba(47, 125, 74, 0.13);
}

/* ---------------------------------------------------------------- home */
.home-hero {
  position: relative;
  overflow: hidden;
  display: grid;
  grid-template-columns: 1.35fr 0.85fr;
  gap: 28px;
  padding: 34px 34px 30px;
  border-radius: 30px;
  border: 1px solid var(--line);
  background: linear-gradient(125deg, rgba(255,255,255,0.80), rgba(255,243,184,0.55));
  background-size: 180% 180%;
  box-shadow: var(--shadow-soft);
  animation: fadeUp 0.7s ease both, gradientShift 18s ease infinite;
}

.home-hero::before {
  content: "";
  position: absolute;
  width: 300px;
  height: 300px;
  right: -90px;
  top: -120px;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(255, 216, 77, 0.70), rgba(255, 216, 77, 0) 70%);
  animation: float 9s ease-in-out infinite;
  pointer-events: none;
}

.home-kicker {
  font-size: 0.70rem;
  font-weight: 800;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--muted);
}

.home-title {
  position: relative;
  z-index: 1;
  font-size: clamp(2.4rem, 4vw, 3.8rem);
  line-height: 1.02;
  letter-spacing: -0.05em;
  font-weight: 850;
  color: var(--ink);
  margin: 12px 0 14px;
}

.home-title .accent {
  background: linear-gradient(transparent 62%, var(--yellow) 62%);
  padding: 0 0.05em;
}

.home-sub {
  position: relative;
  z-index: 1;
  color: #5F5F5F;
  font-size: 1rem;
  line-height: 1.65;
  max-width: 640px;
  margin: 0;
}

.home-hero-right {
  position: relative;
  z-index: 1;
  display: flex;
  align-items: center;
  justify-content: center;
}

.preview-card {
  width: 100%;
  max-width: 330px;
  background: var(--charcoal);
  border-radius: 26px;
  padding: 20px;
  color: #fff;
  box-shadow: 0 24px 55px rgba(34, 34, 34, 0.22);
  animation: float 10s ease-in-out infinite;
}

.preview-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}

.preview-kicker {
  color: rgba(255, 255, 255, 0.52);
  font-size: 0.66rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  font-weight: 800;
}

.preview-chip {
  background: var(--yellow);
  color: var(--charcoal);
  border-radius: 999px;
  padding: 4px 10px;
  font-size: 0.68rem;
  font-weight: 850;
}

.preview-ring {
  width: 138px;
  height: 138px;
  border-radius: 50%;
  margin: 18px auto 16px;
  background: conic-gradient(var(--yellow) 0 67%, rgba(255,255,255,0.14) 67% 100%);
  display: grid;
  place-items: center;
}

.preview-ring-inner {
  width: 104px;
  height: 104px;
  border-radius: 50%;
  background: var(--charcoal);
  display: grid;
  place-items: center;
  text-align: center;
}

.preview-ring-inner b { font-size: 1.7rem; line-height: 1; }
.preview-ring-inner span {
  display: block;
  font-size: 0.58rem;
  color: rgba(255,255,255,0.5);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-top: 5px;
}

.preview-row {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 12px;
  padding: 9px 0;
  border-top: 1px solid rgba(255,255,255,0.10);
  font-size: 0.74rem;
  color: rgba(255,255,255,0.62);
}

.preview-row strong { color: #fff; font-size: 0.78rem; }

.st-key-home_cta { margin: 16px 0 6px; }

.st-key-home_cta button[kind="primary"] {
  background: var(--yellow) !important;
  color: var(--charcoal) !important;
  border: 1px solid rgba(34,34,34,0.10) !important;
  border-radius: 999px !important;
  padding: 12px 20px !important;
  font-weight: 850 !important;
  box-shadow: 0 12px 26px rgba(255,216,77,0.30) !important;
  transition: transform 0.2s ease, box-shadow 0.2s ease, background 0.2s ease;
}

.st-key-home_cta button[kind="primary"]:hover {
  transform: translateY(-2px) scale(1.01);
  box-shadow: 0 16px 32px rgba(255,216,77,0.38) !important;
}

.st-key-home_cta button[kind="secondary"] {
  background: rgba(255,255,255,0.55) !important;
  color: var(--ink) !important;
  border: 1px solid var(--line) !important;
  border-radius: 999px !important;
  padding: 12px 20px !important;
  font-weight: 700 !important;
  transition: transform 0.2s ease, background 0.2s ease;
}

.st-key-home_cta button[kind="secondary"]:hover {
  transform: translateY(-2px);
  background: var(--card-strong) !important;
}

.stat-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 14px;
  margin: 18px 0 8px;
}

.stat-card, .glass-card {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 22px;
  padding: 18px 20px;
  box-shadow: var(--shadow-soft);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  transition: transform 0.25s ease, box-shadow 0.25s ease;
}

.stat-card:hover, .glass-card:hover {
  transform: translateY(-4px);
  box-shadow: var(--shadow);
}

.stat-label {
  font-size: 0.66rem;
  font-weight: 800;
  letter-spacing: 0.11em;
  text-transform: uppercase;
  color: var(--muted);
}

.stat-number {
  font-size: 3.1rem;
  font-weight: 850;
  letter-spacing: -0.06em;
  color: var(--ink);
  line-height: 1.05;
  margin-top: 4px;
}

.stat-number .stat-unit {
  font-size: 0.85rem;
  font-weight: 600;
  color: var(--muted);
  letter-spacing: 0;
  margin-left: 4px;
}

.stat-foot { color: var(--muted); font-size: 0.76rem; margin-top: 2px; }

@property --num {
  syntax: "<integer>";
  initial-value: 0;
  inherits: false;
}

.stat-number.count-60 { --num: 60; --target: 60; animation: count60 1.6s cubic-bezier(0.2, 0.7, 0.2, 1) both; }
.stat-number.count-5 { --num: 5; --target: 5; animation: count5 1.6s cubic-bezier(0.2, 0.7, 0.2, 1) both; }
.stat-number.count-60, .stat-number.count-5 { counter-reset: num var(--num); }
.stat-number.count-60::after, .stat-number.count-5::after { content: counter(num); }

@keyframes count60 { from { --num: 0; } to { --num: 60; } }
@keyframes count5 { from { --num: 0; } to { --num: 5; } }

.home-section { margin-top: 24px; }

.section-kicker {
  font-size: 0.66rem;
  font-weight: 800;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 8px;
}

.flow-row, .flow {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
}

.flow-step {
  background: rgba(255,255,255,0.65);
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 8px 12px;
  font-size: 0.8rem;
  font-weight: 700;
  color: var(--ink);
}

.flow-chip {
  background: rgba(255,255,255,0.65);
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 8px 13px;
  font-size: 0.78rem;
  font-weight: 700;
  color: var(--ink);
}

.flow-arrow { color: var(--muted); font-size: 0.9rem; }

/* -------------------------------------------------------- inner pages */
.page-head { margin: 6px 0 20px; }

.page-kicker {
  font-size: 0.68rem;
  font-weight: 800;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--muted);
}

.page-title {
  font-size: clamp(2rem, 3.6vw, 3.2rem);
  line-height: 1.04;
  letter-spacing: -0.04em;
  font-weight: 850;
  color: var(--ink);
  margin: 10px 0 12px;
}

.page-sub {
  max-width: 820px;
  color: #5F5F5F;
  line-height: 1.65;
  font-size: 0.98rem;
  margin: 0;
}

.stat-strip {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 14px;
  margin: 20px 0;
}

.purpose-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 14px;
  margin-top: 8px;
}

.purpose-card {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 24px;
  padding: 22px;
  box-shadow: var(--shadow-soft);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  transition: transform 0.25s ease, box-shadow 0.25s ease;
}

.purpose-card:hover { transform: translateY(-4px); box-shadow: var(--shadow); }

.purpose-icon {
  width: 38px;
  height: 38px;
  border-radius: 12px;
  background: var(--yellow);
  display: grid;
  place-items: center;
  font-weight: 850;
  color: var(--charcoal);
  margin-bottom: 12px;
  font-size: 0.8rem;
}

.purpose-card h3 { font-size: 1.02rem; margin: 0 0 6px; color: var(--ink); }
.purpose-card p { font-size: 0.86rem; line-height: 1.62; color: #5F5F5F; margin: 0; }

.team-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 14px;
  margin: 8px 0 18px;
}

.member-card {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 24px;
  padding: 20px;
  box-shadow: var(--shadow-soft);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  transition: transform 0.25s ease, box-shadow 0.25s ease;
}

.member-card:hover { transform: translateY(-4px); box-shadow: var(--shadow); }

.avatar {
  width: 52px;
  height: 52px;
  border-radius: 16px;
  background: var(--charcoal);
  color: var(--yellow);
  display: grid;
  place-items: center;
  font-weight: 850;
  font-size: 0.9rem;
  margin-bottom: 12px;
}

.member-name { font-weight: 850; color: var(--ink); font-size: 1rem; }
.member-handle { color: var(--muted); font-size: 0.74rem; margin-top: 2px; }
.member-affiliation { color: var(--muted); font-size: 0.72rem; margin-top: 4px; }
.member-role { color: var(--charcoal); font-size: 0.78rem; font-weight: 700; margin-top: 10px; }
.member-bio { color: #5F5F5F; font-size: 0.82rem; line-height: 1.55; margin-top: 4px; }
.member-links { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 12px; font-size: 0.72rem; }
.member-links a {
  color: var(--ink);
  text-decoration: none;
  border-bottom: 1px solid var(--line);
  padding-bottom: 1px;
  overflow-wrap: anywhere;
}
.member-links a:hover { border-color: var(--charcoal); }

.project-card {
  background: var(--charcoal);
  color: rgba(255,255,255,0.78);
  border-radius: 24px;
  padding: 22px;
  box-shadow: 0 18px 40px rgba(34,34,34,0.18);
}

.project-card h3 { color: #fff; margin: 0 0 8px; font-size: 1.02rem; }
.project-card a { color: var(--yellow); text-decoration: none; font-weight: 700; }
.project-card a:hover { text-decoration: underline; }
.project-card .chip {
  background: rgba(255,255,255,0.10);
  border-color: rgba(255,255,255,0.12);
  color: rgba(255,255,255,0.82);
}
.project-card p { margin: 0; font-size: 0.84rem; line-height: 1.6; color: rgba(255,255,255,0.66); }
.project-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 10px;
  margin-top: 14px;
}
.project-item { border-top: 1px solid rgba(255,255,255,0.10); padding-top: 8px; }
.project-item b { display: block; color: #fff; font-size: 0.9rem; }
.project-item span { color: rgba(255,255,255,0.5); font-size: 0.72rem; }

.chip-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }
.chip {
  background: rgba(255,255,255,0.65);
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 7px 12px;
  font-size: 0.74rem;
  font-weight: 700;
  color: var(--ink);
}

/* ----------------------------------------------------------- dashboard */
.hero {
  position: relative;
  overflow: hidden;
  background: radial-gradient(circle at 82% 18%, rgba(255,216,77,0.28), transparent 38%),
              linear-gradient(135deg, #222222 0%, #2B2B2B 100%);
  color: #fff;
  padding: 24px 28px 22px;
  border-radius: 26px;
  margin-bottom: 18px;
  box-shadow: var(--shadow-soft);
}

.hero::after {
  content: "";
  position: absolute;
  right: -60px;
  bottom: -80px;
  width: 220px;
  height: 220px;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(255,216,77,0.20), transparent 70%);
  pointer-events: none;
}

.hero-kicker {
  letter-spacing: 0.16em;
  font-size: 0.68rem;
  font-weight: 800;
  opacity: 0.58;
  text-transform: uppercase;
}

.hero h1 {
  font-size: 2.0rem;
  line-height: 1.08;
  margin: 8px 0 8px;
  font-weight: 850;
  letter-spacing: -0.03em;
  color: #fff;
}

.hero p {
  max-width: 760px;
  margin: 0;
  font-size: 0.92rem;
  line-height: 1.55;
  color: rgba(255,255,255,0.70);
}

.hero-badges { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }

.hero-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 5px 11px;
  border-radius: 999px;
  font-size: 0.72rem;
  font-weight: 700;
  background: rgba(255,255,255,0.08);
  border: 1px solid rgba(255,255,255,0.14);
  color: rgba(255,255,255,0.86);
}

.hero-badge.live {
  background: rgba(47,125,74,0.28);
  border-color: rgba(129,199,132,0.45);
}

.hero-badge.pending {
  background: rgba(201,138,0,0.22);
  border-color: rgba(255,183,77,0.45);
}

.card-label {
  font-size: 0.66rem;
  font-weight: 800;
  letter-spacing: 0.11em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 2px;
}

.target-line { color: var(--muted); font-size: 0.86rem; margin: 4px 0 14px 2px; }
.target-line b { color: var(--ink); }

.section-title {
  font-size: 1.08rem;
  font-weight: 850;
  color: var(--ink);
  margin: 10px 0 8px 2px;
  letter-spacing: -0.01em;
}

.tier-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 5px 11px;
  border-radius: 999px;
  font-size: 0.76rem;
  font-weight: 750;
}

.risk-cases { font-size: 1.95rem; font-weight: 850; line-height: 1.15; color: var(--ink); margin-top: 2px; }
.risk-cases span { font-size: 0.86rem; font-weight: 500; color: var(--muted); }
.risk-sub { font-size: 0.79rem; color: var(--muted); margin-top: 4px; }

.mini-card {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 14px 16px;
  min-height: 96px;
  box-shadow: 0 1px 2px rgba(34,34,34,0.04);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  transition: transform 0.22s ease, box-shadow 0.22s ease;
}

.mini-card:hover { transform: translateY(-3px); box-shadow: var(--shadow-soft); }

.mini-label {
  font-size: 0.64rem;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--muted);
}

.mini-value { font-size: 1.38rem; font-weight: 850; color: var(--ink); line-height: 1.25; margin-top: 4px; }
.mini-sub { font-size: 0.73rem; color: var(--muted); margin-top: 2px; }

.pending {
  background: #FFF8E6;
  border: 1px dashed #E0B44C;
  border-radius: 12px;
  padding: 14px 16px;
  color: #6B4E00;
  font-size: 0.84rem;
  line-height: 1.5;
}

.note {
  background: rgba(255,255,255,0.55);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 12px 14px;
  color: var(--muted);
  font-size: 0.78rem;
  line-height: 1.5;
}

.st-key-controls_card,
.st-key-risk_card,
.st-key-forecast_card,
.st-key-driver_card,
.st-key-rainfall_card {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 22px;
  padding: 16px 18px;
  box-shadow: var(--shadow-soft);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
}

.st-key-controls_card { padding: 18px 20px; }

.stTabs [role="tablist"] {
  gap: 6px;
  background: rgba(255,255,255,0.55);
  padding: 4px;
  border-radius: 999px;
  width: fit-content;
  border: 1px solid var(--line);
}

.stTabs [role="tab"] {
  border-radius: 999px;
  padding: 7px 13px;
  color: var(--muted);
  font-weight: 700;
  font-size: 0.78rem;
}

.stTabs [role="tab"][aria-selected="true"] {
  background: var(--charcoal);
  color: #fff;
}

.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"],
.stTabs .react-aria-SelectionIndicator {
  display: none !important;
}

[data-testid="stDataFrame"] {
  border-radius: 14px;
  overflow: hidden;
  border: 1px solid var(--line);
}

[data-testid="stMetricValue"] {
  font-weight: 850;
  color: var(--ink);
}

/* -------------------------------------------------------------- footer */
.site-footer {
  margin-top: 32px;
  background: var(--charcoal);
  color: rgba(255,255,255,0.72);
  border-radius: 24px;
  padding: 20px 24px;
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 14px;
}

.footer-brand {
  display: flex;
  align-items: center;
  gap: 10px;
  font-weight: 850;
  color: #fff;
  font-size: 0.94rem;
}

.footer-tag { color: rgba(255,255,255,0.52); font-size: 0.78rem; margin-top: 4px; }
.footer-meta { font-size: 0.74rem; color: rgba(255,255,255,0.48); text-align: right; line-height: 1.6; }
.site-footer a { color: var(--yellow); text-decoration: none; font-weight: 700; }

/* --------------------------------------------------------- animations */
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(14px); }
  to { opacity: 1; transform: none; }
}

@keyframes float {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-10px); }
}

@keyframes gradientShift {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}

.anim { animation: fadeUp 0.6s ease both; }
.delay-1 { animation-delay: 0.08s; }
.delay-2 { animation-delay: 0.16s; }
.delay-3 { animation-delay: 0.24s; }

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation: none !important;
    transition: none !important;
    scroll-behavior: auto !important;
  }
}

/* ---------------------------------------------------------- responsive */
@media (max-width: 1100px) {
  .block-container { margin: 10px; padding: 92px 18px 24px; border-radius: 24px; }
  .st-key-topnav { top: 10px; width: calc(100vw - 24px); }
  .home-hero { grid-template-columns: 1fr; }
  .home-hero-right { max-width: 420px; }
  .purpose-grid { grid-template-columns: 1fr; }
  .nav-note { display: none; }
  .st-key-topnav [data-testid="stColumn"]:last-child { display: none; }
}

@media (max-width: 780px) {
  .home-title { font-size: 2.2rem; }
  .stat-grid, .stat-strip, .project-grid { grid-template-columns: 1fr; }
  .team-grid { grid-template-columns: 1fr; }
  .site-footer { flex-direction: column; align-items: flex-start; }
  .footer-meta { text-align: left; }
  .st-key-topnav [data-testid="stHorizontalBlock"] {
    flex-wrap: nowrap;
    overflow-x: auto;
    gap: 2px;
    scrollbar-width: none;
  }
  .st-key-topnav [data-testid="stHorizontalBlock"]::-webkit-scrollbar { display: none; }
  .st-key-topnav [data-testid="stColumn"] {
    flex: 0 0 auto;
    width: auto;
    min-width: 0;
    justify-content: flex-start;
  }
  .st-key-topnav button[kind="primary"],
  .st-key-topnav button[kind="secondary"] {
    padding: 6px 8px !important;
    font-size: 0.7rem !important;
    white-space: nowrap;
  }
}
"""


def apply() -> None:
    """Inject the global CSS once per page render."""
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
