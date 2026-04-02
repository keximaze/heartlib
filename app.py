from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import gradio as gr
import soundfile as sf
import torch

from heartlib import HeartMuLaGenPipeline, HeartTranscriptorPipeline


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = ROOT_DIR / "ckpt"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "assets" / "gui_outputs"
DEFAULT_LYRICS_PATH = ROOT_DIR / "assets" / "lyrics.txt"
DEFAULT_TAGS_PATH = ROOT_DIR / "assets" / "tags.txt"
GUI_STATE_DIR = ROOT_DIR / "assets" / "gui_state"
GUI_STATE_DIR.mkdir(parents=True, exist_ok=True)
STUDIO_PREFERENCES_PATH = GUI_STATE_DIR / "studio_preferences.json"
APP_HARVEST_DIR = ROOT_DIR / "training" / "experiments" / "app_harvest_v1"
APP_HARVEST_SELECTED_PATH = APP_HARVEST_DIR / "selected.jsonl"
APP_HARVEST_REPORT_PATH = APP_HARVEST_DIR / "report.md"
APP_HARVEST_DATASET_DIR = ROOT_DIR / "training" / "experiments" / "app_harvest_v1_dataset"
APP_HARVEST_MANIFEST_PATH = APP_HARVEST_DATASET_DIR / "manifest.jsonl"
APP_EXPERIMENT_DIR = ROOT_DIR / "training" / "experiments" / "app_experiment_v1"
APP_EXPERIMENT_MANIFEST_PATH = APP_EXPERIMENT_DIR / "manifest.jsonl"
APP_EXPERIMENT_SMOKE_PATH = APP_EXPERIMENT_DIR / "manifest_smoke.jsonl"
APP_EXPERIMENT_FORWARD_SUMMARY_PATH = APP_EXPERIMENT_DIR / "forward_only_summary.json"
APP_EXPERIMENT_SMOKE_CHECKPOINT_DIR = ROOT_DIR / "training" / "checkpoints" / "app_experiment_v1_smoke"
APP_EXPERIMENT_SMOKE_SUMMARY_PATH = APP_EXPERIMENT_DIR / "train_smoke_summary.json"
APP_EXPERIMENT_SMOKE_EVAL_JSON_PATH = APP_EXPERIMENT_DIR / "checkpoint_eval_smoke.json"
APP_EXPERIMENT_SMOKE_EVAL_MD_PATH = APP_EXPERIMENT_DIR / "checkpoint_eval_smoke.md"
APP_EXPERIMENT_TRAIN_MANIFEST_PATH = APP_EXPERIMENT_DIR / "manifest_train.jsonl"
APP_EXPERIMENT_VALIDATION_MANIFEST_PATH = APP_EXPERIMENT_DIR / "manifest_validation.jsonl"
APP_EXPERIMENT_CLEAN_CHECKPOINT_DIR = ROOT_DIR / "training" / "checkpoints" / "app_experiment_v1_cycle1"
APP_EXPERIMENT_CLEAN_SUMMARY_PATH = APP_EXPERIMENT_DIR / "cycle1_train_summary.json"
APP_EXPERIMENT_CLEAN_EVAL_JSON_PATH = APP_EXPERIMENT_DIR / "checkpoint_eval_cycle1.json"
APP_EXPERIMENT_CLEAN_EVAL_MD_PATH = APP_EXPERIMENT_DIR / "checkpoint_eval_cycle1.md"
MIN_APP_EXPERIMENT_ROWS = 6
MIN_APP_TRAIN_ROWS = 4
MIN_APP_VALIDATION_ROWS = 2
MIN_APP_HARVEST_QUALITY = 60
LYRICS_DRAFT_PATH = GUI_STATE_DIR / "lyrics_draft.txt"
TAGS_DRAFT_PATH = GUI_STATE_DIR / "tags_draft.txt"
RENDER_HISTORY_PATH = GUI_STATE_DIR / "render_history.json"
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RENDER_STATE_LOCK = threading.RLock()
RENDER_EXECUTION_LOCK = threading.Lock()
RENDER_QUEUE_WORKER: threading.Thread | None = None
RENDER_HISTORY_LIMIT = 24
RENDER_QUEUE_POLL_SECONDS = 1.5

GENERATION_PIPELINES: Dict[
    Tuple[str, str, str, str, str, bool], HeartMuLaGenPipeline
] = {}
TRANSCRIPTION_PIPELINES: Dict[
    Tuple[str, str, str], HeartTranscriptorPipeline
] = {}

CUSTOM_CSS = """
html, body {
  min-height: 100%;
  background:
    radial-gradient(circle at 12% 10%, rgba(99, 102, 241, 0.18), transparent 30%),
    radial-gradient(circle at 88% 14%, rgba(45, 212, 191, 0.16), transparent 24%),
    radial-gradient(circle at 18% 100%, rgba(56, 189, 248, 0.14), transparent 28%),
    linear-gradient(180deg, rgba(245, 248, 253, 0.92) 0%, rgba(236, 241, 248, 0.82) 46%, rgba(231, 237, 246, 0.78) 100%);
}
body::before,
body::after {
  content: "";
  position: fixed;
  inset: auto;
  width: 320px;
  height: 320px;
  border-radius: 999px;
  filter: blur(80px);
  z-index: 0;
  pointer-events: none;
}
body::before {
  top: 48px;
  right: 48px;
  background: rgba(52, 211, 153, 0.22);
}
body::after {
  left: 32px;
  bottom: 48px;
  background: rgba(99, 102, 241, 0.18);
}
.gradio-container {
  width: calc(100vw - 28px) !important;
  max-width: none !important;
  min-height: calc(100vh - 28px) !important;
  margin: 14px auto !important;
  padding: 24px !important;
  box-sizing: border-box !important;
  font-size: 15px !important;
  position: relative;
  z-index: 1;
  background:
    linear-gradient(180deg, rgba(255,255,255,0.26) 0%, rgba(255,255,255,0.14) 100%),
    linear-gradient(135deg, rgba(255,255,255,0.20) 0%, rgba(255,255,255,0.08) 100%) !important;
  backdrop-filter: blur(34px) saturate(150%);
  -webkit-backdrop-filter: blur(34px) saturate(150%);
  border: 1px solid rgba(255, 255, 255, 0.62);
  border-radius: 34px !important;
  box-shadow:
    inset 0 1px 0 rgba(255,255,255,0.58),
    0 28px 90px rgba(15, 23, 42, 0.12),
    0 10px 30px rgba(255, 255, 255, 0.18);
  isolation: isolate;
}
@media (max-width: 960px) {
  .gradio-container {
    width: calc(100vw - 16px) !important;
    min-height: calc(100vh - 16px) !important;
    margin: 8px auto !important;
    padding: 16px !important;
    border-radius: 24px !important;
  }
}
.gradio-container * {
  font-family: "SF Pro Display", "Segoe UI", "Helvetica Neue", sans-serif !important;
}
.gradio-container button[role="tab"] {
  border-radius: 999px !important;
  padding: 11px 18px !important;
  font-size: 15px !important;
  font-weight: 600 !important;
  color: #1f2937 !important;
  background: rgba(255, 255, 255, 0.30) !important;
  border: 1px solid rgba(255, 255, 255, 0.48) !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.62), 0 8px 18px rgba(148,163,184,0.08);
}
.gradio-container button[role="tab"][aria-selected="true"] {
  color: #ffffff !important;
  background: linear-gradient(135deg, rgba(15,23,42,0.92), rgba(37,99,235,0.92)) !important;
  border-color: transparent !important;
  box-shadow: 0 10px 22px rgba(37, 99, 235, 0.22);
}
.gradio-container label,
.gradio-container .label-wrap {
  color: #0f172a !important;
  font-weight: 650 !important;
  font-size: 14px !important;
  line-height: 1.4 !important;
}
.gradio-container input:not([type="range"]):not([type="checkbox"]):not([type="radio"]),
.gradio-container textarea {
  background: rgba(255, 255, 255, 0.34) !important;
  border: 1px solid rgba(255, 255, 255, 0.52) !important;
  border-radius: 18px !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.72), 0 18px 34px rgba(148,163,184,0.08);
  font-size: 15px !important;
}
.gradio-container .wrap,
.gradio-container .gr-group,
.gradio-container .gr-box,
.gradio-container .gr-panel,
.gradio-container .gr-block {
  overflow: visible !important;
}
.gradio-container textarea {
  font-size: 16px !important;
  line-height: 1.62 !important;
}
.gradio-container textarea,
.gradio-container textarea::placeholder,
.gradio-container input:not([type="range"]):not([type="checkbox"]):not([type="radio"]),
.gradio-container input:not([type="range"]):not([type="checkbox"]):not([type="radio"])::placeholder {
  color: #f8fafc !important;
}
.gradio-container textarea {
  background: linear-gradient(180deg, rgba(15, 23, 42, 0.82) 0%, rgba(15, 23, 42, 0.74) 100%) !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.08), 0 18px 32px rgba(15,23,42,0.18);
}
.gradio-container .gradio-dropdown,
.gradio-container [data-testid="dropdown"] {
  overflow: visible !important;
}
.gradio-container [data-testid="dropdown"] button,
.gradio-container [data-testid="dropdown"] .wrap,
.gradio-container [data-testid="dropdown"] .icon-wrap {
  pointer-events: auto !important;
}
.gradio-container .gradio-audio label,
.gradio-container .gradio-file label,
.gradio-container [data-testid="audio"] label,
.gradio-container [data-testid="file"] label {
  color: #ffffff !important;
}
.gradio-container .output-block {
  --block-label-text-color: #f8fafc !important;
  --block-label-background-fill: rgba(51, 65, 85, 0.92) !important;
  --block-label-border-color: rgba(255,255,255,0.14) !important;
  --body-text-color: #f8fafc !important;
  --body-text-color-subdued: rgba(248,250,252,0.84) !important;
  --link-text-color: #f8fafc !important;
  --link-text-color-hover: #ffffff !important;
  --link-text-color-visited: #f8fafc !important;
  --link-text-color-active: #ffffff !important;
}
.gradio-container .output-block [data-testid="block-label"] {
  color: #f8fafc !important;
  background: rgba(51, 65, 85, 0.92) !important;
  border-color: rgba(255,255,255,0.14) !important;
}
.gradio-container .output-block [data-testid="block-label"] *,
.gradio-container .output-block [data-testid="block-label"] span,
.gradio-container .output-block [data-testid="block-label"] svg {
  color: #f8fafc !important;
  fill: #f8fafc !important;
  opacity: 1 !important;
}
.gradio-container .gradio-audio,
.gradio-container .gradio-file,
.gradio-container [data-testid="audio"],
.gradio-container [data-testid="file"] {
  color: #ffffff !important;
  background: rgba(15, 23, 42, 0.88) !important;
  border: 1px solid rgba(255,255,255,0.1) !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.06), 0 16px 32px rgba(15,23,42,0.16);
  --block-label-text-color: #f8fafc !important;
  --block-label-background-fill: rgba(51, 65, 85, 0.92) !important;
  --block-label-border-color: rgba(255,255,255,0.14) !important;
  --body-text-color: #f8fafc !important;
  --body-text-color-subdued: rgba(248,250,252,0.84) !important;
  --link-text-color: #f8fafc !important;
  --link-text-color-hover: #ffffff !important;
  --link-text-color-visited: #f8fafc !important;
  --link-text-color-active: #ffffff !important;
}
.gradio-container .gradio-audio *,
.gradio-container .gradio-file *,
.gradio-container [data-testid="audio"] *,
.gradio-container [data-testid="file"] * {
  color: #ffffff !important;
}
.gradio-container .gradio-audio :is(label, legend, span, p, strong, button),
.gradio-container .gradio-file :is(label, legend, span, p, strong, button),
.gradio-container [data-testid="audio"] :is(label, legend, span, p, strong, button),
.gradio-container [data-testid="file"] :is(label, legend, span, p, strong, button) {
  color: #f8fafc !important;
  opacity: 1 !important;
}
.gradio-container .gradio-audio .label-wrap,
.gradio-container .gradio-file .label-wrap,
.gradio-container [data-testid="audio"] .label-wrap,
.gradio-container [data-testid="file"] .label-wrap,
.gradio-container .gradio-audio [data-testid="block-label"],
.gradio-container .gradio-file [data-testid="block-label"],
.gradio-container [data-testid="audio"] [data-testid="block-label"],
.gradio-container [data-testid="file"] [data-testid="block-label"],
.gradio-container .gradio-audio legend,
.gradio-container .gradio-file legend,
.gradio-container [data-testid="audio"] legend,
.gradio-container [data-testid="file"] legend {
  color: #f8fafc !important;
  background: rgba(51, 65, 85, 0.92) !important;
  border-color: rgba(255,255,255,0.14) !important;
  opacity: 1 !important;
  text-shadow: 0 1px 0 rgba(15,23,42,0.24);
}
.gradio-container .gradio-audio [data-testid="block-label"] *,
.gradio-container .gradio-file [data-testid="block-label"] *,
.gradio-container [data-testid="audio"] [data-testid="block-label"] *,
.gradio-container [data-testid="file"] [data-testid="block-label"] * {
  color: #f8fafc !important;
  fill: #f8fafc !important;
  opacity: 1 !important;
}
.gradio-container .gradio-audio > :first-child,
.gradio-container .gradio-file > :first-child,
.gradio-container [data-testid="audio"] > :first-child,
.gradio-container [data-testid="file"] > :first-child {
  color: #f8fafc !important;
}
.gradio-container .gradio-audio > :first-child :is(span, label, p, button, svg),
.gradio-container .gradio-file > :first-child :is(span, label, p, button, svg),
.gradio-container [data-testid="audio"] > :first-child :is(span, label, p, button, svg),
.gradio-container [data-testid="file"] > :first-child :is(span, label, p, button, svg) {
  color: #f8fafc !important;
  fill: #f8fafc !important;
  opacity: 1 !important;
}
.gradio-container .gradio-audio svg,
.gradio-container .gradio-file svg,
.gradio-container [data-testid="audio"] svg,
.gradio-container [data-testid="file"] svg {
  color: #f8fafc !important;
  fill: #f8fafc !important;
}
.gradio-container .gradio-audio .wrap,
.gradio-container .gradio-file .wrap,
.gradio-container [data-testid="audio"] .wrap,
.gradio-container [data-testid="file"] .wrap,
.gradio-container .gradio-audio .or,
.gradio-container .gradio-file .or,
.gradio-container [data-testid="audio"] .or,
.gradio-container [data-testid="file"] .or {
  color: #f8fafc !important;
}
.gradio-container footer,
.gradio-container footer * {
  color: #334155 !important;
  opacity: 1 !important;
  text-shadow: none !important;
}
.gradio-container footer a {
  color: #1e293b !important;
}
.gradio-container .block,
.gradio-container .gr-box,
.gradio-container .gr-form,
.gradio-container .form,
.gradio-container .panel {
  border-radius: 22px !important;
}
.gradio-container .secondary,
.gradio-container .gr-button-secondary {
  background: linear-gradient(180deg, rgba(255,255,255,0.34) 0%, rgba(255,255,255,0.24) 100%) !important;
  color: #0f172a !important;
  border: 1px solid rgba(255,255,255,0.52) !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.7), 0 14px 24px rgba(148,163,184,0.10);
  font-size: 15px !important;
  font-weight: 650 !important;
  min-height: 46px !important;
}
.gradio-container .primary,
.gradio-container .gr-button-primary {
  background: linear-gradient(135deg, #0f172a 0%, #2563eb 55%, #38bdf8 100%) !important;
  color: #ffffff !important;
  border: 0 !important;
  box-shadow: 0 16px 28px rgba(37, 99, 235, 0.28);
  font-size: 16px !important;
  font-weight: 700 !important;
  min-height: 48px !important;
}
.gradio-container .primary:hover,
.gradio-container .gr-button-primary:hover {
  filter: brightness(1.03);
}
.gradio-container .gr-button,
.gradio-container button:not([role="tab"]) {
  letter-spacing: -0.01em !important;
}
.gradio-container [data-testid="dropdown"] button,
.gradio-container [data-testid="dropdown"] input,
.gradio-container [data-testid="dropdown"] span {
  font-size: 15px !important;
}
.gradio-container [data-testid="textbox"] textarea,
.gradio-container [data-testid="textbox"] input {
  font-size: 16px !important;
}
.gradio-container [data-testid="textbox"] textarea::placeholder,
.gradio-container [data-testid="textbox"] input::placeholder {
  font-size: 15px !important;
}
.gradio-container [data-testid="slider"] * {
  font-size: 14px !important;
}
.gradio-container [data-testid="checkbox"] label,
.gradio-container [data-testid="checkbox-group"] label {
  font-size: 14px !important;
  font-weight: 600 !important;
}
.gradio-container .output-block,
.gradio-container .output-block * {
  font-size: 15px !important;
}
.workspace-shell {
  gap: 20px;
  align-items: flex-start;
}
.workspace-main {
  gap: 16px;
}
.sidebar-rail {
  position: sticky !important;
  top: 16px;
  gap: 12px;
}
.sidebar-brand-card {
  padding: 18px;
  border-radius: 26px;
  background: linear-gradient(180deg, rgba(15,23,42,0.82) 0%, rgba(37,99,235,0.62) 100%);
  border: 1px solid rgba(255,255,255,0.18);
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.12), 0 22px 42px rgba(15,23,42,0.18);
  color: #f8fafc;
}
.sidebar-brand-top {
  margin: 0 0 10px 0;
  font-size: 12px;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  opacity: 0.72;
}
.sidebar-brand-title {
  margin: 0;
  font-size: 34px;
  line-height: 1;
  letter-spacing: -0.04em;
}
.sidebar-brand-copy {
  margin: 10px 0 0 0;
  font-size: 14px;
  line-height: 1.55;
  color: rgba(248,250,252,0.88);
}
.sidebar-nav-stack {
  display: grid;
  gap: 10px;
}
.sidebar-nav-button button {
  width: 100%;
  justify-content: flex-start !important;
  min-height: 54px !important;
  border-radius: 18px !important;
  padding-inline: 18px !important;
  font-size: 15px !important;
  font-weight: 700 !important;
  transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}
.sidebar-nav-button button:hover {
  transform: translateX(3px) !important;
  box-shadow: 0 8px 20px rgba(37,99,235,0.12), inset 0 1px 0 rgba(255,255,255,0.72) !important;
}
.sidebar-footnote {
  padding: 16px;
  border-radius: 22px;
  background: linear-gradient(180deg, rgba(255,255,255,0.20) 0%, rgba(255,255,255,0.12) 100%);
  border: 1px solid rgba(255,255,255,0.56);
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.58), 0 16px 30px rgba(148,163,184,0.10);
  color: #334155;
  font-size: 13px;
  line-height: 1.5;
}
.workspace-intro {
  padding: 18px 20px;
  border-radius: 24px;
  background: linear-gradient(180deg, rgba(255,255,255,0.22) 0%, rgba(255,255,255,0.14) 100%);
  border: 1px solid rgba(255,255,255,0.60);
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.62), 0 18px 36px rgba(148,163,184,0.10);
  backdrop-filter: blur(24px) saturate(145%);
  -webkit-backdrop-filter: blur(24px) saturate(145%);
}
.workspace-intro-top {
  margin: 0 0 8px 0;
  font-size: 12px;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: #64748b;
}
.workspace-intro h2 {
  margin: 0;
  font-size: 30px;
  letter-spacing: -0.03em;
  color: #0f172a;
}
.workspace-intro p {
  margin: 8px 0 0 0;
  color: #475569;
  font-size: 15px;
  line-height: 1.55;
  max-width: 760px;
}
.workspace-view {
  gap: 16px;
}
.workspace-panel {
  background: linear-gradient(145deg, rgba(255,255,255,0.24) 0%, rgba(255,255,255,0.14) 100%);
  border: 1px solid rgba(255,255,255,0.62);
  border-radius: 26px;
  padding: 20px;
  box-shadow:
    inset 0 1px 0 rgba(255,255,255,0.64),
    0 4px 8px rgba(148,163,184,0.06),
    0 20px 40px rgba(148,163,184,0.11);
  backdrop-filter: blur(28px) saturate(148%);
  -webkit-backdrop-filter: blur(28px) saturate(148%);
  transition: box-shadow 0.2s ease;
}
.workspace-panel:hover {
  box-shadow:
    inset 0 1px 0 rgba(255,255,255,0.72),
    0 8px 20px rgba(148,163,184,0.10),
    0 28px 52px rgba(148,163,184,0.14);
}
.workspace-panel-title {
  margin: 0 0 6px 0;
  color: #0f172a;
  font-size: 22px;
  letter-spacing: -0.02em;
}
.workspace-panel-copy {
  margin: 0;
  color: #475569;
  font-size: 14px;
  line-height: 1.5;
}
@media (max-width: 1180px) {
  .workspace-shell {
    flex-direction: column;
  }
  .sidebar-rail {
    position: static !important;
  }
}
.hero {
  background:
    linear-gradient(135deg, rgba(15, 23, 42, 0.76) 0%, rgba(37, 99, 235, 0.62) 42%, rgba(16, 185, 129, 0.50) 100%),
    rgba(255,255,255,0.10);
  color: #f8fafc;
  padding: 34px 38px;
  border-radius: 28px;
  margin-bottom: 18px;
  border: 1px solid rgba(255,255,255,0.30);
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.16), 0 28px 60px rgba(15, 23, 42, 0.16);
  backdrop-filter: blur(28px) saturate(150%);
  -webkit-backdrop-filter: blur(28px) saturate(150%);
}
.hero-topline {
  margin: 0 0 10px 0;
  font-size: 13px;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  opacity: 0.72;
}
.hero h1 { margin: 0 0 10px 0; font-size: clamp(42px, 5vw, 64px); letter-spacing: -0.04em; }
.hero p { margin: 0; opacity: 0.94; max-width: 820px; font-size: 20px; line-height: 1.55; }
.hero-stats { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 18px; }
.hero-stat {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 10px 14px;
  border-radius: 999px;
  background: rgba(255,255,255,0.16);
  border: 1px solid rgba(255,255,255,0.24);
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.22);
  font-size: 14px;
  font-weight: 600;
}
.panel-note {
  background: linear-gradient(180deg, rgba(255,255,255,0.22) 0%, rgba(255,255,255,0.16) 100%);
  border: 1px solid rgba(255, 255, 255, 0.58);
  border-radius: 18px;
  padding: 14px 16px;
  color: #334155;
  font-size: 14px;
  line-height: 1.5;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.62), 0 16px 30px rgba(148,163,184,0.08);
  backdrop-filter: blur(24px) saturate(150%);
  -webkit-backdrop-filter: blur(24px) saturate(150%);
}
.advanced-accordion {
  margin: 12px 0 14px 0 !important;
  border-radius: 22px !important;
  border: 1px solid rgba(255,255,255,0.58) !important;
  background: linear-gradient(180deg, rgba(255,255,255,0.20) 0%, rgba(255,255,255,0.12) 100%) !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.54), 0 16px 30px rgba(148,163,184,0.08);
  overflow: hidden !important;
}
.advanced-accordion label,
.advanced-accordion .label-wrap {
  color: #1e293b !important;
}
.sticky-player-dock {
  position: relative !important;
  width: 100%;
  display: grid !important;
  grid-template-columns: minmax(280px, 360px) minmax(0, 1fr);
  gap: 14px;
  align-items: center;
  margin-top: 16px;
  padding: 14px 16px !important;
  border-radius: 26px !important;
  background: linear-gradient(180deg, rgba(15, 23, 42, 0.80) 0%, rgba(30, 41, 59, 0.72) 100%) !important;
  border: 1px solid rgba(255,255,255,0.14) !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.10), 0 24px 56px rgba(15,23,42,0.24);
  backdrop-filter: blur(26px) saturate(135%);
  -webkit-backdrop-filter: blur(26px) saturate(135%);
}
.sticky-player-info-shell {
  display: flex;
  align-items: center;
  gap: 14px;
  min-width: 0;
}
.sticky-player-cover {
  width: 64px;
  height: 64px;
  flex: 0 0 64px;
  border-radius: 18px;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.1), 0 14px 24px rgba(15,23,42,0.18);
}
.sticky-player-copy {
  min-width: 0;
}
.pick-pill-row {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 8px;
}
.pick-pill {
  display: inline-flex;
  align-items: center;
  padding: 4px 9px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.02em;
  border: 1px solid transparent;
}
.pick-pill--best {
  background: rgba(59,130,246,0.14);
  color: #dbeafe;
  border-color: rgba(147,197,253,0.28);
}
.pick-pill--favorite {
  background: rgba(244,114,182,0.14);
  color: #fce7f3;
  border-color: rgba(244,114,182,0.26);
}
.pick-pill--training {
  background: rgba(56,189,248,0.16);
  color: #e0f2fe;
  border-color: rgba(125,211,252,0.28);
}
.pick-pill--archived {
  background: rgba(148,163,184,0.18);
  color: #0f172a;
  border-color: rgba(100,116,139,0.24);
}
.sticky-player-copy .pick-pill--archived,
.candidate-copy .pick-pill--archived {
  color: #e2e8f0;
  background: rgba(148,163,184,0.18);
  border-color: rgba(226,232,240,0.18);
}
.sticky-player-eyebrow {
  margin: 0 0 4px 0;
  color: rgba(226,232,240,0.76);
  font-size: 12px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}
.sticky-player-title {
  margin: 0;
  color: #f8fafc;
  font-size: 20px;
  line-height: 1.2;
  letter-spacing: -0.03em;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.sticky-player-sub {
  margin: 6px 0 0 0;
  color: rgba(226,232,240,0.82);
  font-size: 14px;
  line-height: 1.4;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.sticky-player-audio,
.sticky-player-audio [data-testid="audio"] {
  background: rgba(255,255,255,0.06) !important;
  border: 1px solid rgba(255,255,255,0.08) !important;
  box-shadow: none !important;
}
.sticky-player-audio [data-testid="block-label"] {
  display: none !important;
}
.candidate-shell {
  margin-top: 14px;
  padding: 16px;
  border-radius: 24px;
  background: linear-gradient(180deg, rgba(255,255,255,0.22) 0%, rgba(255,255,255,0.14) 100%);
  border: 1px solid rgba(255,255,255,0.56);
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.58), 0 18px 32px rgba(148,163,184,0.09);
  backdrop-filter: blur(24px) saturate(140%);
  -webkit-backdrop-filter: blur(24px) saturate(140%);
}
.candidate-header {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  align-items: baseline;
  margin-bottom: 12px;
}
.candidate-title-main {
  margin: 0;
  color: #0f172a;
  font-size: 20px;
  letter-spacing: -0.03em;
}
.candidate-sub {
  margin: 4px 0 0 0;
  color: #475569;
  font-size: 14px;
  line-height: 1.45;
}
.candidate-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
}
.candidate-card {
  border-radius: 22px;
  padding: 16px;
  background: linear-gradient(145deg, rgba(15,23,42,0.82) 0%, rgba(15,23,42,0.64) 100%);
  border: 1px solid rgba(255,255,255,0.12);
  box-shadow:
    inset 0 1px 0 rgba(255,255,255,0.10),
    0 4px 8px rgba(15,23,42,0.10),
    0 16px 32px rgba(15,23,42,0.14);
  transition: transform 0.18s ease, box-shadow 0.18s ease;
}
.candidate-card:hover {
  transform: translateY(-2px) scale(1.01);
  box-shadow:
    inset 0 1px 0 rgba(255,255,255,0.14),
    0 8px 18px rgba(15,23,42,0.14),
    0 24px 48px rgba(15,23,42,0.18);
}
.candidate-card.selected {
  border-color: rgba(59,130,246,0.55);
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.08), 0 0 0 1px rgba(59,130,246,0.28);
}
.candidate-card-head {
  display: flex;
  gap: 12px;
  align-items: flex-start;
  justify-content: space-between;
}
.candidate-swatch {
  width: 56px;
  height: 56px;
  flex: 0 0 56px;
  border-radius: 16px;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.08), 0 10px 18px rgba(15,23,42,0.18);
}
.candidate-copy {
  min-width: 0;
}
.candidate-kicker {
  margin: 0 0 4px 0;
  color: rgba(191,219,254,0.86);
  font-size: 11px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}
.candidate-name {
  margin: 0;
  color: #f8fafc;
  font-size: 17px;
  line-height: 1.25;
  letter-spacing: -0.02em;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.candidate-meta {
  margin: 8px 0 0 0;
  color: rgba(226,232,240,0.82);
  font-size: 13px;
  line-height: 1.45;
}
.candidate-score-copy {
  margin: 8px 0 0 0;
  color: rgba(226,232,240,0.9);
  font-size: 13px;
  line-height: 1.45;
}
.candidate-chip-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 12px;
}
.candidate-chip {
  display: inline-flex;
  align-items: center;
  padding: 4px 10px;
  border-radius: 999px;
  background: rgba(255,255,255,0.08);
  border: 1px solid rgba(255,255,255,0.08);
  color: #e2e8f0;
  font-size: 12px;
}
.candidate-empty {
  color: #475569;
  font-size: 14px;
  line-height: 1.5;
}
@media (max-width: 1100px) {
  .sticky-player-dock {
    grid-template-columns: 1fr;
    gap: 10px;
    padding: 12px !important;
  }
  .candidate-grid {
    grid-template-columns: 1fr;
  }
  .sticky-player-title {
    font-size: 18px;
  }
  .sticky-player-sub {
    white-space: normal;
  }
}
.studio-grid { gap: 16px; align-items: flex-start; }
.studio-card {
  background: linear-gradient(145deg, rgba(255,255,255,0.24) 0%, rgba(255,255,255,0.14) 100%);
  border: 1px solid rgba(255, 255, 255, 0.62);
  border-radius: 26px;
  padding: 20px;
  box-shadow:
    inset 0 1px 0 rgba(255,255,255,0.64),
    0 4px 8px rgba(148,163,184,0.06),
    0 20px 40px rgba(148, 163, 184, 0.11);
  backdrop-filter: blur(28px) saturate(148%);
  -webkit-backdrop-filter: blur(28px) saturate(148%);
  transition: box-shadow 0.2s ease, transform 0.2s ease;
}
.studio-card:hover {
  transform: translateY(-1px);
  box-shadow:
    inset 0 1px 0 rgba(255,255,255,0.72),
    0 8px 20px rgba(148,163,184,0.10),
    0 28px 52px rgba(148,163,184,0.14);
}
.library-shell {
  background: linear-gradient(180deg, rgba(255,255,255,0.22) 0%, rgba(255,255,255,0.14) 100%);
  border: 1px solid rgba(255,255,255,0.60);
  border-radius: 24px;
  padding: 18px;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.64), 0 20px 42px rgba(148,163,184,0.12);
  backdrop-filter: blur(26px) saturate(145%);
  -webkit-backdrop-filter: blur(26px) saturate(145%);
}
.library-heading { margin: 0; font-size: 34px; color: #0f172a; letter-spacing: -0.03em; }
.library-sub { margin: 6px 0 0 0; color: #475569; font-size: 15px; line-height: 1.5; }
.detail-title { margin: 0 0 6px 0; font-size: 26px; color: #111827; letter-spacing: -0.03em; }
.detail-meta { margin: 0 0 12px 0; color: #64748b; font-size: 14px; }
.quality-hero {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-bottom: 12px;
}
.quality-summary {
  margin: 0;
  color: #334155;
  font-size: 14px;
  line-height: 1.5;
}
.quality-badge {
  display: inline-flex;
  align-items: baseline;
  gap: 6px;
  width: fit-content;
  padding: 6px 12px;
  border-radius: 999px;
  font-size: 13px;
  font-weight: 700;
  letter-spacing: -0.01em;
  border: 1px solid transparent;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.18);
}
.quality-badge small {
  font-size: 11px;
  font-weight: 700;
  opacity: 0.8;
}
.quality-badge--excellent {
  background: rgba(16,185,129,0.18);
  color: #065f46;
  border-color: rgba(16,185,129,0.28);
}
.quality-badge--strong {
  background: rgba(59,130,246,0.16);
  color: #1d4ed8;
  border-color: rgba(59,130,246,0.24);
}
.quality-badge--solid {
  background: rgba(245,158,11,0.16);
  color: #92400e;
  border-color: rgba(245,158,11,0.24);
}
.quality-badge--review {
  background: rgba(239,68,68,0.14);
  color: #b91c1c;
  border-color: rgba(239,68,68,0.22);
}
.chip-row { display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0 14px 0; }
.chip {
  background: rgba(255,255,255,0.42);
  color: #0f172a;
  border: 1px solid rgba(255,255,255,0.58);
  border-radius: 999px;
  padding: 4px 10px;
  font-size: 13px;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.68);
}
.lyrics-box {
  background: linear-gradient(180deg, rgba(255,255,255,0.34) 0%, rgba(255,255,255,0.24) 100%);
  border-radius: 18px;
  padding: 12px;
  max-height: 240px;
  overflow: auto;
  white-space: pre-wrap;
  color: #111827;
  font-size: 14px;
  line-height: 1.55;
  border: 1px solid rgba(255,255,255,0.56);
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.70);
}
.cover-art {
  width: 100%;
  aspect-ratio: 1 / 1;
  border-radius: 18px;
  margin-bottom: 14px;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.15), 0 16px 28px rgba(17,24,39,0.14);
}
.mini-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  grid-auto-rows: auto;
  gap: 10px;
}
.mini-grid > .mini-card:first-child {
  grid-column: span 2;
  grid-row: span 2;
}
.mini-grid > .mini-card:first-child .mini-cover {
  aspect-ratio: 16 / 9;
  border-radius: 14px;
}
.mini-grid > .mini-card:nth-child(2) {
  grid-column: span 2;
}
@media (max-width: 900px) {
  .mini-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
  .mini-grid > .mini-card:first-child {
    grid-column: span 2;
    grid-row: span 1;
  }
  .mini-grid > .mini-card:first-child .mini-cover {
    aspect-ratio: 1 / 1;
  }
  .mini-grid > .mini-card:nth-child(2) {
    grid-column: span 2;
  }
}
.mini-card {
  position: relative;
  background: linear-gradient(145deg, rgba(255,255,255,0.26) 0%, rgba(255,255,255,0.14) 100%);
  border: 1px solid rgba(255,255,255,0.60);
  border-radius: 20px;
  padding: 12px;
  box-shadow:
    inset 0 1px 0 rgba(255,255,255,0.62),
    0 4px 6px rgba(148,163,184,0.06),
    0 12px 24px rgba(148,163,184,0.08);
  backdrop-filter: blur(28px) saturate(150%);
  -webkit-backdrop-filter: blur(28px) saturate(150%);
  transition: transform 0.18s ease, box-shadow 0.18s ease;
}
.mini-card:hover {
  transform: translateY(-2px) scale(1.01);
  box-shadow:
    inset 0 1px 0 rgba(255,255,255,0.70),
    0 8px 16px rgba(148,163,184,0.10),
    0 20px 40px rgba(148,163,184,0.14);
}
.mini-card .quality-badge {
  margin-bottom: 8px;
}
.mini-card .pick-pill-row {
  margin: 0 0 8px 0;
}
.mini-card .pick-pill--best {
  color: #1d4ed8;
  background: rgba(59,130,246,0.14);
  border-color: rgba(59,130,246,0.18);
}
.mini-card .pick-pill--favorite {
  color: #be185d;
  background: rgba(244,114,182,0.14);
  border-color: rgba(244,114,182,0.2);
}
.mini-card .pick-pill--archived {
  color: #334155;
  background: rgba(148,163,184,0.18);
  border-color: rgba(148,163,184,0.26);
}
.pinned-shell {
  margin-bottom: 14px;
}
.pinned-title {
  margin: 0 0 10px 0;
  color: #0f172a;
  font-size: 20px;
  letter-spacing: -0.02em;
}
.pinned-empty {
  margin: 0;
  color: #64748b;
  font-size: 14px;
  line-height: 1.5;
}
.pinned-card {
  box-shadow: 0 16px 28px rgba(148,163,184,0.12);
}
.mini-cover {
  width: 100%;
  aspect-ratio: 1 / 1;
  border-radius: 12px;
  margin-bottom: 8px;
}
.mini-title {
  font-size: 14px;
  font-weight: 700;
  color: #111827;
  margin: 0 0 4px 0;
}
.mini-meta {
  font-size: 13px;
  color: #6b7280;
  margin: 0;
}
.history-shell {
  margin-bottom: 14px;
}
.history-summary {
  margin: 0 0 12px 0;
  color: #475569;
  font-size: 13px;
  line-height: 1.45;
}
.history-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
}
.history-grid > .history-card:first-child {
  grid-column: span 2;
}
@media (max-width: 1100px) {
  .history-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
  .history-grid > .history-card:first-child {
    grid-column: span 2;
  }
}
@media (max-width: 720px) {
  .history-grid {
    grid-template-columns: 1fr;
  }
  .history-grid > .history-card:first-child {
    grid-column: span 1;
  }
}
.history-card {
  background: linear-gradient(145deg, rgba(255,255,255,0.22) 0%, rgba(255,255,255,0.14) 100%);
  border: 1px solid rgba(255,255,255,0.58);
  border-radius: 20px;
  padding: 14px;
  box-shadow:
    inset 0 1px 0 rgba(255,255,255,0.64),
    0 4px 8px rgba(148,163,184,0.06),
    0 12px 28px rgba(148,163,184,0.09);
  backdrop-filter: blur(26px) saturate(148%);
  -webkit-backdrop-filter: blur(26px) saturate(148%);
  transition: transform 0.18s ease, box-shadow 0.18s ease;
}
.history-card:hover {
  transform: translateY(-2px);
  box-shadow:
    inset 0 1px 0 rgba(255,255,255,0.70),
    0 8px 20px rgba(148,163,184,0.12),
    0 20px 44px rgba(148,163,184,0.14);
}
.history-head {
  display: flex;
  justify-content: space-between;
  gap: 10px;
  align-items: center;
  margin-bottom: 8px;
}
.history-time,
.history-meta,
.history-note,
.history-score {
  margin: 0;
  font-size: 13px;
  line-height: 1.45;
}
.history-time,
.history-meta {
  color: #64748b;
}
.history-title {
  margin: 0 0 6px 0;
  color: #0f172a;
  font-size: 16px;
  letter-spacing: -0.02em;
}
.history-score {
  color: #334155;
  font-weight: 700;
}
.history-note {
  color: #475569;
}
.render-status {
  display: inline-flex;
  align-items: center;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  border: 1px solid transparent;
}
.render-status--queued {
  color: #92400e;
  background: rgba(245,158,11,0.16);
  border-color: rgba(245,158,11,0.24);
}
.render-status--running {
  color: #1d4ed8;
  background: rgba(59,130,246,0.16);
  border-color: rgba(59,130,246,0.24);
}
.render-status--done {
  color: #065f46;
  background: rgba(16,185,129,0.16);
  border-color: rgba(16,185,129,0.24);
}
.render-status--failed {
  color: #b91c1c;
  background: rgba(239,68,68,0.14);
  border-color: rgba(239,68,68,0.22);
}

/* ═══════════════════════════════════
   BENTO GRID SYSTEM
   ═══════════════════════════════════ */
.bento-grid {
  display: grid;
  grid-template-columns: repeat(12, minmax(0, 1fr));
  grid-auto-rows: minmax(80px, auto);
  gap: 12px;
}
.bento-col-2  { grid-column: span 2; }
.bento-col-3  { grid-column: span 3; }
.bento-col-4  { grid-column: span 4; }
.bento-col-6  { grid-column: span 6; }
.bento-col-8  { grid-column: span 8; }
.bento-col-12 { grid-column: span 12; }
.bento-row-2  { grid-row: span 2; }
.bento-row-3  { grid-row: span 3; }

.bento-cell {
  position: relative;
  overflow: hidden;
  background: linear-gradient(145deg, rgba(255,255,255,0.24) 0%, rgba(255,255,255,0.14) 100%);
  border: 1px solid rgba(255,255,255,0.62);
  border-radius: 24px;
  padding: 18px;
  box-shadow:
    inset 0 1px 0 rgba(255,255,255,0.64),
    0 4px 8px rgba(148,163,184,0.06),
    0 16px 36px rgba(148,163,184,0.10);
  backdrop-filter: blur(28px) saturate(150%);
  -webkit-backdrop-filter: blur(28px) saturate(150%);
  transition: transform 0.2s cubic-bezier(.22,.68,0,1.2), box-shadow 0.2s ease;
}
.bento-cell:hover {
  transform: translateY(-3px) scale(1.008);
  box-shadow:
    inset 0 1px 0 rgba(255,255,255,0.72),
    0 8px 20px rgba(148,163,184,0.12),
    0 24px 52px rgba(148,163,184,0.16);
}
.bento-cell--dark {
  background: linear-gradient(145deg, rgba(15,23,42,0.84) 0%, rgba(15,23,42,0.68) 100%);
  border-color: rgba(255,255,255,0.10);
  box-shadow:
    inset 0 1px 0 rgba(255,255,255,0.08),
    0 4px 10px rgba(15,23,42,0.10),
    0 16px 36px rgba(15,23,42,0.14);
}
.bento-cell--accent {
  background: linear-gradient(135deg, rgba(15,23,42,0.92) 0%, rgba(37,99,235,0.82) 55%, rgba(56,189,248,0.72) 100%);
  border-color: rgba(255,255,255,0.18);
  color: #f8fafc;
}
.bento-stat {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 12px;
  border-radius: 999px;
  background: rgba(255,255,255,0.18);
  border: 1px solid rgba(255,255,255,0.32);
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.28);
  font-size: 13px;
  font-weight: 700;
  color: #0f172a;
}
.bento-divider {
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.48), transparent);
  margin: 12px 0;
  border: 0;
}
.bento-label {
  display: inline-block;
  font-size: 11px;
  font-weight: 800;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: #64748b;
  margin-bottom: 6px;
}
.bento-heading {
  margin: 0 0 4px 0;
  font-size: 22px;
  letter-spacing: -0.03em;
  color: #0f172a;
  line-height: 1.2;
}
.bento-heading--large { font-size: 32px; letter-spacing: -0.04em; }
.bento-heading--dark  { color: #f8fafc; }
.bento-copy {
  margin: 0;
  font-size: 14px;
  line-height: 1.55;
  color: #475569;
}
.bento-copy--dark { color: rgba(226,232,240,0.84); }
.bento-cell::before {
  content: "";
  position: absolute;
  inset: auto;
  width: 160px;
  height: 160px;
  border-radius: 999px;
  pointer-events: none;
  opacity: 0;
  transition: opacity 0.3s ease;
}
.bento-cell:hover::before { opacity: 1; }
.bento-cell--glow-indigo::before {
  top: -40px; right: -40px;
  background: radial-gradient(circle, rgba(99,102,241,0.28) 0%, transparent 70%);
}
.bento-cell--glow-teal::before {
  bottom: -40px; left: -40px;
  background: radial-gradient(circle, rgba(45,212,191,0.28) 0%, transparent 70%);
}
.bento-cell--glow-sky::before {
  top: -30px; right: -30px;
  background: radial-gradient(circle, rgba(56,189,248,0.24) 0%, transparent 70%);
}
.bento-cell:focus-within {
  outline: 2px solid rgba(37,99,235,0.4);
  outline-offset: 2px;
}
.pinned-card {
  box-shadow:
    inset 0 1px 0 rgba(255,255,255,0.68),
    0 4px 10px rgba(148,163,184,0.08),
    0 16px 32px rgba(148,163,184,0.12);
}
.library-shell {
  background: linear-gradient(145deg, rgba(255,255,255,0.24) 0%, rgba(255,255,255,0.14) 100%);
  border: 1px solid rgba(255,255,255,0.62);
  border-radius: 26px;
  padding: 20px;
  box-shadow:
    inset 0 1px 0 rgba(255,255,255,0.66),
    0 4px 8px rgba(148,163,184,0.06),
    0 20px 42px rgba(148,163,184,0.12);
  backdrop-filter: blur(28px) saturate(150%);
  -webkit-backdrop-filter: blur(28px) saturate(150%);
}
"""

SECTION_HEADER_RE = re.compile(r"^\s*\[[^\]]+\]\s*$")
WORD_RE = re.compile(r"\b[\w']+\b")
SECTION_LINE_LIMITS = {
    "intro": 2,
    "verse": 4,
    "prechorus": 3,
    "chorus": 4,
    "bridge": 3,
    "outro": 2,
}
LIBRARY_FILTER_OPTIONS = [
    "All Tracks",
    "Favorites",
    "Training Candidates",
    "Token Ready",
    "Best Picks",
    "Top Rated",
    "Has Outro",
    "Multi-Take Sets",
    "Archived",
]
WORKSPACE_VIEWS = {
    "create": {
        "label": "Create",
        "eyebrow": "Workspace",
        "title": "Create Music",
        "copy": "Compose, guide, and render songs without the library and training tools crowding the same page.",
    },
    "library": {
        "label": "Library",
        "eyebrow": "Workspace",
        "title": "Track Library",
        "copy": "Browse saved renders, review best takes, and manage favorites, archives, and downloads in one focused view.",
    },
    "training": {
        "label": "Training Lab",
        "eyebrow": "Workspace",
        "title": "Training Lab",
        "copy": "Harvest strong app outputs, prepare experiments, and monitor smoke or clean-cycle training without stretching the create view.",
    },
    "transcribe": {
        "label": "Transcribe",
        "eyebrow": "Workspace",
        "title": "Lyric Transcription",
        "copy": "Run HeartTranscriptor on uploads or local files in a separate workspace built for lyrics review.",
    },
    "settings": {
        "label": "Settings",
        "eyebrow": "Workspace",
        "title": "Studio Settings",
        "copy": "Keep model paths, checkpoints, and runtime defaults in one saved workspace instead of mixing them into writing and review.",
    },
}
GENERATION_PRESETS = {
    "Fast Render": {
        "max_audio_length_s": 60,
        "codec_num_steps": 4,
        "codec_guidance_scale": 1.1,
        "mac_safe": True,
        "keep_models_loaded": True,
        "note": "Fastest local draft preset for checking prompt direction and song shape.",
    },
    "Balanced 2 Min": {
        "max_audio_length_s": 120,
        "codec_num_steps": 6,
        "codec_guidance_scale": 1.25,
        "mac_safe": True,
        "keep_models_loaded": True,
        "note": "Best default balance for local 2-minute songs.",
    },
    "Best Quality": {
        "max_audio_length_s": 120,
        "codec_num_steps": 10,
        "codec_guidance_scale": 1.4,
        "mac_safe": True,
        "keep_models_loaded": True,
        "note": "Highest local decode quality. Slowest preset, best for final exports.",
    },
}
GENERATION_PRESET_ALIASES = {
    "Fast Draft": "Fast Render",
    "Clean 2 Min": "Best Quality",
}
TAG_STYLE_PRESETS = {
    "Starter Pop": "piano pop, uplifting, bright female vocal, warm acoustic drums, melodic chorus, clean radio mix",
    "Alt Rock": "alt rock, gritty male vocal, crunchy electric guitar, live drums, driving chorus, wide stereo mix",
    "Arena Rock": "arena rock, powerful vocal, big electric guitars, punchy drums, anthem chorus, polished stadium mix",
    "Hard Rock": "hard rock, raspy vocal, distorted guitar riffs, heavy live drums, aggressive groove, thick wall-of-sound mix",
    "Thrash Metal": "thrash metal, aggressive shouted vocal, fast palm-muted guitars, double-kick drums, relentless riffing, raw heavy mix",
    "Metalcore": "metalcore, screamed verses and sung chorus, downtuned guitars, breakdown drums, huge hook, modern heavy mix",
    "Acoustic Ballad": "acoustic ballad, intimate vocal, fingerpicked guitar, soft percussion, emotional chorus, warm close mix",
}
LYRIC_STYLE_PRESETS = {
    "Rock Anthem": "[Intro]\n\n[Verse]\nStreetlights wake up and the avenue glows\nBoots hit the ground and the whole block knows\nWe came to turn the silence into sparks tonight\n\n[Prechorus]\nEvery closed door starts shaking loose\nWhen the crowd and the amps cut through\n\n[Chorus]\nWe were made for the fire\nMade for the noise\nHands in the air with a thousand voices\nNo backing down when the night gets wild\nWe turn the dark electric for a little while\n\n[Bridge]\nTurn it up\nLet it ride\n\n[Outro]\nWe are still here",
    "Alt Rock Story": "[Intro]\n\n[Verse]\nRain on the windshield and a note on the dash\nHalf of your promises still burn like ash\nI keep driving through the streets we knew\n\n[Prechorus]\nLate night radio hums the same old tune\nSome things fade but they don't leave you\n\n[Chorus]\nI run through the static\nRun through the smoke\nHolding on to words we never spoke\nIf I break I break into the light\nSomewhere in the noise I find my life\n\n[Bridge]\nMaybe every ending opens up a road\nMaybe every silence lifts another load\n\n[Outro]\nThrough the static\nI come alive",
    "Thrash Metal": "[Intro]\n\n[Verse]\nSteel-tooth morning chewing through the haze\nSirens in my bloodstream, furnace in my veins\nEvery smiling tyrant wants another name to burn\nI sharpen all my anger till the tables turn\n\n[Prechorus]\nNo surrender to the fear they sell\nRaise the riff and ring the alarm bell\n\n[Chorus]\nBreak the chain\nTorch the throne\nRun the system down to the bone\nFaster, louder, all-out attack\nNo mercy and no turning back\n\n[Bridge]\nCut the wire\nSplit the sky\nLet the old machine die\n\n[Outro]\nAshes rise",
    "Metalcore Breakdown": "[Intro]\n\n[Verse]\nHeavy lungs in the cold air\nOld wounds under every prayer\nI was raised in the wreckage\nNow I bring the pressure there\n\n[Prechorus]\nEvery setback carved a deeper line\nStill I get up every time\n\n[Chorus]\nI won't bend\nI won't break\nPull me under I still wake\nFrom the dark into the flood of light\nI bring the weight and survive the night\n\n[Breakdown]\nPush back\nStand fast\nNo fear when the shadows crash\nBuilt to outlast\n\n[Outro]\nStill alive",
    "Soft Rock Ballad": "[Intro]\n\n[Verse]\nYour jacket on the back of my old chair\nSunday light and quiet moving through the air\nYou laugh and all the noise in me gets low\nLike every restless road suddenly feels like home\n\n[Prechorus]\nTime moves slow when your hand finds mine\nEven the silence starts to shine\n\n[Chorus]\nStay with me through the fading blue\nAll I ever wanted leans toward you\nEasy hearts can still run deep\nYou are the promise that I want to keep\n\n[Bridge]\nIf the night gets long\nI will still be here\n\n[Outro]\nStay with me",
}
LYRIC_STYLE_LONGFORM_PRESETS = {
    "Rock Anthem": "[Intro]\n\n[Verse]\nStreetlights wake up and the avenue glows\nBoots hit the ground and the block knows\nWe came to turn the silence into sparks tonight\nEvery tired shadow starts to burn in neon light\n\n[Prechorus]\nEvery closed door starts shaking loose\nWhen the crowd and the amps cut through\nWe hear the whole city breathing in time\nWaiting for the chorus line\n\n[Chorus]\nWe were made for the fire\nMade for the noise\nHands in the air with a thousand voices\nNo backing down when the night gets wild\nWe turn the dark electric for a little while\n\n[Verse 2]\nHeadlights roll like thunder down the boulevard\nBroken plans and old scars never hit this hard\nWe turn the pressure into something we can raise\nBuilt a little hope inside the feedback haze\n\n[Prechorus]\nEvery lost kid starts finding truth\nWhen the drums and the wires break through\nNow the whole room is ready to rise\nWith the voltage in our eyes\n\n[Chorus]\nWe were made for the fire\nMade for the noise\nHands in the air with a thousand voices\nNo backing down when the night gets wild\nWe turn the dark electric for a little while\n\n[Bridge]\nTurn it up\nLet it ride\nHold the line\nThrough the light\n\n[Outro]\nWe are still here\nStill burning bright",
    "Alt Rock Story": "[Intro]\n\n[Verse]\nRain on the windshield and a note on the dash\nHalf your promises still burn like ash\nI keep driving through the streets we knew\nCounting signals that point to you\n\n[Prechorus]\nLate night radio hums the same old tune\nSome things fade but they don't leave you\nEven the skyline keeps your shape\nIn every fire escape\n\n[Chorus]\nI run through the static\nRun through the smoke\nHolding on to words we never spoke\nIf I break I break into the light\nSomewhere in the noise I find my life\n\n[Verse 2]\nCoffee in a paper cup goes cold in my hand\nEvery empty station feels like borrowed land\nI hear your name inside the overhead train\nLike a ghost in the wires calling me again\n\n[Prechorus]\nNo clean ending ever comes too soon\nSome truths stay and rewrite you\nI let the rearview blur behind\nAnd keep the road in line\n\n[Chorus]\nI run through the static\nRun through the smoke\nHolding on to words we never spoke\nIf I break I break into the light\nSomewhere in the noise I find my life\n\n[Bridge]\nMaybe every ending opens up a road\nMaybe every silence lifts another load\nI keep the engine warm tonight\nAnd drive straight through the white\n\n[Outro]\nThrough the static\nI come alive",
    "Thrash Metal": "[Intro]\n\n[Verse]\nSteel-tooth morning chewing through the haze\nSirens in my bloodstream, furnace in my veins\nEvery smiling tyrant wants another name to burn\nI sharpen all my anger till the tables turn\n\n[Prechorus]\nNo surrender to the fear they sell\nRaise the riff and ring the alarm bell\nSteel on steel and eye to eye\nWe are the answer when the wires fry\n\n[Chorus]\nBreak the chain\nTorch the throne\nRun the system down to the bone\nFaster, louder, all-out attack\nNo mercy and no turning back\n\n[Verse 2]\nBlack smoke banners hanging over every gate\nCold hands counting bodies while they manufacture fate\nI feed the engine broken gears and razor light\nKick the rotten scaffolds down and drag them into night\n\n[Prechorus]\nNo retreat when the signal swells\nHit the floor like artillery shells\nRiot hearts and a sharpened cry\nWe leave the old machine to die\n\n[Chorus]\nBreak the chain\nTorch the throne\nRun the system down to the bone\nFaster, louder, all-out attack\nNo mercy and no turning back\n\n[Bridge]\nCut the wire\nSplit the sky\nBring the pressure\nMultiply\n\n[Outro]\nAshes rise\nWe survive",
    "Metalcore Breakdown": "[Intro]\n\n[Verse]\nHeavy lungs in the cold air\nOld wounds under every prayer\nI was raised in the wreckage\nNow I bring the pressure there\nEvery scar learned my name\nEvery setback fed the flame\nI turned weight to a weapon\nI won't go down the same\n\n[Prechorus]\nEvery setback carved a deeper line\nStill I get up every time\nWhen the ceiling starts to cave\nI become the shape I save\n\n[Chorus]\nI won't bend\nI won't break\nPull me under I still wake\nFrom the dark into the flood of light\nI bring the weight and survive the night\n\n[Verse 2]\nBroken sleep and crowded rooms\nSpent years learning how to move\nThrough the panic and the pressure\nWith nothing left to prove\nNow the pulse is in my chest\nLike a war drum through the stress\nEvery fear that tried to hold me\nOnly taught me to resist\n\n[Prechorus]\nI can feel the ground align\nPain became a sharper spine\nWhen the shadows close in fast\nI become the thing that lasts\n\n[Chorus]\nI won't bend\nI won't break\nPull me under I still wake\nFrom the dark into the flood of light\nI bring the weight and survive the night\n\n[Breakdown]\nPush back\nStand fast\nNo fear when shadows crash\nBuilt to outlast\nCome back\n\n[Outro]\nStill alive\nStill intact",
    "Soft Rock Ballad": "[Intro]\n\n[Verse]\nYour jacket on the back of my old chair\nSunday light and quiet moving through the air\nYou laugh and all the noise in me gets low\nLike every restless road suddenly feels like home\n\n[Prechorus]\nTime moves slow when your hand finds mine\nEven the silence starts to shine\nThe room gets warm without a word\nLike a song we've always heard\n\n[Chorus]\nStay with me through the fading blue\nAll I ever wanted leans toward you\nEasy hearts can still run deep\nYou are the promise that I want to keep\n\n[Verse 2]\nMorning coffee and the window half awake\nLittle plans we trace across the day we make\nEvery small thing lands exactly where it should\nYou turn the ordinary into something good\n\n[Prechorus]\nWhen the long week wears me thin\nYou let the quiet pull me in\nEven the dark can feel sincere\nWhen I know you're here\n\n[Chorus]\nStay with me through the fading blue\nAll I ever wanted leans toward you\nEasy hearts can still run deep\nYou are the promise that I want to keep\n\n[Bridge]\nIf the night gets long\nI will still be here\nIf the world runs hard\nWe can hold it near\n\n[Outro]\nStay with me\nStay right here",
}
GENRE_COMBO_KITS = {
    "Starter Pop Kit": {
        "generation_preset": "Fast Render",
        "lyric_style_preset": "Rock Anthem",
        "tag_style_preset": "Starter Pop",
        "note": "Quick pop starter with a clean chorus-first shape.",
    },
    "Alt Rock Kit": {
        "generation_preset": "Fast Render",
        "lyric_style_preset": "Alt Rock Story",
        "tag_style_preset": "Alt Rock",
        "note": "Moody alt-rock structure with a tighter story arc.",
    },
    "Arena Rock Kit": {
        "generation_preset": "Fast Render",
        "lyric_style_preset": "Rock Anthem",
        "tag_style_preset": "Arena Rock",
        "note": "Big-hook rock starter aimed at wide choruses and polished guitars.",
    },
    "Hard Rock Kit": {
        "generation_preset": "Fast Render",
        "lyric_style_preset": "Rock Anthem",
        "tag_style_preset": "Hard Rock",
        "note": "Heavier rock starter with more aggressive guitar tone and groove.",
    },
    "Thrash Metal Kit": {
        "generation_preset": "Fast Render",
        "lyric_style_preset": "Thrash Metal",
        "tag_style_preset": "Thrash Metal",
        "note": "Fast, hostile starter for riff-driven thrash ideas.",
    },
    "Metalcore Kit": {
        "generation_preset": "Fast Render",
        "lyric_style_preset": "Metalcore Breakdown",
        "tag_style_preset": "Metalcore",
        "note": "Breakdown-ready starter with a singable chorus frame.",
    },
    "Soft Rock Kit": {
        "generation_preset": "Fast Render",
        "lyric_style_preset": "Soft Rock Ballad",
        "tag_style_preset": "Acoustic Ballad",
        "note": "Softer emotional starter with a warm ballad mix target.",
    },
}


def lyric_text_for_style(style_name: str, max_audio_length_s: int) -> str:
    resolved_name = str(style_name or "").strip()
    if max_audio_length_s >= 90:
        return LYRIC_STYLE_LONGFORM_PRESETS.get(
            resolved_name,
            LYRIC_STYLE_PRESETS.get(resolved_name, LYRIC_STYLE_PRESETS["Rock Anthem"]),
        )
    return LYRIC_STYLE_PRESETS.get(resolved_name, LYRIC_STYLE_PRESETS["Rock Anthem"])


def fitted_lyric_text_for_style(
    style_name: str,
    max_audio_length_s: int,
    tags: str = "",
    preset_name: str = "Balanced 2 Min",
) -> str:
    lyrics = lyric_text_for_style(style_name, max_audio_length_s)
    _, recommended_max = recommended_word_range(max_audio_length_s)
    if count_words(lyrics) > recommended_max:
        lyrics, _, _, _ = optimize_lyrics_for_length(lyrics, max_audio_length_s, tags, preset_name)
    return lyrics


def resolve_generation_preset(preset_name: str | None) -> tuple[str, dict]:
    requested = str(preset_name or "").strip()
    canonical = GENERATION_PRESET_ALIASES.get(requested, requested)
    if canonical in GENERATION_PRESETS:
        return canonical, GENERATION_PRESETS[canonical]
    return "Balanced 2 Min", GENERATION_PRESETS["Balanced 2 Min"]


def apply_tag_style_preset(
    style_name: str,
    lyrics: str,
    max_audio_length_s: int,
    preset_name: str,
):
    resolved_name = str(style_name or "").strip()
    tags = TAG_STYLE_PRESETS.get(resolved_name, TAG_STYLE_PRESETS["Starter Pop"])
    prompt_guidance = build_prompt_guidance(lyrics, tags, max_audio_length_s, preset_name)
    status = f"Tag pack applied: {resolved_name or 'Starter Pop'}."
    return tags, prompt_guidance, status


def apply_lyric_style_preset(
    style_name: str,
    tags: str,
    max_audio_length_s: int,
    preset_name: str,
):
    resolved_name = str(style_name or "").strip()
    lyrics = fitted_lyric_text_for_style(resolved_name, int(max_audio_length_s), tags, preset_name)
    guidance = build_lyrics_guidance(lyrics, max_audio_length_s)
    prompt_guidance = build_prompt_guidance(lyrics, tags, max_audio_length_s, preset_name)
    status = f"Lyric starter applied: {resolved_name or 'Rock Anthem'} for {int(max_audio_length_s)}s."
    return lyrics, guidance, prompt_guidance, status


def apply_genre_combo_kit(combo_name: str):
    resolved_name = str(combo_name or "").strip()
    combo = GENRE_COMBO_KITS.get(resolved_name, GENRE_COMBO_KITS["Starter Pop Kit"])
    generation_preset_name, generation_preset = resolve_generation_preset(combo["generation_preset"])
    lyric_style_name = str(combo["lyric_style_preset"])
    tag_style_name = str(combo["tag_style_preset"])
    max_audio_length_s = int(generation_preset["max_audio_length_s"])
    tags = TAG_STYLE_PRESETS.get(tag_style_name, TAG_STYLE_PRESETS["Starter Pop"])
    lyrics = fitted_lyric_text_for_style(lyric_style_name, max_audio_length_s, tags, generation_preset_name)
    guidance = build_lyrics_guidance(lyrics, max_audio_length_s)
    prompt_guidance = build_prompt_guidance(lyrics, tags, max_audio_length_s, generation_preset_name)
    status = f"Combo kit applied: {resolved_name or 'Starter Pop Kit'}. {combo['note']}"
    return (
        generation_preset_name,
        lyric_style_name,
        tag_style_name,
        lyrics,
        tags,
        max_audio_length_s,
        int(generation_preset["codec_num_steps"]),
        float(generation_preset["codec_guidance_scale"]),
        bool(generation_preset["mac_safe"]),
        bool(generation_preset["keep_models_loaded"]),
        guidance,
        prompt_guidance,
        status,
    )


def mps_available() -> bool:
    return getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()


def default_devices(mac_safe: bool) -> tuple[torch.device, torch.device]:
    if mac_safe:
        if not mps_available():
            raise gr.Error("Mac-safe mode requires MPS support on this machine.")
        return torch.device("mps"), torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.device("cuda")
    if mps_available():
        return torch.device("mps"), torch.device("cpu")
    return torch.device("cpu"), torch.device("cpu")


def default_dtypes(mula_device: torch.device, codec_device: torch.device):
    if mula_device.type == "cuda":
        mula_dtype = torch.bfloat16
    elif mula_device.type == "mps":
        mula_dtype = torch.float16
    else:
        mula_dtype = torch.float32

    codec_dtype = torch.float32
    print(f"⚡ Devices: HeartMuLa → {mula_device} ({mula_dtype}), HeartCodec → {codec_device} ({codec_dtype})")
    return mula_dtype, codec_dtype


def sanitize_filename(filename: str, default_name: str) -> str:
    clean = (filename or "").strip()
    if not clean:
        clean = default_name
    clean = Path(clean).name
    if not clean.endswith(".mp3"):
        clean = f"{clean}.mp3"
    return clean


def coerce_bool(value, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    if value is None:
        return default
    return bool(value)


def default_studio_preferences() -> dict:
    promoted = load_promoted_checkpoint_metadata()
    return {
        "model_path": str(DEFAULT_MODEL_PATH),
        "output_name": "marks-studio.mp3",
        "finetune_checkpoint": str(promoted.get("promoted_checkpoint") or "") if promoted else "",
        "codec_num_steps": 6,
        "codec_guidance_scale": 1.25,
        "mac_safe": True,
        "keep_models_loaded": True,
        "transcribe_model_path": str(DEFAULT_MODEL_PATH),
        "transcribe_mac_safe": True,
    }


def normalize_studio_preferences(payload: dict | None) -> dict:
    defaults = default_studio_preferences()
    prefs = defaults.copy()
    if isinstance(payload, dict):
        prefs.update(payload)
    prefs["model_path"] = str(prefs.get("model_path") or defaults["model_path"]).strip() or defaults["model_path"]
    prefs["output_name"] = sanitize_filename(str(prefs.get("output_name") or defaults["output_name"]), defaults["output_name"])
    prefs["finetune_checkpoint"] = str(prefs.get("finetune_checkpoint") or "").strip()
    prefs["codec_num_steps"] = max(4, min(10, int(float(prefs.get("codec_num_steps", defaults["codec_num_steps"])))))
    prefs["codec_guidance_scale"] = round(
        max(1.0, min(2.0, float(prefs.get("codec_guidance_scale", defaults["codec_guidance_scale"])))),
        2,
    )
    prefs["mac_safe"] = coerce_bool(prefs.get("mac_safe"), defaults["mac_safe"])
    prefs["keep_models_loaded"] = coerce_bool(prefs.get("keep_models_loaded"), defaults["keep_models_loaded"])
    prefs["transcribe_model_path"] = (
        str(prefs.get("transcribe_model_path") or defaults["transcribe_model_path"]).strip()
        or defaults["transcribe_model_path"]
    )
    prefs["transcribe_mac_safe"] = coerce_bool(
        prefs.get("transcribe_mac_safe"),
        defaults["transcribe_mac_safe"],
    )
    return prefs


def read_studio_preferences() -> tuple[dict, bool]:
    if STUDIO_PREFERENCES_PATH.exists():
        try:
            payload = json.loads(STUDIO_PREFERENCES_PATH.read_text(encoding="utf-8"))
            return normalize_studio_preferences(payload), True
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    return normalize_studio_preferences(None), False


def render_settings_overview() -> str:
    promoted = load_promoted_checkpoint_metadata()
    promoted_copy = "Current Best Clean is not promoted yet."
    if promoted:
        promoted_copy = html.escape(str(promoted.get("promoted_checkpoint") or ""))
    return (
        "<div class='history-shell'>"
        "<h3 class='pinned-title'>Studio Defaults</h3>"
        "<div class='history-grid'>"
        "<div class='history-card'>"
        "<span class='render-status'>Output Folder</span>"
        f"<p class='history-title'>{html.escape(str(DEFAULT_OUTPUT_DIR))}</p>"
        "<p class='history-meta'>New renders, metadata, and token sidecars save here.</p>"
        "</div>"
        "<div class='history-card'>"
        "<span class='render-status'>Best Clean Alias</span>"
        f"<p class='history-title'>{promoted_copy}</p>"
        "<p class='history-meta'>Use Saved Checkpoints to switch generation without typing long local paths.</p>"
        "</div>"
        "</div>"
        "</div>"
    )


def load_studio_preferences_for_ui():
    prefs, found = read_studio_preferences()
    status = "Loaded saved studio settings." if found else "Using default studio settings."
    return (
        prefs["model_path"],
        prefs["output_name"],
        prefs["finetune_checkpoint"],
        refresh_finetune_checkpoint_choices(prefs["finetune_checkpoint"]),
        prefs["codec_num_steps"],
        prefs["codec_guidance_scale"],
        prefs["mac_safe"],
        prefs["keep_models_loaded"],
        prefs["transcribe_model_path"],
        prefs["transcribe_mac_safe"],
        status,
        render_settings_overview(),
    )


def save_studio_preferences(
    model_path: str,
    output_name: str,
    finetune_checkpoint: str,
    codec_num_steps: int,
    codec_guidance_scale: float,
    mac_safe: bool,
    keep_models_loaded: bool,
    transcribe_model_path: str,
    transcribe_mac_safe: bool,
):
    prefs = normalize_studio_preferences(
        {
            "model_path": model_path,
            "output_name": output_name,
            "finetune_checkpoint": finetune_checkpoint,
            "codec_num_steps": codec_num_steps,
            "codec_guidance_scale": codec_guidance_scale,
            "mac_safe": mac_safe,
            "keep_models_loaded": keep_models_loaded,
            "transcribe_model_path": transcribe_model_path,
            "transcribe_mac_safe": transcribe_mac_safe,
        }
    )
    STUDIO_PREFERENCES_PATH.write_text(json.dumps(prefs, indent=2), encoding="utf-8")
    return (
        prefs["model_path"],
        prefs["output_name"],
        prefs["finetune_checkpoint"],
        refresh_finetune_checkpoint_choices(prefs["finetune_checkpoint"]),
        prefs["codec_num_steps"],
        prefs["codec_guidance_scale"],
        prefs["mac_safe"],
        prefs["keep_models_loaded"],
        prefs["transcribe_model_path"],
        prefs["transcribe_mac_safe"],
        f"Studio settings saved at {datetime.now().strftime('%H:%M:%S')}.",
        render_settings_overview(),
    )


def reset_studio_preferences_for_ui():
    if STUDIO_PREFERENCES_PATH.exists():
        STUDIO_PREFERENCES_PATH.unlink()
    prefs = normalize_studio_preferences(None)
    return (
        prefs["model_path"],
        prefs["output_name"],
        prefs["finetune_checkpoint"],
        refresh_finetune_checkpoint_choices(prefs["finetune_checkpoint"]),
        prefs["codec_num_steps"],
        prefs["codec_guidance_scale"],
        prefs["mac_safe"],
        prefs["keep_models_loaded"],
        prefs["transcribe_model_path"],
        prefs["transcribe_mac_safe"],
        "Studio settings reset to defaults.",
        render_settings_overview(),
    )


def load_editor_text(default_path: Path, draft_path: Path) -> str:
    if draft_path.exists():
        return draft_path.read_text(encoding="utf-8")
    return default_path.read_text(encoding="utf-8")


def save_generate_drafts(lyrics: str, tags: str):
    LYRICS_DRAFT_PATH.write_text(lyrics, encoding="utf-8")
    TAGS_DRAFT_PATH.write_text(tags, encoding="utf-8")
    return f"Draft saved at {datetime.now().strftime('%H:%M:%S')}"


def reload_generate_drafts():
    return (
        load_editor_text(DEFAULT_LYRICS_PATH, LYRICS_DRAFT_PATH),
        load_editor_text(DEFAULT_TAGS_PATH, TAGS_DRAFT_PATH),
        "Loaded current draft values.",
    )


def reset_generate_defaults():
    if LYRICS_DRAFT_PATH.exists():
        LYRICS_DRAFT_PATH.unlink()
    if TAGS_DRAFT_PATH.exists():
        TAGS_DRAFT_PATH.unlink()
    return (
        DEFAULT_LYRICS_PATH.read_text(encoding="utf-8"),
        DEFAULT_TAGS_PATH.read_text(encoding="utf-8"),
        "Draft cleared. Loaded repo defaults.",
    )


def reload_generate_drafts_with_guidance(max_audio_length_s: int, preset_name: str):
    lyrics, tags, status = reload_generate_drafts()
    return (
        lyrics,
        tags,
        status,
        build_lyrics_guidance(lyrics, max_audio_length_s),
        build_prompt_guidance(lyrics, tags, max_audio_length_s, preset_name),
    )


def reset_generate_defaults_with_guidance(max_audio_length_s: int, preset_name: str):
    lyrics, tags, status = reset_generate_defaults()
    return (
        lyrics,
        tags,
        status,
        build_lyrics_guidance(lyrics, max_audio_length_s),
        build_prompt_guidance(lyrics, tags, max_audio_length_s, preset_name),
    )


def count_words(text: str) -> int:
    return len(WORD_RE.findall(text or ""))


def recommended_word_range(max_audio_length_s: int) -> tuple[int, int]:
    target_minutes = max(max_audio_length_s, 1) / 60
    recommended_min = max(20, int(target_minutes * 70))
    recommended_max = max(recommended_min, int(target_minutes * 110))
    return recommended_min, recommended_max


def parse_lyrics_sections(lyrics: str):
    sections = []
    current_header = None
    current_lines = []

    for raw_line in (lyrics or "").splitlines():
        line = raw_line.rstrip()
        if SECTION_HEADER_RE.match(line.strip()):
            if current_header is not None or current_lines:
                sections.append((current_header, current_lines))
            current_header = line.strip()
            current_lines = []
            continue
        if line.strip():
            current_lines.append(line.strip())

    if current_header is not None or current_lines:
        sections.append((current_header, current_lines))

    return sections


def serialize_lyrics_sections(sections) -> str:
    blocks = []
    for header, lines in sections:
        block_lines = []
        if header:
            block_lines.append(header)
        block_lines.extend(lines)
        if block_lines:
            blocks.append("\n".join(block_lines))
    return "\n\n".join(blocks).strip()


def optimize_lyrics_for_length(lyrics: str, max_audio_length_s: int, tags: str, preset_name: str):
    original = (lyrics or "").strip()
    if not original:
        guidance = build_lyrics_guidance(original, max_audio_length_s)
        return original, "Lyrics are empty. Nothing to optimize.", guidance, build_prompt_guidance(original, tags, max_audio_length_s, preset_name)

    sections = parse_lyrics_sections(original)
    if not sections:
        guidance = build_lyrics_guidance(original, max_audio_length_s)
        return original, "Lyrics format was not recognized. No changes made.", guidance, build_prompt_guidance(original, tags, max_audio_length_s, preset_name)

    optimized_sections = []
    chorus_count = 0
    for header, lines in sections:
        section_name = (header or "").strip("[]").strip().lower()
        trimmed_lines = [line for line in lines if line]
        line_limit = SECTION_LINE_LIMITS.get(section_name)

        if section_name == "chorus":
            chorus_count += 1
            if chorus_count > 1:
                line_limit = 2

        if line_limit is not None:
            trimmed_lines = trimmed_lines[:line_limit]

        optimized_sections.append((header, trimmed_lines))

    optimized = serialize_lyrics_sections(optimized_sections)
    original_words = count_words(original)
    optimized_words = count_words(optimized)
    guidance = build_lyrics_guidance(optimized, max_audio_length_s)
    prompt_guidance = build_prompt_guidance(optimized, tags, max_audio_length_s, preset_name)

    if optimized == original:
        status = "Lyrics already look compact for the selected length. No trim applied."
    else:
        status = (
            f"Conservative trim applied for {max_audio_length_s}s: "
            f"{original_words} -> {optimized_words} words. Review the ending before generating."
        )

    if "[outro]" not in optimized.lower():
        status += " No [Outro] section found; adding a short outro manually can help the ending feel more complete."

    return optimized, status, guidance, prompt_guidance


def build_lyrics_guidance(lyrics: str, max_audio_length_s: int) -> str:
    words = count_words(lyrics)
    recommended_min, recommended_max = recommended_word_range(max_audio_length_s)

    if words > recommended_max:
        fit_note = "Lyrics look too long for the target length. The song may cut off before a clean ending."
    elif words < recommended_min:
        fit_note = "Lyrics look short for the target length. The model may stretch or repeat ideas."
    else:
        fit_note = "Lyrics length looks reasonable for the target length."

    outro_note = "For cleaner endings, end the lyrics with a short [Outro] section."
    return (
        f"Word count: {words}. Recommended for {max_audio_length_s}s: "
        f"{recommended_min}-{recommended_max} words. {fit_note} {outro_note}"
    )


def build_prompt_guidance(
    lyrics: str,
    tags: str,
    max_audio_length_s: int,
    preset_name: str = "Balanced 2 Min",
) -> str:
    preset_name, _ = resolve_generation_preset(preset_name)
    notes = []
    tag_parts = [piece.strip() for piece in (tags or "").split(",") if piece.strip()]
    lyrics_lower = (lyrics or "").lower()
    tags_lower = (tags or "").lower()
    recommended_min, recommended_max = recommended_word_range(max_audio_length_s)
    word_count = count_words(lyrics)
    section_headers = [header for header, _ in parse_lyrics_sections(lyrics)]

    if len(tag_parts) < 3:
        notes.append("Add 4-6 short tags: genre, mood, lead instrument, vocal style, groove, and mix.")
    elif len(tag_parts) > 6:
        notes.append("Trim tags to the strongest 3-6 phrases so the prompt stays focused.")
    else:
        notes.append("Tag density looks good. Keep tags short and specific across genre, mood, vocal, groove, and mix.")

    vocal_cues = ["vocal", "singer", "male vocal", "female vocal", "choir", "duet", "rap"]
    if not any(cue in tags_lower for cue in vocal_cues):
        notes.append("If vocals matter, add a vocal cue like 'female vocal', 'warm male vocal', or 'airy duet'.")

    if "[chorus]" not in lyrics_lower:
        notes.append("Add a short [Chorus] so the model has a repeat anchor.")
    if "[outro]" not in lyrics_lower:
        notes.append("Add a short [Outro] to help the ending resolve cleanly.")

    if not section_headers:
        notes.append("Use section headers like [Verse], [Chorus], and [Outro] for better structure control.")
    elif len(section_headers) < 3:
        notes.append("Structure is a bit thin. A verse, chorus, and outro usually land more cleanly than a single block.")

    if word_count > recommended_max:
        notes.append("Your lyrics are dense for this runtime. Shorten repeated chorus lines or trim verse lines.")
    elif word_count < recommended_min:
        notes.append("Your lyrics are short for this runtime. Add one more verse or a fuller chorus if you want less repetition.")

    if preset_name == "Fast Render":
        notes.append("Fast Render is for prompt iteration. Switch to Balanced 2 Min or Best Quality for the final pass.")
    elif preset_name == "Best Quality":
        notes.append("Best Quality spends more time in the codec decode stage. Use it when the arrangement is already locked.")

    return "Prompt Coach:\n- " + "\n- ".join(notes[:5])


def analyze_audio_output(audio_path: Path) -> dict:
    try:
        waveform, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
    except RuntimeError:
        return {}

    if waveform.size == 0:
        return {}

    mono = waveform.mean(axis=1)
    abs_mono = abs(mono)
    rms = float((mono**2).mean() ** 0.5)
    peak = float(abs_mono.max())
    tail_frames = max(1, min(len(mono), int(sample_rate * 4)))
    tail = mono[-tail_frames:]
    tail_rms = float((tail**2).mean() ** 0.5)
    tail_ratio = float(tail_rms / rms) if rms > 1e-6 else 0.0
    silence_ratio = float((abs_mono < 0.0025).mean())
    return {
        "audio_peak": round(peak, 4),
        "audio_rms": round(rms, 4),
        "tail_rms": round(tail_rms, 4),
        "tail_ratio": round(tail_ratio, 4),
        "silence_ratio": round(silence_ratio, 4),
    }


def compute_track_quality(track: dict) -> dict:
    lyrics = track.get("lyrics", "") or ""
    target_duration = track.get("duration_s")
    actual_duration = track.get("actual_duration_s")
    words = count_words(lyrics)
    recommended_min, recommended_max = recommended_word_range(int(target_duration or 120))
    positives = []
    cautions = []
    sections = parse_lyrics_sections(lyrics)
    section_headers = [header for header, _ in sections if header]
    structure_bonus = 0

    if recommended_min <= words <= recommended_max:
        lyric_score = 25
        positives.append("lyrics fit the runtime")
    elif words < recommended_min:
        gap = (recommended_min - words) / max(recommended_min, 1)
        lyric_score = max(0, round(25 - gap * 28))
        cautions.append("lyrics may be too short")
    else:
        gap = (words - recommended_max) / max(recommended_max, 1)
        lyric_score = max(0, round(25 - gap * 26))
        cautions.append("lyrics may be too dense")

    if isinstance(actual_duration, (float, int)) and target_duration:
        duration_gap = abs(float(actual_duration) - float(target_duration)) / max(float(target_duration), 1.0)
        duration_score = max(0, round(25 - duration_gap * 95))
        if duration_gap <= 0.08:
            positives.append("render length lands close to target")
        elif float(actual_duration) < float(target_duration) * 0.84:
            cautions.append("render may end early")
        else:
            cautions.append("render length drifts from target")
    else:
        duration_score = 15

    has_outro = "[outro]" in lyrics.lower()
    outro_score = 10 if has_outro else 2
    if has_outro:
        positives.append("lyrics include an outro cue")
    else:
        cautions.append("no explicit outro cue")

    if section_headers:
        structure_bonus += 3
        positives.append("section structure is clear")
    else:
        cautions.append("structure is loose")
    if "[chorus]" in lyrics.lower():
        structure_bonus += 2
    if len(section_headers) >= 3:
        structure_bonus += 1

    peak = track.get("audio_peak")
    rms = track.get("audio_rms")
    tail_ratio = track.get("tail_ratio")
    silence_ratio = track.get("silence_ratio")
    if isinstance(peak, (float, int)) and isinstance(rms, (float, int)) and isinstance(tail_ratio, (float, int)):
        if peak < 0.92:
            clip_score = 12
            positives.append("low clipping risk")
        elif peak < 0.97:
            clip_score = 9
        elif peak < 1.0:
            clip_score = 4
            cautions.append("possible clipping")
        else:
            clip_score = 0
            cautions.append("hard clipping detected")

        rms_score = max(0, min(12, round(12 - abs(float(rms) - 0.16) * 52)))
        if 0.07 <= float(rms) <= 0.24:
            positives.append("loudness looks healthy")
        elif float(rms) < 0.04:
            cautions.append("render may feel too quiet")
        elif float(rms) > 0.28:
            cautions.append("render may be over-compressed")

        if 0.14 <= float(tail_ratio) <= 0.95 and float(silence_ratio or 0.0) < 0.72:
            ending_score = 16
            positives.append("ending shape looks natural")
        elif 0.08 <= float(tail_ratio) <= 1.15:
            ending_score = 11
        else:
            ending_score = 5
            cautions.append("ending may feel abrupt")
    else:
        clip_score = 7
        rms_score = 7
        ending_score = 10

    total_score = int(
        max(
            0,
            min(
                100,
                lyric_score + duration_score + outro_score + clip_score + rms_score + ending_score + structure_bonus,
            ),
        )
    )
    if total_score >= 88:
        quality_label = "Excellent"
        quality_class = "excellent"
    elif total_score >= 76:
        quality_label = "Strong"
        quality_class = "strong"
    elif total_score >= 62:
        quality_label = "Solid"
        quality_class = "solid"
    else:
        quality_label = "Review"
        quality_class = "review"

    highlights = positives[:2]
    if cautions and total_score < 88:
        highlights.append(f"check: {cautions[0]}")
    elif len(positives) >= 3:
        highlights.append(positives[2])
    quality_summary = " • ".join(highlights) if highlights else "Heuristic score based on timing, lyrics, and audio shape."

    return {
        "quality_score": total_score,
        "quality_label": quality_label,
        "quality_class": quality_class,
        "quality_summary": quality_summary,
        "has_outro": has_outro,
        "word_count": words,
        "duration_score": duration_score,
        "ending_score": ending_score,
        "clip_score": clip_score,
        "rms_score": rms_score,
        "lyric_score": lyric_score,
        "structure_bonus": structure_bonus,
        "has_chorus": "[chorus]" in lyrics.lower(),
        "section_count": len(section_headers),
        "search_blob": " ".join(
            [
                str(track.get("title", "")),
                lyrics,
                str(track.get("tags", "")),
                str(track.get("preset_name", "")),
                str(track.get("created_at", "")),
            ]
        ).lower(),
    }


def enrich_track_quality(track: dict) -> dict:
    enriched = dict(track)
    enriched.update(compute_track_quality(enriched))
    return enriched


def build_rerank_summary(track: dict) -> str:
    reasons = []
    if int(track.get("ending_score", 0)) >= 14:
        reasons.append("cleaner ending")
    elif int(track.get("ending_score", 0)) <= 7:
        reasons.append("ending needs review")

    if int(track.get("duration_score", 0)) >= 20:
        reasons.append("close to target")
    elif int(track.get("duration_score", 0)) <= 10:
        reasons.append("runtime drift")

    if track.get("has_outro"):
        reasons.append("outro cue present")
    if int(track.get("clip_score", 0)) >= 10:
        reasons.append("low clip risk")
    if int(track.get("structure_bonus", 0)) >= 4:
        reasons.append("strong structure")

    if not reasons:
        reasons.append("balanced overall take")
    return " • ".join(reasons[:3])


def rerank_sort_key(track: dict):
    return (
        int(track.get("quality_score", 0)),
        int(track.get("ending_score", 0)),
        int(track.get("duration_score", 0)),
        int(track.get("structure_bonus", 0)),
        1 if track.get("has_outro") else 0,
        int(track.get("clip_score", 0)),
        -int(track.get("candidate_index") or 999),
    )


def annotate_candidate_tracks(tracks: list[dict]) -> list[dict]:
    annotated = []
    for rank, track in enumerate(tracks, start=1):
        enriched = dict(track)
        enriched["rank_position"] = rank
        enriched["rerank_summary"] = build_rerank_summary(enriched)
        annotated.append(enriched)
    return annotated


def rerank_candidate_tracks(tracks: list[dict]) -> list[dict]:
    reranked = sorted(tracks, key=rerank_sort_key, reverse=True)
    return annotate_candidate_tracks(reranked)


def render_quality_badge(track: dict) -> str:
    quality_score = track.get("quality_score")
    quality_label = html.escape(track.get("quality_label", "Scored"))
    quality_class = html.escape(track.get("quality_class", "solid"))
    if quality_score is None:
        return ""
    return (
        f"<span class='quality-badge quality-badge--{quality_class}'>"
        f"{quality_score}<small>/100</small> {quality_label}"
        "</span>"
    )


def render_track_markers(track: dict) -> str:
    markers = []
    if track.get("is_best_take"):
        markers.append("<span class='pick-pill pick-pill--best'>Best Pick</span>")
    if track.get("is_favorite"):
        markers.append("<span class='pick-pill pick-pill--favorite'>Favorite</span>")
    if track.get("is_training_candidate"):
        markers.append("<span class='pick-pill pick-pill--training'>Training</span>")
    if track.get("is_archived"):
        markers.append("<span class='pick-pill pick-pill--archived'>Archived</span>")
    if not markers:
        return ""
    return "<div class='pick-pill-row'>" + "".join(markers) + "</div>"


def favorite_button_copy(track: dict | None) -> str:
    if not track:
        return "Favorite Selected"
    return "Unfavorite Selected" if track.get("is_favorite") else "Favorite Selected"


def archive_button_copy(track: dict | None) -> str:
    if not track:
        return "Archive Selected"
    return "Restore Selected" if track.get("is_archived") else "Archive Selected"


def training_button_copy(track: dict | None) -> str:
    if not track:
        return "Mark For Training"
    return "Unmark Training" if track.get("is_training_candidate") else "Mark For Training"


def render_empty_detail(title: str, message: str) -> str:
    return (
        "<div class='studio-card'>"
        f"<h3 class='detail-title'>{html.escape(title)}</h3>"
        f"<p class='detail-meta'>{html.escape(message)}</p>"
        "</div>"
    )


def apply_generation_preset(preset_name: str, lyrics: str, tags: str):
    resolved_name, preset = resolve_generation_preset(preset_name)
    guidance = build_lyrics_guidance(lyrics, preset["max_audio_length_s"])
    prompt_guidance = build_prompt_guidance(lyrics, tags, preset["max_audio_length_s"], resolved_name)
    status = f"Preset applied: {resolved_name}. {preset['note']}"
    return (
        preset["max_audio_length_s"],
        preset["codec_num_steps"],
        preset["codec_guidance_scale"],
        preset["mac_safe"],
        preset["keep_models_loaded"],
        guidance,
        prompt_guidance,
        status,
    )


def metadata_path_for_audio(audio_path: Path) -> Path:
    return audio_path.with_suffix(".json")


def audio_tokens_path_for_audio(audio_path: Path) -> Path:
    return audio_path.with_suffix(".audio_tokens.npy")


def read_track_metadata(audio_path: Path) -> dict:
    metadata_path = metadata_path_for_audio(audio_path)
    if not metadata_path.exists():
        return {}
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def write_track_metadata(audio_path: Path, metadata: dict):
    metadata_path_for_audio(audio_path).write_text(
        json.dumps(metadata, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


def set_track_metadata_fields(audio_path: Path, **updates):
    metadata = read_track_metadata(audio_path)
    metadata.update(updates)
    if "audio_path" not in metadata:
        metadata["audio_path"] = str(audio_path)
    if "title" not in metadata:
        metadata["title"] = audio_path.stem
    write_track_metadata(audio_path, metadata)


def save_track_metadata(
    save_path: Path,
    lyrics: str,
    tags: str,
    max_audio_length_s: int,
    codec_num_steps: int,
    codec_guidance_scale: float,
    mac_safe: bool,
    preset_name: str,
    candidate_group_id: str | None = None,
    candidate_index: int | None = None,
    candidate_count: int | None = None,
    actual_duration_s: float | None = None,
    audio_metrics: dict | None = None,
    quality_score: int | None = None,
    quality_label: str | None = None,
    quality_summary: str | None = None,
    audio_tokens_path: str | None = None,
    finetune_checkpoint: str | None = None,
):
    resolved_audio_tokens_path = str(audio_tokens_path or "").strip()
    if not resolved_audio_tokens_path:
        inferred_audio_tokens_path = audio_tokens_path_for_audio(save_path)
        if inferred_audio_tokens_path.exists():
            resolved_audio_tokens_path = str(inferred_audio_tokens_path)
    metadata = {
        "title": save_path.stem,
        "audio_path": str(save_path),
        "audio_tokens_path": resolved_audio_tokens_path,
        "has_audio_tokens": bool(resolved_audio_tokens_path),
        "finetune_checkpoint": str(finetune_checkpoint or "").strip(),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "duration_s": max_audio_length_s,
        "actual_duration_s": actual_duration_s,
        "tags": tags,
        "lyrics": lyrics,
        "codec_num_steps": codec_num_steps,
        "codec_guidance_scale": codec_guidance_scale,
        "mac_safe": mac_safe,
        "preset_name": preset_name,
        "candidate_group_id": candidate_group_id,
        "candidate_index": candidate_index,
        "candidate_count": candidate_count,
        "quality_score": quality_score,
        "quality_label": quality_label,
        "quality_summary": quality_summary,
        "is_favorite": False,
        "is_best_take": False,
        "is_training_candidate": False,
        "is_archived": False,
    }
    if audio_metrics:
        metadata.update(audio_metrics)
    write_track_metadata(save_path, metadata)


def set_group_best_take(selected_audio_path: str | None):
    if not selected_audio_path:
        return
    selected_path = Path(selected_audio_path)
    selected_metadata = read_track_metadata(selected_path)
    candidate_group_id = selected_metadata.get("candidate_group_id")
    for audio_path in DEFAULT_OUTPUT_DIR.glob("*.mp3"):
        metadata = read_track_metadata(audio_path)
        if not metadata:
            continue
        if candidate_group_id and metadata.get("candidate_group_id") == candidate_group_id:
            metadata["is_best_take"] = str(audio_path) == selected_audio_path
            write_track_metadata(audio_path, metadata)
        elif not candidate_group_id and str(audio_path) == selected_audio_path:
            metadata["is_best_take"] = True
            write_track_metadata(audio_path, metadata)


def toggle_track_favorite(selected_audio_path: str | None) -> bool:
    if not selected_audio_path:
        return False
    audio_path = Path(selected_audio_path)
    metadata = read_track_metadata(audio_path)
    next_value = not bool(metadata.get("is_favorite"))
    metadata["is_favorite"] = next_value
    if "audio_path" not in metadata:
        metadata["audio_path"] = str(audio_path)
    if "title" not in metadata:
        metadata["title"] = audio_path.stem
    write_track_metadata(audio_path, metadata)
    return next_value


def toggle_track_archived(selected_audio_path: str | None) -> bool:
    if not selected_audio_path:
        return False
    audio_path = Path(selected_audio_path)
    metadata = read_track_metadata(audio_path)
    next_value = not bool(metadata.get("is_archived"))
    metadata["is_archived"] = next_value
    if "audio_path" not in metadata:
        metadata["audio_path"] = str(audio_path)
    if "title" not in metadata:
        metadata["title"] = audio_path.stem
    write_track_metadata(audio_path, metadata)
    return next_value


def toggle_track_training_candidate(selected_audio_path: str | None) -> bool:
    if not selected_audio_path:
        return False
    audio_path = Path(selected_audio_path)
    metadata = read_track_metadata(audio_path)
    next_value = not bool(metadata.get("is_training_candidate"))
    metadata["is_training_candidate"] = next_value
    if "audio_path" not in metadata:
        metadata["audio_path"] = str(audio_path)
    if "title" not in metadata:
        metadata["title"] = audio_path.stem
    write_track_metadata(audio_path, metadata)
    return next_value


def delete_track(selected_audio_path: str | None) -> bool:
    if not selected_audio_path:
        return False
    audio_path = Path(selected_audio_path)
    deleted_any = False
    metadata = read_track_metadata(audio_path)
    metadata_path = metadata_path_for_audio(audio_path)
    token_candidates = [audio_tokens_path_for_audio(audio_path)]
    explicit_audio_tokens_path = str(metadata.get("audio_tokens_path", "")).strip()
    if explicit_audio_tokens_path:
        token_candidates.insert(0, Path(explicit_audio_tokens_path).expanduser())
    if audio_path.exists():
        audio_path.unlink()
        deleted_any = True
    for token_path in token_candidates:
        if token_path.exists():
            token_path.unlink()
            deleted_any = True
    if metadata_path.exists():
        metadata_path.unlink()
        deleted_any = True
    return deleted_any


def load_library_tracks():
    tracks = []
    for audio_path in sorted(
        DEFAULT_OUTPUT_DIR.glob("*.mp3"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    ):
        metadata_path = metadata_path_for_audio(audio_path)
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                metadata = {}
        else:
            metadata = {}
        audio_tokens_path = str(metadata.get("audio_tokens_path", "")).strip()
        if not audio_tokens_path:
            inferred_audio_tokens_path = audio_tokens_path_for_audio(audio_path)
            if inferred_audio_tokens_path.exists():
                audio_tokens_path = str(inferred_audio_tokens_path)
        tracks.append(
            {
                "title": metadata.get("title", audio_path.stem),
                "audio_path": str(audio_path),
                "audio_tokens_path": audio_tokens_path,
                "has_audio_tokens": bool(audio_tokens_path),
                "finetune_checkpoint": metadata.get("finetune_checkpoint", ""),
                "created_at": metadata.get(
                    "created_at",
                    datetime.fromtimestamp(audio_path.stat().st_mtime).isoformat(timespec="seconds"),
                ),
                "duration_s": metadata.get("duration_s"),
                "actual_duration_s": metadata.get("actual_duration_s"),
                "tags": metadata.get("tags", ""),
                "lyrics": metadata.get("lyrics", ""),
                "preset_name": metadata.get("preset_name", "Custom"),
                "candidate_group_id": metadata.get("candidate_group_id"),
                "candidate_index": metadata.get("candidate_index"),
                "candidate_count": metadata.get("candidate_count"),
                "audio_peak": metadata.get("audio_peak"),
                "audio_rms": metadata.get("audio_rms"),
                "tail_rms": metadata.get("tail_rms"),
                "tail_ratio": metadata.get("tail_ratio"),
                "silence_ratio": metadata.get("silence_ratio"),
                "quality_score": metadata.get("quality_score"),
                "quality_label": metadata.get("quality_label"),
                "quality_summary": metadata.get("quality_summary"),
                "is_favorite": bool(metadata.get("is_favorite")),
                "is_best_take": bool(metadata.get("is_best_take")),
                "is_training_candidate": bool(metadata.get("is_training_candidate")),
                "is_archived": bool(metadata.get("is_archived")),
            }
        )
    return [enrich_track_quality(track) for track in tracks]


def track_choice_value(track: dict) -> str:
    created = track.get("created_at", "").replace("T", " ")
    duration_s = track.get("duration_s")
    duration_label = f"{duration_s}s" if duration_s else "unknown length"
    quality_score = track.get("quality_score")
    quality_label = f"{quality_score}/100" if isinstance(quality_score, int) else "unscored"
    flags = []
    if track.get("is_best_take"):
        flags.append("best")
    if track.get("is_favorite"):
        flags.append("favorite")
    if track.get("is_training_candidate"):
        flags.append("training")
    if track.get("is_archived"):
        flags.append("archived")
    flag_label = f" | {'/'.join(flags)}" if flags else ""
    return f"{track['title']} | {quality_label}{flag_label} | {duration_label} | {created}"


def cover_style_for_track(track: dict) -> str:
    seed = f"{track.get('title','')}-{track.get('tags','')}-{track.get('preset_name','')}"
    palette = [
        ("#0f172a", "#2563eb", "#38bdf8"),
        ("#1f2937", "#b45309", "#f59e0b"),
        ("#1e293b", "#7c3aed", "#ec4899"),
        ("#111827", "#059669", "#84cc16"),
        ("#172554", "#0891b2", "#f97316"),
    ]
    idx = sum(ord(ch) for ch in seed) % len(palette)
    a, b, c = palette[idx]
    return (
        f"background: radial-gradient(circle at 24% 22%, {c} 0%, transparent 28%), "
        f"linear-gradient(135deg, {a} 0%, {b} 55%, {c} 100%);"
    )


def render_recent_cards(tracks: list[dict]) -> str:
    if not tracks:
        return ""
    cards = []
    for track in tracks[:4]:
        title = html.escape(track.get("title", "Untitled Track"))
        created = html.escape(track.get("created_at", "").replace("T", " "))
        duration_s = track.get("duration_s")
        duration = f"{duration_s}s" if duration_s else "unknown length"
        cards.append(
            "<div class='mini-card bento-cell'>"
            f"{render_track_markers(track)}"
            f"{render_quality_badge(track)}"
            f"<div class='mini-cover' style=\"{cover_style_for_track(track)}\"></div>"
            f"<p class='mini-title'>{title}</p>"
            f"<p class='mini-meta'>{duration} \u2022 {created}</p>"
            "</div>"
        )
    return "<div class='mini-grid'>" + "".join(cards) + "</div>"


def render_pinned_cards(tracks: list[dict]) -> str:
    pinned = []
    for track in tracks:
        if track.get("is_favorite") and not track.get("is_archived"):
            pinned.append(track)
    for track in tracks:
        if track.get("is_best_take") and not track.get("is_archived") and track not in pinned:
            pinned.append(track)
    if not pinned:
        return (
            "<div class='pinned-shell'>"
            "<h3 class='pinned-title'>Pinned Picks</h3>"
            "<p class='pinned-empty'>Favorite tracks and saved best takes will appear here.</p>"
            "</div>"
        )

    cards = []
    for track in pinned[:4]:
        title = html.escape(track.get("title", "Untitled Track"))
        created = html.escape(track.get("created_at", "").replace("T", " "))
        cards.append(
            "<div class='mini-card pinned-card bento-cell'>"
            f"{render_track_markers(track)}"
            f"{render_quality_badge(track)}"
            f"<div class='mini-cover' style=\"{cover_style_for_track(track)}\"></div>"
            f"<p class='mini-title'>{title}</p>"
            f"<p class='mini-meta'>{created}</p>"
            "</div>"
        )
    return (
        "<div class='pinned-shell'>"
        "<h3 class='pinned-title'>Pinned Picks</h3>"
        "<div class='mini-grid'>" + "".join(cards) + "</div>"
        "</div>"
    )


def load_render_history() -> list[dict]:
    with RENDER_STATE_LOCK:
        if not RENDER_HISTORY_PATH.exists():
            return []
        try:
            data = json.loads(RENDER_HISTORY_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        return data if isinstance(data, list) else []


def save_render_history(entries: list[dict]):
    with RENDER_STATE_LOCK:
        RENDER_HISTORY_PATH.write_text(
            json.dumps(entries[:RENDER_HISTORY_LIMIT], ensure_ascii=True, indent=2),
            encoding="utf-8",
        )


def update_render_history(entry_id: str, **fields):
    entries = load_render_history()
    for entry in entries:
        if entry.get("id") == entry_id:
            entry.update(fields)
            break
    save_render_history(entries)


def begin_render_history(
    lyrics: str,
    model_path: str,
    finetune_checkpoint: str,
    output_name: str,
    preset_name: str,
    candidate_count: int,
    target_length_s: int,
    tags: str,
    codec_num_steps: int,
    codec_guidance_scale: float,
    mac_safe: bool,
    keep_models_loaded: bool,
    status: str = "Rendering",
    note: str = "",
) -> str:
    entries = load_render_history()
    created_at = datetime.now().isoformat(timespec="seconds")
    entry_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    tag_preview = ", ".join([piece.strip() for piece in (tags or "").split(",") if piece.strip()][:2])
    entries.insert(
        0,
        {
            "id": entry_id,
            "created_at": created_at,
            "title": Path(output_name).stem or "marks-studio",
            "preset_name": preset_name,
            "candidate_count": candidate_count,
            "target_length_s": target_length_s,
            "tag_preview": tag_preview,
            "status": status,
            "best_score": None,
            "primary_output": None,
            "note": note,
            "lyrics": lyrics,
            "tags": tags,
            "model_path": model_path,
            "finetune_checkpoint": finetune_checkpoint,
            "output_name": output_name,
            "codec_num_steps": codec_num_steps,
            "codec_guidance_scale": codec_guidance_scale,
            "mac_safe": mac_safe,
            "keep_models_loaded": keep_models_loaded,
            "started_at": created_at if status == "Rendering" else None,
            "finished_at": None,
        },
    )
    save_render_history(entries)
    return entry_id


def finish_render_history(
    entry_id: str,
    status: str,
    best_score: int | None = None,
    primary_output: str | None = None,
    note: str = "",
):
    entries = load_render_history()
    for entry in entries:
        if entry.get("id") == entry_id:
            entry["status"] = status
            entry["best_score"] = best_score
            entry["primary_output"] = primary_output
            entry["note"] = note
            if status in {"Ready", "Completed", "Failed"}:
                entry["finished_at"] = datetime.now().isoformat(timespec="seconds")
            break
    save_render_history(entries)


def queued_job_position(entry_id: str, items: list[dict] | None = None) -> int | None:
    entries = items if items is not None else load_render_history()
    queued_ids = [entry.get("id") for entry in reversed(entries) if entry.get("status") == "Queued"]
    if entry_id not in queued_ids:
        return None
    return queued_ids.index(entry_id) + 1


def claim_next_queued_render() -> dict | None:
    entries = load_render_history()
    queued_entries = [entry for entry in reversed(entries) if entry.get("status") == "Queued"]
    if not queued_entries:
        return None
    next_entry_id = queued_entries[0].get("id")
    for entry in entries:
        if entry.get("id") == next_entry_id:
            entry["status"] = "Rendering"
            entry["started_at"] = datetime.now().isoformat(timespec="seconds")
            entry["note"] = "Preparing pipeline"
            claimed = dict(entry)
            save_render_history(entries)
            return claimed
    return None


def render_render_history(entries: list[dict] | None = None) -> str:
    items = entries if entries is not None else load_render_history()
    if not items:
        return (
            "<div class='history-shell'>"
            "<h3 class='pinned-title'>Render Queue</h3>"
            "<p class='pinned-empty'>Queued, active, and finished generation jobs will appear here.</p>"
            "</div>"
        )

    queued_count = sum(1 for entry in items if entry.get("status") == "Queued")
    rendering_count = sum(1 for entry in items if entry.get("status") == "Rendering")
    ready_count = sum(1 for entry in items if entry.get("status") in {"Ready", "Completed"})
    failed_count = sum(1 for entry in items if entry.get("status") == "Failed")
    queue_lookup = {
        entry.get("id"): position
        for position, entry in enumerate(
            [queued_entry for queued_entry in reversed(items) if queued_entry.get("status") == "Queued"],
            start=1,
        )
    }
    cards = []
    for idx, entry in enumerate(items[:8]):
        created = html.escape(str(entry.get("created_at", "")).replace("T", " "))
        title = html.escape(entry.get("title", "Untitled render"))
        preset_name = html.escape(entry.get("preset_name", "Custom"))
        candidate_count = int(entry.get("candidate_count") or 1)
        target_length = entry.get("target_length_s")
        target_copy = f"{target_length}s" if target_length else "custom length"
        best_score = entry.get("best_score")
        tag_preview = html.escape(entry.get("tag_preview", ""))
        status = entry.get("status", "Completed")
        if status == "Rendering":
            status_class = "render-status--running"
            card_extra = "bento-cell--glow-sky"
        elif status == "Queued":
            status_class = "render-status--queued"
            card_extra = ""
        elif status == "Failed":
            status_class = "render-status--failed"
            card_extra = ""
        else:
            status_class = "render-status--done"
            card_extra = "bento-cell--glow-teal"
        note = html.escape(entry.get("note", ""))
        if status == "Queued":
            queue_position = queue_lookup.get(entry.get("id"))
            score_copy = f"Queue #{queue_position}" if queue_position else "Waiting for render slot"
            score_bar = ""
        else:
            score_copy = f"Best {best_score}/100" if isinstance(best_score, int) else "Scoring pending"
            if isinstance(best_score, int):
                bar_color = (
                    "#10b981" if best_score >= 80
                    else "#3b82f6" if best_score >= 60
                    else "#f59e0b" if best_score >= 40
                    else "#ef4444"
                )
                score_bar = (
                    f"<div style='margin-top:8px;height:4px;border-radius:999px;"
                    f"background:rgba(0,0,0,0.08);overflow:hidden;'>"
                    f"<div style='height:100%;width:{best_score}%;border-radius:999px;"
                    f"background:{bar_color};transition:width 0.4s ease;'></div></div>"
                )
            else:
                score_bar = ""
        meta = f"{candidate_count} take(s) \u2022 {target_copy} \u2022 {preset_name}"
        # First card spans 2 columns for featured look
        span_style = " style='grid-column:span 2;'" if idx == 0 else ""
        cards.append(
            f"<div class='history-card bento-cell {card_extra}'{span_style}>"
            f"<div class='history-head'><span class='render-status {status_class}'>{html.escape(status)}</span>"
            f"<span class='history-time'>{created}</span></div>"
            f"<h4 class='history-title'>{title}</h4>"
            f"<p class='history-meta'>{html.escape(meta)}</p>"
            f"<p class='history-score'>{html.escape(score_copy)}</p>"
            + score_bar
            + (f"<p class='history-meta' style='margin-top:6px;'>{tag_preview}</p>" if tag_preview else "")
            + (f"<p class='history-note'>{note}</p>" if note else "")
            + "</div>"
        )
    summary_stats = (
        f"<div style='display:flex;flex-wrap:wrap;gap:8px;margin-bottom:12px;'>"
        f"<span class='bento-stat'>{queued_count} queued</span>"
        f"<span class='bento-stat'>{rendering_count} rendering</span>"
        f"<span class='bento-stat'>{ready_count} ready</span>"
        + (f"<span class='bento-stat' style='color:#b91c1c;'>{failed_count} failed</span>" if failed_count else "")
        + "</div>"
    )
    return (
        "<div class='history-shell'>"
        "<h3 class='pinned-title'>Render Queue</h3>"
        + summary_stats
        + "<div class='history-grid'>" + "".join(cards) + "</div>"
        "</div>"
    )


def load_dashboard(
    selected_audio_path: str | None = None,
    search_query: str = "",
    filter_mode: str = "All Tracks",
):
    return (*library_snapshot(selected_audio_path, search_query, filter_mode), render_render_history())


def render_track_details(track: dict | None) -> str:
    if not track:
        return render_empty_detail("No tracks yet", "Generate a song to start building your local library.")

    chips = []
    if track.get("preset_name"):
        chips.append(f"<span class='chip'>{html.escape(track['preset_name'])}</span>")
    if track.get("duration_s"):
        chips.append(f"<span class='chip'>{track['duration_s']}s</span>")
    if isinstance(track.get("actual_duration_s"), (float, int)):
        chips.append(f"<span class='chip'>{track['actual_duration_s']:.1f}s render</span>")
    if track.get("has_audio_tokens"):
        chips.append("<span class='chip'>audio tokens ready</span>")
    if track.get("is_training_candidate"):
        chips.append("<span class='chip'>training candidate</span>")
    if str(track.get("finetune_checkpoint", "")).strip():
        chips.append("<span class='chip'>fine-tuned checkpoint</span>")
    tag_text = track.get("tags", "").strip()
    if tag_text:
        for tag in [piece.strip() for piece in tag_text.split(",") if piece.strip()][:6]:
            chips.append(f"<span class='chip'>{html.escape(tag)}</span>")

    lyrics = html.escape(track.get("lyrics", "").strip() or "No lyrics stored.")
    created = html.escape(track.get("created_at", "").replace("T", " "))
    title = html.escape(track.get("title", "Untitled Track"))
    quality_summary = html.escape(track.get("quality_summary", "Heuristic score based on timing, lyrics, and ending shape."))
    return (
        "<div class='studio-card'>"
        f"<div class='cover-art' style=\"{cover_style_for_track(track)}\"></div>"
        f"<h3 class='detail-title'>{title}</h3>"
        f"<p class='detail-meta'>Created {created}</p>"
        f"{render_track_markers(track)}"
        "<div class='quality-hero'>"
        f"{render_quality_badge(track)}"
        f"<p class='quality-summary'>{quality_summary}</p>"
        "</div>"
        f"<div class='chip-row'>{''.join(chips)}</div>"
        f"<div class='lyrics-box'>{lyrics}</div>"
        "</div>"
    )


def load_checkpoint_eval_scores() -> dict[str, dict]:
    experiments_root = ROOT_DIR / "training" / "experiments"
    if not experiments_root.exists():
        return {}

    score_map: dict[str, dict] = {}
    report_paths = sorted(
        experiments_root.glob("**/checkpoint_eval*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for report_path in report_paths:
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        report_mtime = report_path.stat().st_mtime
        rows = int(payload.get("rows") or 0)
        manifest = str(payload.get("manifest") or "")
        for variant in payload.get("variants", []):
            checkpoint_value = str(variant.get("checkpoint_path") or "").strip()
            if not checkpoint_value or checkpoint_value == "base":
                continue
            checkpoint_path = Path(checkpoint_value).expanduser()
            if not checkpoint_path.exists():
                continue
            resolved = str(checkpoint_path.resolve())
            candidate = {
                "label": str(variant.get("label") or ""),
                "loss": variant.get("loss"),
                "delta_vs_base": variant.get("delta_vs_base"),
                "rows": rows,
                "manifest": manifest,
                "report_path": str(report_path.resolve()),
                "report_mtime": report_mtime,
            }
            existing = score_map.get(resolved)
            if existing is None or (candidate["rows"], candidate["report_mtime"]) > (
                existing["rows"],
                existing["report_mtime"],
            ):
                score_map[resolved] = candidate
    return score_map


def load_promoted_checkpoint_metadata() -> dict | None:
    metadata_path = ROOT_DIR / "training" / "checkpoints" / "current_best_clean.json"
    if not metadata_path.exists():
        return None
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    checkpoint_value = str(payload.get("promoted_checkpoint") or "").strip()
    if not checkpoint_value:
        return None
    checkpoint_path = Path(checkpoint_value).expanduser()
    if not checkpoint_path.exists():
        return None
    payload["promoted_checkpoint"] = str(checkpoint_path)
    payload["target_checkpoint"] = str(checkpoint_path.resolve())
    return payload


def read_jsonl_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def write_jsonl_rows(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def render_training_harvest_status() -> str:
    tracks = load_library_tracks()
    marked = [track for track in tracks if track.get("is_training_candidate") and not track.get("is_archived")]
    token_ready = [track for track in tracks if track.get("has_audio_tokens") and not track.get("is_archived")]
    eligible = [
        track
        for track in marked
        if track.get("has_audio_tokens")
        and str(track.get("lyrics", "")).strip()
        and str(track.get("tags", "")).strip()
        and isinstance(track.get("quality_score"), int)
        and int(track.get("quality_score")) >= MIN_APP_HARVEST_QUALITY
        and (int(track.get("candidate_count") or 1) <= 1 or track.get("is_best_take"))
    ]
    harvested_rows = read_jsonl_rows(APP_HARVEST_SELECTED_PATH)
    packaged_rows = read_jsonl_rows(APP_HARVEST_MANIFEST_PATH)
    experiment_rows = read_jsonl_rows(APP_EXPERIMENT_MANIFEST_PATH)
    experiment_smoke_rows = read_jsonl_rows(APP_EXPERIMENT_SMOKE_PATH)
    smoke_eval_payload = None
    clean_eval_payload = None
    if APP_EXPERIMENT_SMOKE_EVAL_JSON_PATH.exists():
        try:
            smoke_eval_payload = json.loads(APP_EXPERIMENT_SMOKE_EVAL_JSON_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            smoke_eval_payload = None
    if APP_EXPERIMENT_CLEAN_EVAL_JSON_PATH.exists():
        try:
            clean_eval_payload = json.loads(APP_EXPERIMENT_CLEAN_EVAL_JSON_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            clean_eval_payload = None

    last_harvest_copy = "No harvest yet."
    if APP_HARVEST_SELECTED_PATH.exists():
        harvested_at = datetime.fromtimestamp(APP_HARVEST_SELECTED_PATH.stat().st_mtime).isoformat(timespec="seconds")
        last_harvest_copy = f"Last harvest: {harvested_at.replace('T', ' ')}"
    last_experiment_copy = "No experiment yet."
    if APP_EXPERIMENT_MANIFEST_PATH.exists():
        experiment_at = datetime.fromtimestamp(APP_EXPERIMENT_MANIFEST_PATH.stat().st_mtime).isoformat(timespec="seconds")
        last_experiment_copy = f"Last experiment: {experiment_at.replace('T', ' ')}"
    smoke_eval_copy = "Smoke train not run yet."
    if smoke_eval_payload:
        variants = smoke_eval_payload.get("variants", [])
        best_variant = next((row for row in variants if row.get("label") == "smoke"), None)
        if best_variant is not None:
            delta = best_variant.get("delta_vs_base")
            if isinstance(delta, (int, float)):
                smoke_eval_copy = f"Smoke delta vs base: {delta:+.4f}"
    clean_eval_copy = "Clean cycle not run yet."
    if clean_eval_payload:
        variants = clean_eval_payload.get("variants", [])
        best_variant = next((row for row in variants if row.get("label") == "cycle1"), None)
        if best_variant is not None:
            delta = best_variant.get("delta_vs_base")
            rows = clean_eval_payload.get("rows")
            if isinstance(delta, (int, float)):
                clean_eval_copy = f"Clean delta vs base: {delta:+.4f}"
                if isinstance(rows, int) and rows > 0:
                    clean_eval_copy += f" on {rows} val row(s)"
    experiment_train_rows = [row for row in experiment_rows if str(row.get("split", "train")) == "train"]
    experiment_validation_rows = [row for row in experiment_rows if str(row.get("split", "")) == "validation"]
    readiness_copy = (
        f"Need at least {MIN_APP_EXPERIMENT_ROWS} rows total and {MIN_APP_VALIDATION_ROWS} validation rows for a clean cycle."
        if len(experiment_rows) < MIN_APP_EXPERIMENT_ROWS or len(experiment_validation_rows) < MIN_APP_VALIDATION_ROWS
        else "Enough rows are available for a guarded clean CPU cycle."
    )

    return (
        "<div class='history-shell'>"
        "<h3 class='pinned-title'>Training Harvest</h3>"
        "<p class='history-summary'>"
        f"{len(marked)} marked • {len(token_ready)} token-ready • {len(eligible)} eligible now • {len(experiment_rows)} experiment row(s)"
        "</p>"
        "<div class='history-grid'>"
        "<div class='history-card'>"
        "<span class='render-status'>Curated In App</span>"
        f"<p class='history-title'>{len(marked)} track(s) marked for training</p>"
        "<p class='history-meta'>Use the library button to mark the good tokenized takes.</p>"
        "</div>"
        "<div class='history-card'>"
        "<span class='render-status render-status--ready'>Harvest Snapshot</span>"
        f"<p class='history-title'>{len(harvested_rows)} shortlisted • {len(packaged_rows)} packaged</p>"
        f"<p class='history-meta'>{html.escape(last_harvest_copy)}</p>"
        "</div>"
        "<div class='history-card'>"
        "<span class='render-status'>Experiment Prep</span>"
        f"<p class='history-title'>{len(experiment_train_rows)} train • {len(experiment_validation_rows)} validation • {len(experiment_smoke_rows)} smoke</p>"
        f"<p class='history-meta'>{html.escape(last_experiment_copy)}</p>"
        "</div>"
        "<div class='history-card'>"
        "<span class='render-status'>Smoke Train</span>"
        "<p class='history-title'>1 CPU step on the app smoke subset</p>"
        f"<p class='history-meta'>{html.escape(smoke_eval_copy)}</p>"
        "</div>"
        "<div class='history-card'>"
        "<span class='render-status'>Clean Cycle</span>"
        "<p class='history-title'>Short CPU train + held-out validation</p>"
        f"<p class='history-meta'>{html.escape(clean_eval_copy)} • {html.escape(readiness_copy)}</p>"
        "</div>"
        "<div class='history-card'>"
        "<span class='render-status'>Current Rule</span>"
        f"<p class='history-title'>quality >= {MIN_APP_HARVEST_QUALITY}, lyrics + tags, audio tokens, best take only</p>"
        "<p class='history-meta'>Experiment prep ranks the shortlist by quality, runtime fit, lyric structure, and prompt richness.</p>"
        "</div>"
        "</div>"
        "</div>"
    )


def render_workspace_intro(view: str) -> str:
    config = WORKSPACE_VIEWS.get(view, WORKSPACE_VIEWS["create"])
    return (
        "<div class='workspace-intro'>"
        f"<p class='workspace-intro-top'>{html.escape(config['eyebrow'])}</p>"
        f"<h2>{html.escape(config['title'])}</h2>"
        f"<p>{html.escape(config['copy'])}</p>"
        "</div>"
    )


def workspace_button_update(active_view: str, view: str):
    return gr.update(
        value=WORKSPACE_VIEWS[view]["label"],
        variant="primary" if active_view == view else "secondary",
    )


def switch_workspace(view: str):
    active_view = view if view in WORKSPACE_VIEWS else "create"
    return (
        active_view,
        render_workspace_intro(active_view),
        workspace_button_update(active_view, "create"),
        workspace_button_update(active_view, "library"),
        workspace_button_update(active_view, "training"),
        workspace_button_update(active_view, "transcribe"),
        workspace_button_update(active_view, "settings"),
        gr.update(visible=active_view == "create"),
        gr.update(visible=active_view == "library"),
        gr.update(visible=active_view == "training"),
        gr.update(visible=active_view == "transcribe"),
        gr.update(visible=active_view == "settings"),
    )


def refresh_training_harvest_panel():
    return render_training_harvest_status(), "Training harvest status refreshed."


def run_python_tool(script_name: str, *args: str) -> subprocess.CompletedProcess[str]:
    script_path = ROOT_DIR / "scripts" / script_name
    return subprocess.run(
        [sys.executable, str(script_path), *[str(arg) for arg in args]],
        cwd=str(ROOT_DIR),
        capture_output=True,
        text=True,
        check=False,
    )


def harvest_training_candidates_from_app():
    harvest_result = run_python_tool(
        "harvest_app_training_candidates.py",
        "--source-dir",
        str(DEFAULT_OUTPUT_DIR),
        "--output-jsonl",
        str(APP_HARVEST_SELECTED_PATH),
        "--report-path",
        str(APP_HARVEST_REPORT_PATH),
        "--min-quality",
        str(MIN_APP_HARVEST_QUALITY),
    )
    if harvest_result.returncode != 0:
        error_copy = (harvest_result.stderr or harvest_result.stdout or "Harvest failed.").strip()
        return render_training_harvest_status(), error_copy

    harvested_rows = read_jsonl_rows(APP_HARVEST_SELECTED_PATH)
    status_lines = [(harvest_result.stdout or "Harvest complete.").strip()]
    if harvested_rows:
        package_result = run_python_tool(
            "package_manifest_subset.py",
            "--input-jsonl",
            str(APP_HARVEST_SELECTED_PATH),
            "--output-dir",
            str(APP_HARVEST_DATASET_DIR),
            "--copy-shortlist",
        )
        if package_result.returncode != 0:
            error_copy = (package_result.stderr or package_result.stdout or "Packaging failed.").strip()
            return render_training_harvest_status(), error_copy
        status_lines.append((package_result.stdout or "").strip())

        manifest_result = run_python_tool(
            "build_training_manifest.py",
            "--dataset-root",
            str(APP_HARVEST_DATASET_DIR / "audio"),
            "--output",
            str(APP_HARVEST_MANIFEST_PATH),
            "--require-lyrics",
            "--require-tags",
            "--require-audio-tokens",
        )
        if manifest_result.returncode != 0:
            error_copy = (manifest_result.stderr or manifest_result.stdout or "Manifest build failed.").strip()
            return render_training_harvest_status(), error_copy
        status_lines.append((manifest_result.stdout or "").strip())
    else:
        status_lines.append("No eligible marked tracks were ready to package.")

    return render_training_harvest_status(), "\n".join(line for line in status_lines if line)


def prepare_app_training_experiment():
    harvested_rows = read_jsonl_rows(APP_HARVEST_MANIFEST_PATH)
    if not harvested_rows:
        return render_training_harvest_status(), "No packaged harvest manifest found. Run Harvest Marked first."

    prepare_result = run_python_tool(
        "prepare_training_experiment.py",
        "--manifest",
        str(APP_HARVEST_MANIFEST_PATH),
        "--output-dir",
        str(APP_EXPERIMENT_DIR),
        "--target-length-s",
        "120",
        "--smoke-count",
        "8",
        "--validation-ratio",
        "0.2",
        "--require-audio-tokens",
    )
    if prepare_result.returncode != 0:
        error_copy = (prepare_result.stderr or prepare_result.stdout or "Experiment prep failed.").strip()
        return render_training_harvest_status(), error_copy

    status_lines = [(prepare_result.stdout or "Prepared experiment.").strip()]
    smoke_rows = read_jsonl_rows(APP_EXPERIMENT_SMOKE_PATH)
    if smoke_rows:
        forward_result = run_python_tool(
            "train_heartmula.py",
            "--manifest",
            str(APP_EXPERIMENT_SMOKE_PATH),
            "--model-root",
            str(DEFAULT_MODEL_PATH),
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--strict-audio-tokens",
            "--train-profile",
            "prediction-heads",
            "--forward-only",
            "--summary-path",
            str(APP_EXPERIMENT_FORWARD_SUMMARY_PATH),
        )
        if forward_result.returncode != 0:
            error_copy = (forward_result.stderr or forward_result.stdout or "Forward-only validation failed.").strip()
            return render_training_harvest_status(), error_copy
        status_lines.append((forward_result.stdout or "").strip())
    else:
        status_lines.append("Experiment prepared, but the smoke subset is empty.")

    return render_training_harvest_status(), "\n".join(line for line in status_lines if line)


def train_app_experiment_smoke():
    smoke_rows = read_jsonl_rows(APP_EXPERIMENT_SMOKE_PATH)
    if not smoke_rows:
        return render_training_harvest_status(), "No smoke manifest found. Run Prepare Experiment first."

    promoted = load_promoted_checkpoint_metadata()
    warm_start_args: list[str] = []
    if promoted:
        warm_start_args = ["--init-delta-checkpoint", str(promoted["promoted_checkpoint"])]

    train_result = run_python_tool(
        "train_heartmula.py",
        "--manifest",
        str(APP_EXPERIMENT_SMOKE_PATH),
        "--model-root",
        str(DEFAULT_MODEL_PATH),
        "--batch-size",
        "1",
        "--device",
        "cpu",
        "--dtype",
        "float32",
        "--strict-audio-tokens",
        "--train-profile",
        "prediction-heads",
        "--checkpoint-dir",
        str(APP_EXPERIMENT_SMOKE_CHECKPOINT_DIR),
        "--max-steps",
        "1",
        "--summary-path",
        str(APP_EXPERIMENT_SMOKE_SUMMARY_PATH),
        *warm_start_args,
    )
    if train_result.returncode != 0:
        error_copy = (train_result.stderr or train_result.stdout or "Smoke training failed.").strip()
        return render_training_harvest_status(), error_copy

    checkpoint_path = APP_EXPERIMENT_SMOKE_CHECKPOINT_DIR / "latest.pt"
    if not checkpoint_path.exists():
        return render_training_harvest_status(), "Smoke training completed but no checkpoint was saved."

    eval_result = run_python_tool(
        "evaluate_heartmula_checkpoint.py",
        "--manifest",
        str(APP_EXPERIMENT_SMOKE_PATH),
        "--model-root",
        str(DEFAULT_MODEL_PATH),
        "--device",
        "cpu",
        "--dtype",
        "float32",
        "--variant",
        "base=base",
        "--variant",
        f"smoke={checkpoint_path}",
        "--output-json",
        str(APP_EXPERIMENT_SMOKE_EVAL_JSON_PATH),
        "--output-md",
        str(APP_EXPERIMENT_SMOKE_EVAL_MD_PATH),
    )
    if eval_result.returncode != 0:
        error_copy = (eval_result.stderr or eval_result.stdout or "Smoke evaluation failed.").strip()
        return render_training_harvest_status(), error_copy

    status_lines = [
        (train_result.stdout or "Smoke training complete.").strip(),
        (eval_result.stdout or "").strip(),
    ]
    return render_training_harvest_status(), "\n".join(line for line in status_lines if line)


def train_app_experiment_cycle1():
    experiment_rows = read_jsonl_rows(APP_EXPERIMENT_MANIFEST_PATH)
    if not experiment_rows:
        return render_training_harvest_status(), "No experiment manifest found. Run Prepare Experiment first."

    promoted = load_promoted_checkpoint_metadata()
    warm_start_args: list[str] = []
    if promoted:
        warm_start_args = ["--init-delta-checkpoint", str(promoted["promoted_checkpoint"])]

    train_rows = [row for row in experiment_rows if str(row.get("split", "train")) == "train"]
    validation_rows = [row for row in experiment_rows if str(row.get("split", "")) == "validation"]
    if len(experiment_rows) < MIN_APP_EXPERIMENT_ROWS or len(train_rows) < MIN_APP_TRAIN_ROWS or len(validation_rows) < MIN_APP_VALIDATION_ROWS:
        return (
            render_training_harvest_status(),
            f"Clean cycle blocked. Need at least {MIN_APP_EXPERIMENT_ROWS} total rows, {MIN_APP_TRAIN_ROWS} train rows, and {MIN_APP_VALIDATION_ROWS} validation rows. "
            f"Current experiment has {len(experiment_rows)} total, {len(train_rows)} train, {len(validation_rows)} validation.",
        )

    write_jsonl_rows(APP_EXPERIMENT_TRAIN_MANIFEST_PATH, train_rows)
    write_jsonl_rows(APP_EXPERIMENT_VALIDATION_MANIFEST_PATH, validation_rows)
    max_steps = max(2, min(4, len(train_rows)))

    train_result = run_python_tool(
        "train_heartmula.py",
        "--manifest",
        str(APP_EXPERIMENT_TRAIN_MANIFEST_PATH),
        "--validation-manifest",
        str(APP_EXPERIMENT_VALIDATION_MANIFEST_PATH),
        "--model-root",
        str(DEFAULT_MODEL_PATH),
        "--batch-size",
        "1",
        "--device",
        "cpu",
        "--dtype",
        "float32",
        "--strict-audio-tokens",
        "--train-profile",
        "prediction-heads",
        "--checkpoint-dir",
        str(APP_EXPERIMENT_CLEAN_CHECKPOINT_DIR),
        "--max-steps",
        str(max_steps),
        "--summary-path",
        str(APP_EXPERIMENT_CLEAN_SUMMARY_PATH),
        *warm_start_args,
    )
    if train_result.returncode != 0:
        error_copy = (train_result.stderr or train_result.stdout or "Clean cycle training failed.").strip()
        return render_training_harvest_status(), error_copy

    checkpoint_path = APP_EXPERIMENT_CLEAN_CHECKPOINT_DIR / "latest.pt"
    if not checkpoint_path.exists():
        return render_training_harvest_status(), "Clean cycle completed but no checkpoint was saved."

    eval_result = run_python_tool(
        "evaluate_heartmula_checkpoint.py",
        "--manifest",
        str(APP_EXPERIMENT_VALIDATION_MANIFEST_PATH),
        "--model-root",
        str(DEFAULT_MODEL_PATH),
        "--device",
        "cpu",
        "--dtype",
        "float32",
        "--variant",
        "base=base",
        "--variant",
        f"cycle1={checkpoint_path}",
        "--output-json",
        str(APP_EXPERIMENT_CLEAN_EVAL_JSON_PATH),
        "--output-md",
        str(APP_EXPERIMENT_CLEAN_EVAL_MD_PATH),
    )
    if eval_result.returncode != 0:
        error_copy = (eval_result.stderr or eval_result.stdout or "Clean cycle evaluation failed.").strip()
        return render_training_harvest_status(), error_copy

    status_lines = [
        f"Ran clean cycle with {len(train_rows)} train row(s), {len(validation_rows)} validation row(s), and {max_steps} CPU step(s).",
        (train_result.stdout or "Clean cycle training complete.").strip(),
        (eval_result.stdout or "").strip(),
    ]
    return render_training_harvest_status(), "\n".join(line for line in status_lines if line)


def list_saved_finetune_checkpoints() -> list[tuple[str, str]]:
    checkpoint_root = ROOT_DIR / "training" / "checkpoints"
    options: list[tuple[str, str]] = [("Base Model", "")]
    score_map = load_checkpoint_eval_scores()
    promoted_checkpoint = load_promoted_checkpoint_metadata()
    if not checkpoint_root.exists():
        return options

    discovered: list[tuple[str, Path]] = []
    if promoted_checkpoint:
        promoted_path = Path(promoted_checkpoint["promoted_checkpoint"]).expanduser()
        promoted_label = "Current Best Clean"
        delta = promoted_checkpoint.get("delta_vs_base")
        selected_label = str(promoted_checkpoint.get("selected_label", "")).strip()
        if isinstance(delta, (int, float)):
            promoted_label += f" • val {delta:+.4f}"
        if selected_label:
            promoted_label += f" • {selected_label}"
        discovered.append((promoted_label, promoted_path))
    for checkpoint_dir in sorted(checkpoint_root.iterdir()):
        if not checkpoint_dir.is_dir():
            continue
        latest_path = checkpoint_dir / "latest.pt"
        if latest_path.exists():
            discovered.append((f"{checkpoint_dir.name} (latest)", latest_path.resolve()))
            continue
        step_paths = sorted(
            checkpoint_dir.glob("step-*.pt"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if step_paths:
            discovered.append((f"{checkpoint_dir.name} ({step_paths[0].name})", step_paths[0].resolve()))

    seen: set[str] = set()
    for label, path in discovered:
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        score_entry = score_map.get(resolved)
        if score_entry:
            delta = score_entry.get("delta_vs_base")
            rows = score_entry.get("rows")
            if isinstance(delta, (int, float)) and " • val " not in label:
                label += f" • val {delta:+.4f}"
            if isinstance(rows, int) and rows > 0:
                label += f" • {rows} row eval"
        size_mb = path.stat().st_size / (1024 ** 2)
        options.append((f"{label} • {size_mb:.0f} MB", str(path)))
    return options


def refresh_finetune_checkpoint_choices(current_value: str = ""):
    normalized_value = str(current_value or "").strip()
    options = list_saved_finetune_checkpoints()
    values = [value for _, value in options]
    selected_value = normalized_value if normalized_value in values else ""
    return gr.update(choices=options, value=selected_value)


def select_finetune_checkpoint(selected_value: str | None):
    return str(selected_value or "")


def render_sticky_player(track: dict | None) -> str:
    if not track:
        return (
            "<div class='sticky-player-info-shell'>"
            "<div class='sticky-player-cover' style=\"background: linear-gradient(135deg, #1e293b 0%, #334155 100%);\"></div>"
            "<div class='sticky-player-copy'>"
            "<p class='sticky-player-eyebrow'>Studio Player</p>"
            "<h3 class='sticky-player-title'>No track selected</h3>"
            "<p class='sticky-player-sub'>Generate a song or choose one from the library to pin playback here.</p>"
            "</div>"
            "</div>"
        )

    title = html.escape(track.get("title", "Untitled Track"))
    created = html.escape(track.get("created_at", "").replace("T", " "))
    duration_s = track.get("duration_s")
    duration = f"{duration_s}s" if duration_s else "unknown length"
    preset_name = html.escape(track.get("preset_name", "Custom"))
    tag_text = track.get("tags", "").strip()
    tag_preview = ", ".join([piece.strip() for piece in tag_text.split(",") if piece.strip()][:2])
    if tag_preview:
        meta = f"{duration} • {preset_name} • {html.escape(tag_preview)}"
    else:
        meta = f"{duration} • {preset_name} • {created}"

    return (
        "<div class='sticky-player-info-shell'>"
        f"<div class='sticky-player-cover' style=\"{cover_style_for_track(track)}\"></div>"
        "<div class='sticky-player-copy'>"
        "<p class='sticky-player-eyebrow'>Studio Player</p>"
        f"<h3 class='sticky-player-title'>{title}</h3>"
        f"<p class='sticky-player-sub'>{meta}</p>"
        f"{render_track_markers(track)}"
        "</div>"
        "</div>"
    )


def tracks_for_paths(candidate_paths: list[str] | None):
    if not candidate_paths:
        return []
    library_map = {track["audio_path"]: track for track in load_library_tracks()}
    ordered = [library_map[path] for path in candidate_paths if path in library_map]
    return annotate_candidate_tracks(ordered)


def candidate_choice_options(tracks: list[dict]):
    options = []
    for idx, track in enumerate(tracks, start=1):
        actual_duration = track.get("actual_duration_s")
        duration_label = (
            f"{actual_duration:.1f}s render"
            if isinstance(actual_duration, (float, int))
            else (f"{track['duration_s']}s target" if track.get("duration_s") else "rendered")
        )
        take_label = track.get("candidate_index") or idx
        quality_score = track.get("quality_score")
        rank_position = track.get("rank_position") or idx
        score_label = f"{quality_score}/100" if isinstance(quality_score, int) else "scored"
        options.append((f"#{rank_position} • Take {take_label} • {score_label} • {duration_label}", track["audio_path"]))
    return options


def render_candidate_compare(tracks: list[dict], selected_audio_path: str | None = None) -> str:
    if not tracks:
        return (
            "<div class='candidate-shell'>"
            "<div class='candidate-empty'>Generate multiple takes to compare versions here. "
            "You will be able to switch the single studio player between candidates and keep the best one.</div>"
            "</div>"
        )

    cards = []
    for idx, track in enumerate(tracks, start=1):
        take_label = track.get("candidate_index") or idx
        rank_position = track.get("rank_position") or idx
        title = html.escape(track.get("title", "Untitled Track"))
        selected_class = " selected" if track.get("audio_path") == selected_audio_path else ""
        actual_duration = track.get("actual_duration_s")
        target_duration = track.get("duration_s")
        if isinstance(actual_duration, (float, int)):
            duration_copy = f"Rendered {actual_duration:.1f}s"
        elif target_duration:
            duration_copy = f"Target {target_duration}s"
        else:
            duration_copy = "Rendered output"
        preset_name = html.escape(track.get("preset_name", "Custom"))
        created = html.escape(track.get("created_at", "").replace("T", " "))
        tag_text = track.get("tags", "").strip()
        chip_values = [preset_name, duration_copy]
        if tag_text:
            chip_values.extend([piece.strip() for piece in tag_text.split(",") if piece.strip()][:2])
        chip_html = "".join(
            f"<span class='candidate-chip'>{html.escape(str(value))}</span>" for value in chip_values if value
        )
        cards.append(
            f"<div class='candidate-card{selected_class}'>"
            "<div class='candidate-card-head'>"
            f"<div class='candidate-swatch' style=\"{cover_style_for_track(track)}\"></div>"
            "<div class='candidate-copy'>"
            f"<p class='candidate-kicker'>Rank #{rank_position} • Take {take_label}</p>"
            f"<h4 class='candidate-name'>{title}</h4>"
            f"<p class='candidate-meta'>{html.escape(duration_copy)} • {created}</p>"
            f"{render_track_markers(track)}"
            f"<p class='candidate-score-copy'>{html.escape(track.get('rerank_summary', track.get('quality_summary', 'Heuristic score pending.')))}</p>"
            "</div>"
            f"{render_quality_badge(track)}"
            "</div>"
            f"<div class='candidate-chip-row'>{chip_html}</div>"
            "</div>"
        )

    total = len(tracks)
    return (
        "<div class='candidate-shell'>"
        "<div class='candidate-header'>"
        "<div>"
        "<h3 class='candidate-title-main'>Candidate Compare</h3>"
        "<p class='candidate-sub'>The best-scored take is preselected automatically. Switch the selector below to audition each take in the single studio player. Scores are heuristics based on timing, lyric fit, and ending shape.</p>"
        "</div>"
        f"<p class='candidate-sub'>{total} take(s)</p>"
        "</div>"
        f"<div class='candidate-grid'>{''.join(cards)}</div>"
        "</div>"
    )


def filter_library_tracks(tracks: list[dict], search_query: str = "", filter_mode: str = "All Tracks") -> list[dict]:
    query = (search_query or "").strip().lower()
    filtered = tracks
    if filter_mode != "Archived":
        filtered = [track for track in filtered if not track.get("is_archived")]
    if query:
        filtered = [track for track in filtered if query in track.get("search_blob", "")]

    if filter_mode == "Archived":
        filtered = [track for track in tracks if track.get("is_archived")]
        if query:
            filtered = [track for track in filtered if query in track.get("search_blob", "")]
    elif filter_mode == "Favorites":
        filtered = [track for track in filtered if track.get("is_favorite")]
    elif filter_mode == "Training Candidates":
        filtered = [track for track in filtered if track.get("is_training_candidate")]
    elif filter_mode == "Token Ready":
        filtered = [track for track in filtered if track.get("has_audio_tokens")]
    elif filter_mode == "Best Picks":
        filtered = [track for track in filtered if track.get("is_best_take")]
        filtered.sort(key=lambda track: (int(track.get("quality_score", 0)), track.get("created_at", "")), reverse=True)
    elif filter_mode == "Top Rated":
        filtered = [track for track in filtered if int(track.get("quality_score", 0)) >= 80]
        filtered.sort(key=lambda track: (int(track.get("quality_score", 0)), track.get("created_at", "")), reverse=True)
    elif filter_mode == "Has Outro":
        filtered = [track for track in filtered if track.get("has_outro")]
    elif filter_mode == "Multi-Take Sets":
        filtered = [track for track in filtered if int(track.get("candidate_count") or 0) > 1]

    return filtered


def library_snapshot(
    selected_audio_path: str | None = None,
    search_query: str = "",
    filter_mode: str = "All Tracks",
):
    search_query = str(search_query or "")
    filter_mode = str(filter_mode or "All Tracks")
    tracks = load_library_tracks()
    if not tracks:
        return (
            gr.update(choices=[], value=None),
            render_track_details(None),
            None,
            None,
            "Library is empty. Generate a track to populate it.",
            "",
            render_sticky_player(None),
            None,
            render_pinned_cards([]),
            gr.update(value=favorite_button_copy(None), interactive=False),
            gr.update(value=training_button_copy(None), interactive=False),
            gr.update(value=archive_button_copy(None), interactive=False),
            gr.update(value="Delete Selected", interactive=False),
        )

    visible_tracks = filter_library_tracks(tracks, search_query, filter_mode)
    if not visible_tracks:
        filter_copy = f" in {filter_mode.lower()}" if filter_mode != "All Tracks" else ""
        query_copy = f" for '{search_query.strip()}'" if search_query.strip() else ""
        return (
            gr.update(choices=[], value=None),
            render_empty_detail("No matching tracks", f"No library tracks matched{query_copy}{filter_copy}."),
            None,
            None,
            f"Showing 0 of {len(tracks)} track(s).",
            "",
            render_sticky_player(None),
            None,
            render_pinned_cards(tracks),
            gr.update(value=favorite_button_copy(None), interactive=False),
            gr.update(value=training_button_copy(None), interactive=False),
            gr.update(value=archive_button_copy(None), interactive=False),
            gr.update(value="Delete Selected", interactive=False),
        )

    selected_track = next(
        (track for track in visible_tracks if track["audio_path"] == selected_audio_path),
        visible_tracks[0],
    )
    choices = [(track_choice_value(track), track["audio_path"]) for track in visible_tracks]
    status = f"Showing {len(visible_tracks)} of {len(tracks)} track(s)"
    if filter_mode != "All Tracks":
        status += f" • {filter_mode}"
    if search_query.strip():
        status += f" • search: {search_query.strip()}"
    return (
        gr.update(choices=choices, value=selected_track["audio_path"]),
        render_track_details(selected_track),
        selected_track["audio_path"],
        selected_track["audio_path"],
        status,
        render_recent_cards(visible_tracks),
        render_sticky_player(selected_track),
        selected_track["audio_path"],
        render_pinned_cards(tracks),
        gr.update(value=favorite_button_copy(selected_track), interactive=True),
        gr.update(value=training_button_copy(selected_track), interactive=True),
        gr.update(value=archive_button_copy(selected_track), interactive=True),
        gr.update(value="Delete Selected", interactive=True),
    )


def toggle_library_favorite(
    selected_audio_path: str | None,
    search_query: str = "",
    filter_mode: str = "All Tracks",
):
    if not selected_audio_path:
        return library_snapshot(None, search_query, filter_mode)
    next_value = toggle_track_favorite(selected_audio_path)
    snapshot = list(library_snapshot(selected_audio_path, search_query, filter_mode))
    action_copy = "Added to favorites" if next_value else "Removed from favorites"
    snapshot[4] = f"{snapshot[4]} • {action_copy}"
    return tuple(snapshot)


def toggle_library_training_candidate(
    selected_audio_path: str | None,
    search_query: str = "",
    filter_mode: str = "All Tracks",
):
    if not selected_audio_path:
        return (*library_snapshot(None, search_query, filter_mode), render_training_harvest_status(), "No track selected.")
    audio_path = Path(selected_audio_path)
    metadata = read_track_metadata(audio_path)
    if not metadata.get("has_audio_tokens") and not audio_tokens_path_for_audio(audio_path).exists():
        snapshot = list(library_snapshot(selected_audio_path, search_query, filter_mode))
        snapshot[4] = f"{snapshot[4]} • Training mark requires audio tokens. Regenerate this track after the token saver update."
        return (*snapshot, render_training_harvest_status(), "Training mark requires audio tokens.")
    next_value = toggle_track_training_candidate(selected_audio_path)
    snapshot = list(library_snapshot(selected_audio_path, search_query, filter_mode))
    action_copy = "Marked for training harvest" if next_value else "Removed from training harvest"
    snapshot[4] = f"{snapshot[4]} • {action_copy}"
    return (*snapshot, render_training_harvest_status(), action_copy)


def toggle_library_archive(
    selected_audio_path: str | None,
    search_query: str = "",
    filter_mode: str = "All Tracks",
):
    if not selected_audio_path:
        return library_snapshot(None, search_query, filter_mode)
    next_value = toggle_track_archived(selected_audio_path)
    snapshot = list(library_snapshot(selected_audio_path, search_query, filter_mode))
    action_copy = "Archived selected track" if next_value else "Restored selected track"
    snapshot[4] = f"{snapshot[4]} • {action_copy}"
    return tuple(snapshot)


def delete_library_track(
    selected_audio_path: str | None,
    search_query: str = "",
    filter_mode: str = "All Tracks",
):
    if not selected_audio_path:
        return library_snapshot(None, search_query, filter_mode)
    deleted = delete_track(selected_audio_path)
    snapshot = list(library_snapshot(None, search_query, filter_mode))
    if deleted:
        snapshot[4] = f"{snapshot[4]} • Deleted selected track"
    else:
        snapshot[4] = f"{snapshot[4]} • Track files were already missing"
    return tuple(snapshot)


def select_candidate(
    audio_path: str | None,
    candidate_paths: list[str] | None,
    search_query: str = "",
    filter_mode: str = "All Tracks",
):
    return _candidate_selection_payload(audio_path, candidate_paths, search_query, filter_mode)


def _candidate_selection_payload(
    audio_path: str | None,
    candidate_paths: list[str] | None,
    search_query: str = "",
    filter_mode: str = "All Tracks",
):
    tracks = tracks_for_paths(candidate_paths)
    if not tracks:
        return (
            None,
            None,
            *library_snapshot(None, search_query, filter_mode),
            render_candidate_compare([]),
        )

    selected_audio_path = audio_path or tracks[0]["audio_path"]
    set_group_best_take(selected_audio_path)
    tracks = tracks_for_paths(candidate_paths)
    compare_html = render_candidate_compare(tracks, selected_audio_path)
    return (
        selected_audio_path,
        selected_audio_path,
        *library_snapshot(selected_audio_path, search_query, filter_mode),
        compare_html,
    )


def _candidate_choice_update(candidate_paths: list[str] | None, selected_audio_path: str | None):
    tracks = tracks_for_paths(candidate_paths)
    return gr.update(
        choices=candidate_choice_options(tracks),
        value=selected_audio_path,
        visible=len(tracks) > 1,
    )


def step_candidate(
    direction: int,
    selected_audio_path: str | None,
    candidate_paths: list[str] | None,
    search_query: str = "",
    filter_mode: str = "All Tracks",
):
    tracks = tracks_for_paths(candidate_paths)
    if not tracks:
        payload = _candidate_selection_payload(None, candidate_paths, search_query, filter_mode)
        return (*payload, _candidate_choice_update(candidate_paths, None))

    path_order = [track["audio_path"] for track in tracks]
    try:
        current_index = path_order.index(selected_audio_path or path_order[0])
    except ValueError:
        current_index = 0
    next_index = (current_index + direction) % len(path_order)
    next_audio_path = path_order[next_index]
    payload = _candidate_selection_payload(next_audio_path, candidate_paths, search_query, filter_mode)
    return (*payload, _candidate_choice_update(candidate_paths, next_audio_path))


def previous_candidate(
    selected_audio_path: str | None,
    candidate_paths: list[str] | None,
    search_query: str = "",
    filter_mode: str = "All Tracks",
):
    return step_candidate(-1, selected_audio_path, candidate_paths, search_query, filter_mode)


def next_candidate(
    selected_audio_path: str | None,
    candidate_paths: list[str] | None,
    search_query: str = "",
    filter_mode: str = "All Tracks",
):
    return step_candidate(1, selected_audio_path, candidate_paths, search_query, filter_mode)


def favorite_candidate(
    selected_audio_path: str | None,
    candidate_paths: list[str] | None,
    search_query: str = "",
    filter_mode: str = "All Tracks",
):
    if selected_audio_path:
        toggle_track_favorite(selected_audio_path)
    payload = _candidate_selection_payload(selected_audio_path, candidate_paths, search_query, filter_mode)
    payload_list = list(payload)
    if selected_audio_path:
        payload_list[6] = f"{payload_list[6]} • Candidate favorite updated"
    return (*payload_list, _candidate_choice_update(candidate_paths, selected_audio_path))


def get_generation_pipeline(
    model_path: str,
    finetune_checkpoint: str,
    mula_device: torch.device,
    codec_device: torch.device,
    mula_dtype: torch.dtype,
    codec_dtype: torch.dtype,
    lazy_load: bool,
) -> HeartMuLaGenPipeline:
    key = (
        str(Path(model_path).resolve()),
        str(Path(finetune_checkpoint).expanduser().resolve()) if finetune_checkpoint.strip() else "",
        str(mula_device),
        str(codec_device),
        str(mula_dtype),
        str(codec_dtype),
        lazy_load,
    )
    if key not in GENERATION_PIPELINES:
        GENERATION_PIPELINES[key] = HeartMuLaGenPipeline.from_pretrained(
            model_path,
            device={"mula": mula_device, "codec": codec_device},
            dtype={"mula": mula_dtype, "codec": codec_dtype},
            version="3B",
            lazy_load=lazy_load,
            heartmula_delta_path=finetune_checkpoint.strip() or None,
        )
    return GENERATION_PIPELINES[key]


def get_transcription_pipeline(
    model_path: str,
    device: torch.device,
    dtype: torch.dtype,
) -> HeartTranscriptorPipeline:
    key = (
        str(Path(model_path).resolve()),
        str(device),
        str(dtype),
    )
    if key not in TRANSCRIPTION_PIPELINES:
        TRANSCRIPTION_PIPELINES[key] = HeartTranscriptorPipeline.from_pretrained(
            model_path,
            device=device,
            dtype=dtype,
        )
    return TRANSCRIPTION_PIPELINES[key]


def update_generation_progress(progress, value: float, desc: str):
    if progress is not None:
        progress(value, desc=desc)


def run_generation_job(
    lyrics: str,
    tags: str,
    model_path: str,
    finetune_checkpoint: str,
    output_name: str,
    max_audio_length_s: int,
    candidate_count: int,
    codec_num_steps: int,
    codec_guidance_scale: float,
    mac_safe: bool,
    keep_models_loaded: bool,
    preset_name: str,
    render_job_id: str | None = None,
    progress=None,
    persist_drafts: bool = True,
) -> dict:
    if not lyrics.strip():
        raise gr.Error("Lyrics are required.")
    if not tags.strip():
        raise gr.Error("Tags are required.")

    model_root = Path(model_path).expanduser().resolve()
    if not model_root.exists():
        raise gr.Error(f"Model path does not exist: {model_root}")
    finetune_checkpoint_text = str(finetune_checkpoint or "").strip()
    if finetune_checkpoint_text:
        resolved_finetune_checkpoint = Path(finetune_checkpoint_text).expanduser().resolve()
        if not resolved_finetune_checkpoint.exists():
            raise gr.Error(f"Fine-tune checkpoint does not exist: {resolved_finetune_checkpoint}")
        finetune_checkpoint_text = str(resolved_finetune_checkpoint)

    save_name = sanitize_filename(output_name, "marks-studio.mp3")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_stem = Path(save_name).stem
    candidate_total = max(1, int(candidate_count))

    mula_device, codec_device = default_devices(mac_safe)
    mula_dtype, codec_dtype = default_dtypes(mula_device, codec_device)
    lazy_load = not keep_models_loaded

    if render_job_id:
        update_render_history(
            render_job_id,
            status="Rendering",
            started_at=datetime.now().isoformat(timespec="seconds"),
            note="Preparing pipeline",
        )
    update_generation_progress(progress, 0, "Preparing pipeline")

    if persist_drafts:
        save_generate_drafts(lyrics, tags)

    pipe = get_generation_pipeline(
        str(model_root),
        finetune_checkpoint_text,
        mula_device,
        codec_device,
        mula_dtype,
        codec_dtype,
        lazy_load,
    )

    generated_paths = []
    update_generation_progress(progress, 0, "Running generation")
    cfg_scale = 1.0 if mac_safe else 1.5
    prepared_inputs = pipe.prepare_inputs({"lyrics": lyrics, "tags": tags}, cfg_scale=cfg_scale)
    pending_decodes: list[tuple[int, Path, dict]] = []
    for idx in range(candidate_total):
        take_number = idx + 1
        if candidate_total == 1:
            save_path = DEFAULT_OUTPUT_DIR / f"{timestamp}-{save_name}"
        else:
            save_path = DEFAULT_OUTPUT_DIR / f"{timestamp}-{base_stem}-take-{take_number}.mp3"

        stage_note = f"Rendering take {take_number}/{candidate_total}"
        update_generation_progress(progress, idx / candidate_total, stage_note)
        if render_job_id:
            update_render_history(render_job_id, note=stage_note)

        with torch.no_grad():
            model_outputs = pipe.generate_frames_from_prepared(
                prepared_inputs,
                max_audio_length_ms=max_audio_length_s * 1000,
                cfg_scale=cfg_scale,
                codec_num_steps=codec_num_steps,
                codec_guidance_scale=codec_guidance_scale,
                show_progress=False,
            )
        pending_decodes.append(
            (
                take_number,
                save_path,
                {"frames": model_outputs["frames"].detach().cpu()},
            )
        )

    if pipe.mula_device != pipe.codec_device and pipe.codec_device.type == "cpu":
        pipe.release_generation_model()

    for idx, (take_number, save_path, model_outputs) in enumerate(pending_decodes):
        stage_note = f"Decoding take {take_number}/{candidate_total}"
        update_generation_progress(progress, (candidate_total + idx) / (candidate_total * 2), stage_note)
        if render_job_id:
            update_render_history(render_job_id, note=stage_note)

        with torch.no_grad():
            generation_result = pipe.decode_frames(
                model_outputs,
                save_path=str(save_path),
                codec_num_steps=codec_num_steps,
                codec_guidance_scale=codec_guidance_scale,
                save_audio_tokens=True,
            )
        audio_tokens_path = ""
        if isinstance(generation_result, dict):
            audio_tokens_path = str(generation_result.get("audio_tokens_path", "")).strip()

        try:
            actual_duration_s = round(float(sf.info(str(save_path)).duration), 1)
        except RuntimeError:
            actual_duration_s = None

        audio_metrics = analyze_audio_output(save_path)
        quality_info = compute_track_quality(
            {
                "lyrics": lyrics,
                "tags": tags,
                "duration_s": max_audio_length_s,
                "actual_duration_s": actual_duration_s,
                "preset_name": preset_name,
                **audio_metrics,
            }
        )

        save_track_metadata(
            save_path,
            lyrics,
            tags,
            max_audio_length_s,
            codec_num_steps,
            codec_guidance_scale,
            mac_safe,
            preset_name,
            candidate_group_id=timestamp,
            candidate_index=take_number,
            candidate_count=candidate_total,
            actual_duration_s=actual_duration_s,
            audio_metrics=audio_metrics,
            quality_score=quality_info["quality_score"],
            quality_label=quality_info["quality_label"],
            quality_summary=quality_info["quality_summary"],
            audio_tokens_path=audio_tokens_path,
            finetune_checkpoint=finetune_checkpoint_text,
        )
        generated_paths.append(str(save_path))

    candidate_tracks = rerank_candidate_tracks(tracks_for_paths(generated_paths))
    generated_paths = [track["audio_path"] for track in candidate_tracks]
    primary_track = candidate_tracks[0]
    primary_path = primary_track["audio_path"]
    best_score = int(primary_track.get("quality_score", 0))
    best_take_number = int(primary_track.get("candidate_index") or 1)
    set_group_best_take(primary_path)

    status = (
        f"Generated {candidate_total} take(s)\n"
        f"Primary output: {primary_path}\n"
        f"Auto-selected best take: {best_take_number} ({best_score}/100) • {primary_track.get('rerank_summary', 'best overall score')}\n"
        f"Audio tokens saved: yes\n"
        f"Preset: {preset_name}\n"
        f"Fine-tune checkpoint: {finetune_checkpoint_text or 'base model'}\n"
        f"Devices: mula={mula_device}, codec={codec_device}\n"
        f"Dtypes: mula={mula_dtype}, codec={codec_dtype}\n"
        f"Codec steps: {codec_num_steps}\n"
        f"Keep models loaded: {keep_models_loaded}"
    )
    return {
        "status": status,
        "primary_path": primary_path,
        "best_score": best_score,
        "best_take_number": best_take_number,
        "primary_track": primary_track,
        "generated_paths": generated_paths,
        "candidate_tracks": candidate_tracks,
    }


def render_queue_worker():
    while True:
        if not RENDER_EXECUTION_LOCK.acquire(blocking=False):
            time.sleep(RENDER_QUEUE_POLL_SECONDS)
            continue

        claimed_job = None
        try:
            claimed_job = claim_next_queued_render()
            if claimed_job:
                try:
                    result = run_generation_job(
                        str(claimed_job.get("lyrics", "")),
                        str(claimed_job.get("tags", "")),
                        str(claimed_job.get("model_path", DEFAULT_MODEL_PATH)),
                        str(claimed_job.get("finetune_checkpoint", "")),
                        str(claimed_job.get("output_name", "marks-studio.mp3")),
                        int(claimed_job.get("target_length_s") or 120),
                        int(claimed_job.get("candidate_count") or 1),
                        int(claimed_job.get("codec_num_steps") or 6),
                        float(claimed_job.get("codec_guidance_scale") or 1.25),
                        bool(claimed_job.get("mac_safe", True)),
                        bool(claimed_job.get("keep_models_loaded", True)),
                        str(claimed_job.get("preset_name", "Balanced 2 Min")),
                        render_job_id=str(claimed_job.get("id")),
                        persist_drafts=False,
                    )
                    finish_render_history(
                        str(claimed_job.get("id")),
                        "Ready",
                        best_score=result["best_score"],
                        primary_output=result["primary_path"],
                        note=f"Best take {result['best_take_number']} selected automatically.",
                    )
                except Exception as exc:
                    finish_render_history(str(claimed_job.get("id")), "Failed", note=str(exc))
        finally:
            RENDER_EXECUTION_LOCK.release()

        time.sleep(0.25 if claimed_job else RENDER_QUEUE_POLL_SECONDS)


def ensure_render_queue_worker():
    global RENDER_QUEUE_WORKER
    if RENDER_QUEUE_WORKER and RENDER_QUEUE_WORKER.is_alive():
        return
    RENDER_QUEUE_WORKER = threading.Thread(
        target=render_queue_worker,
        name="marks-studio-render-worker",
        daemon=True,
    )
    RENDER_QUEUE_WORKER.start()


def queue_music_render(
    lyrics: str,
    tags: str,
    model_path: str,
    finetune_checkpoint: str,
    output_name: str,
    max_audio_length_s: int,
    candidate_count: int,
    codec_num_steps: int,
    codec_guidance_scale: float,
    mac_safe: bool,
    keep_models_loaded: bool,
    preset_name: str,
):
    if not lyrics.strip():
        raise gr.Error("Lyrics are required.")
    if not tags.strip():
        raise gr.Error("Tags are required.")

    model_root = Path(model_path).expanduser().resolve()
    if not model_root.exists():
        raise gr.Error(f"Model path does not exist: {model_root}")
    finetune_checkpoint_text = str(finetune_checkpoint or "").strip()
    if finetune_checkpoint_text:
        resolved_finetune_checkpoint = Path(finetune_checkpoint_text).expanduser().resolve()
        if not resolved_finetune_checkpoint.exists():
            raise gr.Error(f"Fine-tune checkpoint does not exist: {resolved_finetune_checkpoint}")
        finetune_checkpoint_text = str(resolved_finetune_checkpoint)

    save_name = sanitize_filename(output_name, "marks-studio.mp3")
    candidate_total = max(1, int(candidate_count))
    render_job_id = begin_render_history(
        lyrics=lyrics,
        model_path=str(model_root),
        finetune_checkpoint=finetune_checkpoint_text,
        output_name=save_name,
        preset_name=preset_name,
        candidate_count=candidate_total,
        target_length_s=max_audio_length_s,
        tags=tags,
        codec_num_steps=codec_num_steps,
        codec_guidance_scale=codec_guidance_scale,
        mac_safe=mac_safe,
        keep_models_loaded=keep_models_loaded,
        status="Queued",
        note="Waiting for render slot",
    )
    ensure_render_queue_worker()
    queue_position = queued_job_position(render_job_id)
    status = (
        f"Queued {candidate_total} take(s)\n"
        f"Job position: {queue_position or 1}\n"
        f"Preset: {preset_name}\n"
        f"Checkpoint: {finetune_checkpoint_text or 'base model'}\n"
        f"Target length: {max_audio_length_s}s"
    )
    return status, render_render_history()


def best_quality_render_settings(
    max_audio_length_s: int,
    candidate_count: int,
) -> tuple[int, int, int, float, bool, bool, str]:
    resolved_name, preset = resolve_generation_preset("Best Quality")
    return (
        max(1, int(max_audio_length_s)),
        max(1, int(candidate_count)),
        int(preset["codec_num_steps"]),
        float(preset["codec_guidance_scale"]),
        bool(preset["mac_safe"]),
        bool(preset["keep_models_loaded"]),
        resolved_name,
    )


def generate_music(
    lyrics: str,
    tags: str,
    model_path: str,
    finetune_checkpoint: str,
    output_name: str,
    max_audio_length_s: int,
    candidate_count: int,
    codec_num_steps: int,
    codec_guidance_scale: float,
    mac_safe: bool,
    keep_models_loaded: bool,
    preset_name: str,
    progress=gr.Progress(track_tqdm=True),
):
    render_job_id = begin_render_history(
        lyrics=lyrics,
        model_path=model_path,
        finetune_checkpoint=finetune_checkpoint,
        output_name=output_name,
        preset_name=preset_name,
        candidate_count=max(1, int(candidate_count)),
        target_length_s=max_audio_length_s,
        tags=tags,
        codec_num_steps=codec_num_steps,
        codec_guidance_scale=codec_guidance_scale,
        mac_safe=mac_safe,
        keep_models_loaded=keep_models_loaded,
        status="Rendering",
        note="Waiting for render slot",
    )

    try:
        with RENDER_EXECUTION_LOCK:
            result = run_generation_job(
                lyrics,
                tags,
                model_path,
                finetune_checkpoint,
                output_name,
                max_audio_length_s,
                candidate_count,
                codec_num_steps,
                codec_guidance_scale,
                mac_safe,
                keep_models_loaded,
                preset_name,
                render_job_id=render_job_id,
                progress=progress,
                persist_drafts=True,
            )

        primary_path = result["primary_path"]
        primary_track = result["primary_track"]
        best_score = result["best_score"]
        best_take_number = result["best_take_number"]
        candidate_tracks = result["candidate_tracks"]
        generated_paths = result["generated_paths"]
        set_group_best_take(primary_path)
        (
            library_choice,
            library_details,
            library_audio,
            library_file,
            library_status,
            library_cards,
            sticky_player_info,
            sticky_player_audio,
            pinned_cards,
            favorite_button_state,
            training_button_state,
            archive_button_state,
            delete_button_state,
        ) = library_snapshot(primary_path)
        compare_html = render_candidate_compare(candidate_tracks, primary_path)
        candidate_selector = gr.update(
            choices=candidate_choice_options(candidate_tracks),
            value=primary_path,
            visible=len(candidate_tracks) > 1,
        )
        finish_render_history(
            render_job_id,
            "Ready",
            best_score=best_score,
            primary_output=primary_path,
            note=f"Best take {best_take_number} selected automatically.",
        )
        render_history_html = render_render_history()
        return (
            result["status"],
            primary_path,
            primary_path,
            library_choice,
            library_details,
            library_audio,
            library_file,
            library_status,
            library_cards,
            sticky_player_info,
            sticky_player_audio,
            pinned_cards,
            favorite_button_state,
            training_button_state,
            archive_button_state,
            delete_button_state,
            compare_html,
            candidate_selector,
            generated_paths,
            gr.update(value=""),
            gr.update(value="All Tracks"),
            render_history_html,
        )
    except Exception as exc:
        finish_render_history(render_job_id, "Failed", note=str(exc))
        raise


def render_final_music(
    lyrics: str,
    tags: str,
    model_path: str,
    finetune_checkpoint: str,
    output_name: str,
    max_audio_length_s: int,
    candidate_count: int,
    progress=gr.Progress(track_tqdm=True),
):
    (
        resolved_length_s,
        resolved_candidate_count,
        codec_num_steps,
        codec_guidance_scale,
        mac_safe,
        keep_models_loaded,
        preset_name,
    ) = best_quality_render_settings(max_audio_length_s, candidate_count)
    return generate_music(
        lyrics,
        tags,
        model_path,
        finetune_checkpoint,
        output_name,
        resolved_length_s,
        resolved_candidate_count,
        codec_num_steps,
        codec_guidance_scale,
        mac_safe,
        keep_models_loaded,
        preset_name,
        progress=progress,
    )


def transcribe_music(
    uploaded_audio: str | None,
    music_path: str,
    model_path: str,
    mac_safe: bool,
    progress=gr.Progress(track_tqdm=True),
):
    source_path = uploaded_audio or music_path
    if not source_path:
        raise gr.Error("Upload an audio file or provide a path.")

    resolved_source = Path(source_path).expanduser().resolve()
    if not resolved_source.exists():
        raise gr.Error(f"Audio file does not exist: {resolved_source}")

    model_root = Path(model_path).expanduser().resolve()
    if not model_root.exists():
        raise gr.Error(f"Model path does not exist: {model_root}")

    if mac_safe and mps_available():
        device = torch.device("mps")
        dtype = torch.float16
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.float16
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    progress(0, desc="Preparing transcriptor")
    pipe = get_transcription_pipeline(str(model_root), device, dtype)

    progress(0, desc="Running transcription")
    with torch.no_grad():
        result = pipe(
            str(resolved_source),
            **{
                "max_new_tokens": 256,
                "num_beams": 2,
                "task": "transcribe",
                "condition_on_prev_tokens": False,
                "compression_ratio_threshold": 1.8,
                "temperature": (0.0, 0.1, 0.2, 0.4),
                "logprob_threshold": -1.0,
                "no_speech_threshold": 0.4,
            },
        )

    status = f"Transcribed from {resolved_source}\nDevice: {device}\nDtype: {dtype}"
    return status, result


def build_app() -> gr.Blocks:
    with gr.Blocks(theme=gr.themes.Glass(), css=CUSTOM_CSS, title="mark's Studio") as demo:
        candidate_paths_state = gr.State([])
        workspace_state = gr.State("create")

        with gr.Row(elem_classes=["workspace-shell"]):
            with gr.Column(scale=2, min_width=250, elem_classes=["sidebar-rail"]):
                gr.HTML(
                    """
                    <div class="sidebar-brand-card">
                      <p class="sidebar-brand-top">mark&#39;s Studio</p>
                      <h2 class="sidebar-brand-title">Studio</h2>
                      <p class="sidebar-brand-copy">Separate lanes for creation, library review, training prep, and transcription.</p>
                      <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:14px;">
                        <div style="background:rgba(255,255,255,0.10);border:1px solid rgba(255,255,255,0.16);border-radius:14px;padding:10px 12px;">
                          <p style="margin:0 0 2px 0;font-size:10px;letter-spacing:0.12em;text-transform:uppercase;color:rgba(226,232,240,0.6);">Model</p>
                          <p style="margin:0;font-size:13px;font-weight:700;color:#f8fafc;">3B Params</p>
                        </div>
                        <div style="background:rgba(255,255,255,0.10);border:1px solid rgba(255,255,255,0.16);border-radius:14px;padding:10px 12px;">
                          <p style="margin:0 0 2px 0;font-size:10px;letter-spacing:0.12em;text-transform:uppercase;color:rgba(226,232,240,0.6);">Output</p>
                          <p style="margin:0;font-size:13px;font-weight:700;color:#f8fafc;">MP3 + Tokens</p>
                        </div>
                      </div>
                    </div>
                    """
                )
                with gr.Column(elem_classes=["sidebar-nav-stack"]):
                    nav_create_button = gr.Button("Create", variant="primary", elem_classes=["sidebar-nav-button"])
                    nav_library_button = gr.Button("Library", elem_classes=["sidebar-nav-button"])
                    nav_training_button = gr.Button("Training Lab", elem_classes=["sidebar-nav-button"])
                    nav_transcribe_button = gr.Button("Transcribe", elem_classes=["sidebar-nav-button"])
                    nav_settings_button = gr.Button("Settings", elem_classes=["sidebar-nav-button"])
                gr.HTML(
                    """
                    <div class="sidebar-footnote">
                      <p style="margin:0 0 8px 0;font-size:11px;font-weight:800;letter-spacing:0.12em;text-transform:uppercase;color:#64748b;">Quick Stats</p>
                      <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;">
                        <div style="background:rgba(255,255,255,0.36);border:1px solid rgba(255,255,255,0.54);border-radius:12px;padding:8px 10px;box-shadow:inset 0 1px 0 rgba(255,255,255,0.6);">
                          <p style="margin:0;font-size:10px;color:#64748b;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;">Port</p>
                          <p style="margin:0;font-size:13px;font-weight:700;color:#0f172a;">7860</p>
                        </div>
                        <div style="background:rgba(255,255,255,0.36);border:1px solid rgba(255,255,255,0.54);border-radius:12px;padding:8px 10px;box-shadow:inset 0 1px 0 rgba(255,255,255,0.6);">
                          <p style="margin:0;font-size:10px;color:#64748b;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;">Lang</p>
                          <p style="margin:0;font-size:13px;font-weight:700;color:#0f172a;">ZH EN JP</p>
                        </div>
                      </div>
                    </div>
                    """
                )

            with gr.Column(scale=10, elem_classes=["workspace-main"]):
                gr.HTML(
                    """
                    <div class="hero">
                      <p class="hero-topline">&#10022; Glass Studio for Generative Music</p>
                      <h1>mark's Studio</h1>
                      <p>Where AI meets creativity &#8212; compose lyrics, render original music, and transcribe with precision. All in one focused workspace.</p>
                      <div class="hero-stats">
                        <span class="hero-stat">&#127925; Original Music</span>
                        <span class="hero-stat">&#128221; Precise Lyrics</span>
                        <span class="hero-stat">&#63743; Mac-Friendly</span>
                        <span class="hero-stat">&#127905; Fine-Tune Ready</span>
                        <span class="hero-stat">&#9889; Render Queue</span>
                      </div>
                    </div>
                    <div style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin-bottom:14px;">
                      <div class="bento-cell bento-cell--accent" style="padding:16px 18px;">
                        <p class="bento-label" style="color:rgba(226,232,240,0.7);">Mode</p>
                        <p class="bento-heading bento-heading--dark" style="font-size:18px;">&#127932; Create</p>
                        <p class="bento-copy bento-copy--dark" style="font-size:13px;">Compose &amp; render your next track</p>
                      </div>
                      <div class="bento-cell" style="padding:16px 18px;">
                        <p class="bento-label">Engine</p>
                        <p class="bento-heading" style="font-size:18px;">HeartMuLa 3B</p>
                        <p class="bento-copy" style="font-size:13px;">Open-source music foundation model</p>
                      </div>
                      <div class="bento-cell" style="padding:16px 18px;">
                        <p class="bento-label">Codec</p>
                        <p class="bento-heading" style="font-size:18px;">HeartCodec</p>
                        <p class="bento-copy" style="font-size:13px;">High-quality audio generation</p>
                      </div>
                    </div>
                    """
                )
                workspace_intro = gr.HTML(render_workspace_intro("create"))

                with gr.Column(visible=True, elem_classes=["workspace-view"]) as create_workspace:
                    gr.Markdown(
                        "<div class='panel-note'>For this Mac build, generation uses MPS for HeartMuLa and CPU for HeartCodec. Use a Genre Kit for one-click lyric starter + tag pack + render mode, start with Fast Render to iterate quickly, then switch to Best Quality for the final export. Keep Models Loaded speeds up repeat generations without changing quality, but uses more memory.</div>"
                    )
                    with gr.Row(elem_classes=["studio-grid"]):
                        with gr.Column(scale=7):
                            genre_combo_kit = gr.Dropdown(
                                label="Genre Kit",
                                choices=list(GENRE_COMBO_KITS.keys()),
                                value="Starter Pop Kit",
                            )
                            preset_name = gr.Dropdown(
                                label="Preset",
                                choices=list(GENERATION_PRESETS.keys()),
                                value="Fast Render",
                            )
                            lyric_style_preset = gr.Dropdown(
                                label="Lyric Starter",
                                choices=list(LYRIC_STYLE_PRESETS.keys()),
                                value="Rock Anthem",
                            )
                            tag_style_preset = gr.Dropdown(
                                label="Tag Pack",
                                choices=list(TAG_STYLE_PRESETS.keys()),
                                value="Starter Pop",
                            )
                            with gr.Row():
                                lyrics = gr.Textbox(
                                    label="Lyrics",
                                    lines=16,
                                    value=load_editor_text(DEFAULT_LYRICS_PATH, LYRICS_DRAFT_PATH),
                                )
                                tags = gr.Textbox(
                                    label="Tags",
                                    lines=16,
                                    value=load_editor_text(DEFAULT_TAGS_PATH, TAGS_DRAFT_PATH),
                                )
                            with gr.Row():
                                save_draft_button = gr.Button("Save Draft")
                                apply_genre_kit_button = gr.Button("Apply Genre Kit")
                                apply_lyric_pack_button = gr.Button("Apply Lyric Starter")
                                apply_tag_pack_button = gr.Button("Apply Tag Pack")
                                reload_draft_button = gr.Button("Reload Draft")
                                reset_defaults_button = gr.Button("Reset To Default")
                                optimize_lyrics_button = gr.Button("Trim For Length")
                            editor_status = gr.Textbox(
                                label="Editor State",
                                lines=2,
                                value="Drafts load automatically when present.",
                            )
                            with gr.Row():
                                max_audio_length_s = gr.Slider(
                                    label="Target Length (seconds)",
                                    minimum=1,
                                    maximum=120,
                                    step=1,
                                    value=120,
                                )
                                candidate_count = gr.Dropdown(
                                    label="Candidates",
                                    choices=[1, 2, 3, 4],
                                    value=2,
                                )
                            with gr.Row():
                                generate_button = gr.Button("Generate", variant="primary")
                                render_final_button = gr.Button("Render Final")
                                queue_render_button = gr.Button("Queue Render")
                            generation_status = gr.Textbox(label="Run Status", lines=4)

                        with gr.Column(scale=5):
                            lyrics_guidance = gr.Textbox(
                                label="Lyrics Fit",
                                value=build_lyrics_guidance(
                                    load_editor_text(DEFAULT_LYRICS_PATH, LYRICS_DRAFT_PATH),
                                    GENERATION_PRESETS["Fast Render"]["max_audio_length_s"],
                                ),
                                lines=3,
                                interactive=False,
                            )
                            prompt_guidance = gr.Textbox(
                                label="Prompt Coach",
                                value=build_prompt_guidance(
                                    load_editor_text(DEFAULT_LYRICS_PATH, LYRICS_DRAFT_PATH),
                                    load_editor_text(DEFAULT_TAGS_PATH, TAGS_DRAFT_PATH),
                                    GENERATION_PRESETS["Fast Render"]["max_audio_length_s"],
                                    "Fast Render",
                                ),
                                lines=5,
                                interactive=False,
                            )
                            candidate_compare = gr.HTML(value=render_candidate_compare([]))
                            candidate_choice = gr.Radio(
                                label="Best Take",
                                choices=[],
                                value=None,
                                visible=False,
                            )
                            with gr.Row():
                                previous_take_button = gr.Button("Previous Take")
                                next_take_button = gr.Button("Next Take")
                                favorite_take_button = gr.Button("Favorite Current")

                    generation_audio = gr.Audio(
                        label="Generated Audio",
                        type="filepath",
                        elem_classes=["output-block"],
                        visible=False,
                    )
                    generation_file = gr.File(label="Download Output", elem_classes=["output-block"])
                    with gr.Row(elem_classes=["sticky-player-dock"]):
                        sticky_player_info = gr.HTML(value=render_sticky_player(None))
                        sticky_player_audio = gr.Audio(
                            label="Studio Player",
                            show_label=False,
                            type="filepath",
                            elem_classes=["output-block", "sticky-player-audio"],
                        )

                with gr.Column(visible=False, elem_classes=["workspace-view"]) as library_workspace:
                    gr.HTML(
                        """
                        <div class="library-shell">
                          <h2 class="library-heading">Studio Library</h2>
                          <p class="library-sub">Generate, review, and replay local tracks without leaving the app.</p>
                        </div>
                        """
                    )
                    with gr.Row(elem_classes=["studio-grid"]):
                        with gr.Column(scale=4):
                            library_status = gr.Textbox(label="Library Status", lines=2, interactive=False)
                            with gr.Row():
                                library_refresh_button = gr.Button("Refresh Library")
                                favorite_selected_button = gr.Button("Favorite Selected")
                            with gr.Row():
                                training_selected_button = gr.Button("Mark For Training")
                                archive_selected_button = gr.Button("Archive Selected")
                            delete_selected_button = gr.Button("Delete Selected")
                            with gr.Row():
                                library_search = gr.Textbox(
                                    label="Search Library",
                                    placeholder="Search titles, tags, lyrics, presets",
                                )
                                library_filter = gr.Dropdown(
                                    label="Filter",
                                    choices=LIBRARY_FILTER_OPTIONS,
                                    value="All Tracks",
                                )
                            pinned_cards = gr.HTML(value=render_pinned_cards([]))
                            library_choice = gr.Dropdown(label="Recent Tracks", choices=[], value=None)

                        with gr.Column(scale=6):
                            library_cards = gr.HTML(value="")
                            library_details = gr.HTML(value=render_track_details(None))
                            library_audio = gr.Audio(
                                label="Now Playing",
                                type="filepath",
                                elem_classes=["output-block"],
                                visible=False,
                            )
                            library_file = gr.File(
                                label="Track File",
                                elem_classes=["output-block"],
                                visible=False,
                            )

                with gr.Column(visible=False, elem_classes=["workspace-view"]) as training_workspace:
                    gr.Markdown(
                        "<div class='panel-note'>Use this workspace to review queue activity, harvest strong app outputs, prepare experiments, and run guarded CPU smoke or clean-cycle training.</div>"
                    )
                    render_history_html = gr.HTML(value=render_render_history())
                    training_harvest_status = gr.HTML(value=render_training_harvest_status())
                    with gr.Row():
                        refresh_harvest_button = gr.Button("Refresh Harvest")
                        harvest_training_button = gr.Button("Harvest Marked")
                        prepare_experiment_button = gr.Button("Prepare Experiment")
                        train_smoke_button = gr.Button("Train Smoke")
                        train_cycle_button = gr.Button("Train Clean Cycle")
                    queue_refresh_button = gr.Button("Refresh Queue")
                    training_harvest_log = gr.Textbox(
                        label="Training Prep",
                        lines=5,
                        interactive=False,
                        value="Mark token-ready tracks in the library, then harvest them into a packaged dataset.",
                    )

                with gr.Column(visible=False, elem_classes=["workspace-view"]) as transcribe_workspace:
                    gr.Markdown(
                        "<div class='panel-note'>Upload audio or point to a local file path. HeartTranscriptor checkpoints must be present under ckpt/HeartTranscriptor-oss.</div>"
                    )
                    with gr.Row():
                        uploaded_audio = gr.Audio(label="Upload Audio", type="filepath")
                        music_path = gr.Textbox(
                            label="Or Local Audio Path",
                            value=str(ROOT_DIR / "assets" / "output-mac-fast.mp3"),
                        )
                    gr.Markdown(
                        "<div class='panel-note'>Transcription model path and Mac-safe defaults now live in the Settings workspace.</div>"
                    )
                    transcribe_button = gr.Button("Transcribe", variant="primary")
                    transcription_status = gr.Textbox(label="Run Status", lines=3)
                    transcription_text = gr.Textbox(label="Lyrics", lines=18)

                with gr.Column(visible=False, elem_classes=["workspace-view"]) as settings_workspace:
                    gr.Markdown(
                        "<div class='panel-note'>Save your model paths, checkpoint choice, and runtime defaults here so Create stays focused on writing and rendering.</div>"
                    )
                    settings_overview = gr.HTML(value=render_settings_overview())
                    with gr.Accordion("Studio Defaults", open=True, elem_classes=["advanced-accordion"]):
                        with gr.Row():
                            model_path = gr.Textbox(label="Model Path", value=str(DEFAULT_MODEL_PATH))
                            output_name = gr.Textbox(label="Output Filename", value="marks-studio.mp3")
                        finetune_checkpoint = gr.Textbox(
                            label="Fine-Tune Checkpoint",
                            placeholder="Optional: training/checkpoints/.../step-00001.pt",
                            value="",
                        )
                        with gr.Row():
                            finetune_checkpoint_choice = gr.Dropdown(
                                label="Saved Checkpoints",
                                choices=[("Base Model", "")],
                                value="",
                            )
                            refresh_checkpoint_button = gr.Button("Refresh Checkpoints")
                        with gr.Row():
                            codec_num_steps = gr.Slider(
                                label="Codec Steps",
                                minimum=4,
                                maximum=10,
                                step=1,
                                value=6,
                            )
                            codec_guidance_scale = gr.Slider(
                                label="Codec Guidance",
                                minimum=1.0,
                                maximum=2.0,
                                step=0.05,
                                value=1.25,
                            )
                        with gr.Row():
                            mac_safe = gr.Checkbox(label="Mac Safe", value=True)
                            keep_models_loaded = gr.Checkbox(label="Keep Models Loaded", value=True)
                        with gr.Row():
                            transcribe_model_path = gr.Textbox(
                                label="Transcribe Model Path",
                                value=str(DEFAULT_MODEL_PATH),
                            )
                            transcribe_mac_safe = gr.Checkbox(label="Transcribe Mac Safe", value=True)
                    with gr.Row():
                        save_settings_button = gr.Button("Save Settings", variant="primary")
                        reload_settings_button = gr.Button("Reload Settings")
                        reset_settings_button = gr.Button("Reset Settings")
                    settings_status = gr.Textbox(
                        label="Settings State",
                        lines=2,
                        interactive=False,
                        value="Studio defaults load automatically when present.",
                    )

        preset_name.change(
            fn=apply_generation_preset,
            inputs=[preset_name, lyrics, tags],
            outputs=[
                max_audio_length_s,
                codec_num_steps,
                codec_guidance_scale,
                mac_safe,
                keep_models_loaded,
                lyrics_guidance,
                prompt_guidance,
                editor_status,
            ],
        )
        apply_genre_kit_button.click(
            fn=apply_genre_combo_kit,
            inputs=[genre_combo_kit],
            outputs=[
                preset_name,
                lyric_style_preset,
                tag_style_preset,
                lyrics,
                tags,
                max_audio_length_s,
                codec_num_steps,
                codec_guidance_scale,
                mac_safe,
                keep_models_loaded,
                lyrics_guidance,
                prompt_guidance,
                editor_status,
            ],
        )
        apply_lyric_pack_button.click(
            fn=apply_lyric_style_preset,
            inputs=[lyric_style_preset, tags, max_audio_length_s, preset_name],
            outputs=[lyrics, lyrics_guidance, prompt_guidance, editor_status],
        )
        apply_tag_pack_button.click(
            fn=apply_tag_style_preset,
            inputs=[tag_style_preset, lyrics, max_audio_length_s, preset_name],
            outputs=[tags, prompt_guidance, editor_status],
        )
        lyrics.change(
            fn=build_lyrics_guidance,
            inputs=[lyrics, max_audio_length_s],
            outputs=[lyrics_guidance],
        )
        lyrics.change(
            fn=build_prompt_guidance,
            inputs=[lyrics, tags, max_audio_length_s, preset_name],
            outputs=[prompt_guidance],
        )
        tags.change(
            fn=build_prompt_guidance,
            inputs=[lyrics, tags, max_audio_length_s, preset_name],
            outputs=[prompt_guidance],
        )
        max_audio_length_s.change(
            fn=build_lyrics_guidance,
            inputs=[lyrics, max_audio_length_s],
            outputs=[lyrics_guidance],
        )
        max_audio_length_s.change(
            fn=build_prompt_guidance,
            inputs=[lyrics, tags, max_audio_length_s, preset_name],
            outputs=[prompt_guidance],
        )
        refresh_checkpoint_button.click(
            fn=refresh_finetune_checkpoint_choices,
            inputs=[finetune_checkpoint],
            outputs=[finetune_checkpoint_choice],
        )
        finetune_checkpoint_choice.change(
            fn=select_finetune_checkpoint,
            inputs=[finetune_checkpoint_choice],
            outputs=[finetune_checkpoint],
        )
        finetune_checkpoint.change(
            fn=refresh_finetune_checkpoint_choices,
            inputs=[finetune_checkpoint],
            outputs=[finetune_checkpoint_choice],
        )
        save_settings_button.click(
            fn=save_studio_preferences,
            inputs=[
                model_path,
                output_name,
                finetune_checkpoint,
                codec_num_steps,
                codec_guidance_scale,
                mac_safe,
                keep_models_loaded,
                transcribe_model_path,
                transcribe_mac_safe,
            ],
            outputs=[
                model_path,
                output_name,
                finetune_checkpoint,
                finetune_checkpoint_choice,
                codec_num_steps,
                codec_guidance_scale,
                mac_safe,
                keep_models_loaded,
                transcribe_model_path,
                transcribe_mac_safe,
                settings_status,
                settings_overview,
            ],
        )
        reload_settings_button.click(
            fn=load_studio_preferences_for_ui,
            outputs=[
                model_path,
                output_name,
                finetune_checkpoint,
                finetune_checkpoint_choice,
                codec_num_steps,
                codec_guidance_scale,
                mac_safe,
                keep_models_loaded,
                transcribe_model_path,
                transcribe_mac_safe,
                settings_status,
                settings_overview,
            ],
        )
        reset_settings_button.click(
            fn=reset_studio_preferences_for_ui,
            outputs=[
                model_path,
                output_name,
                finetune_checkpoint,
                finetune_checkpoint_choice,
                codec_num_steps,
                codec_guidance_scale,
                mac_safe,
                keep_models_loaded,
                transcribe_model_path,
                transcribe_mac_safe,
                settings_status,
                settings_overview,
            ],
        )

        generate_button.click(
            fn=generate_music,
            inputs=[
                lyrics,
                tags,
                model_path,
                finetune_checkpoint,
                output_name,
                max_audio_length_s,
                candidate_count,
                codec_num_steps,
                codec_guidance_scale,
                mac_safe,
                keep_models_loaded,
                preset_name,
            ],
            outputs=[
                generation_status,
                generation_audio,
                generation_file,
                library_choice,
                library_details,
                library_audio,
                library_file,
                library_status,
                library_cards,
                sticky_player_info,
                sticky_player_audio,
                pinned_cards,
                favorite_selected_button,
                training_selected_button,
                archive_selected_button,
                delete_selected_button,
                candidate_compare,
                candidate_choice,
                candidate_paths_state,
                library_search,
                library_filter,
                render_history_html,
            ],
        )
        render_final_button.click(
            fn=render_final_music,
            inputs=[
                lyrics,
                tags,
                model_path,
                finetune_checkpoint,
                output_name,
                max_audio_length_s,
                candidate_count,
            ],
            outputs=[
                generation_status,
                generation_audio,
                generation_file,
                library_choice,
                library_details,
                library_audio,
                library_file,
                library_status,
                library_cards,
                sticky_player_info,
                sticky_player_audio,
                pinned_cards,
                favorite_selected_button,
                training_selected_button,
                archive_selected_button,
                delete_selected_button,
                candidate_compare,
                candidate_choice,
                candidate_paths_state,
                library_search,
                library_filter,
                render_history_html,
            ],
        )
        queue_render_button.click(
            fn=queue_music_render,
            inputs=[
                lyrics,
                tags,
                model_path,
                finetune_checkpoint,
                output_name,
                max_audio_length_s,
                candidate_count,
                codec_num_steps,
                codec_guidance_scale,
                mac_safe,
                keep_models_loaded,
                preset_name,
            ],
            outputs=[generation_status, render_history_html],
        )
        queue_refresh_button.click(
            fn=render_render_history,
            outputs=[render_history_html],
        )
        refresh_harvest_button.click(
            fn=refresh_training_harvest_panel,
            outputs=[training_harvest_status, training_harvest_log],
        )
        harvest_training_button.click(
            fn=harvest_training_candidates_from_app,
            outputs=[training_harvest_status, training_harvest_log],
        )
        prepare_experiment_button.click(
            fn=prepare_app_training_experiment,
            outputs=[training_harvest_status, training_harvest_log],
        )
        train_smoke_button.click(
            fn=train_app_experiment_smoke,
            outputs=[training_harvest_status, training_harvest_log],
        )
        train_cycle_button.click(
            fn=train_app_experiment_cycle1,
            outputs=[training_harvest_status, training_harvest_log],
        )

        candidate_choice.change(
            fn=select_candidate,
            inputs=[candidate_choice, candidate_paths_state, library_search, library_filter],
            outputs=[
                generation_audio,
                generation_file,
                library_choice,
                library_details,
                library_audio,
                library_file,
                library_status,
                library_cards,
                sticky_player_info,
                sticky_player_audio,
                pinned_cards,
                favorite_selected_button,
                training_selected_button,
                archive_selected_button,
                delete_selected_button,
                candidate_compare,
            ],
        )
        previous_take_button.click(
            fn=previous_candidate,
            inputs=[candidate_choice, candidate_paths_state, library_search, library_filter],
            outputs=[
                generation_audio,
                generation_file,
                library_choice,
                library_details,
                library_audio,
                library_file,
                library_status,
                library_cards,
                sticky_player_info,
                sticky_player_audio,
                pinned_cards,
                favorite_selected_button,
                training_selected_button,
                archive_selected_button,
                delete_selected_button,
                candidate_compare,
                candidate_choice,
            ],
        )
        next_take_button.click(
            fn=next_candidate,
            inputs=[candidate_choice, candidate_paths_state, library_search, library_filter],
            outputs=[
                generation_audio,
                generation_file,
                library_choice,
                library_details,
                library_audio,
                library_file,
                library_status,
                library_cards,
                sticky_player_info,
                sticky_player_audio,
                pinned_cards,
                favorite_selected_button,
                training_selected_button,
                archive_selected_button,
                delete_selected_button,
                candidate_compare,
                candidate_choice,
            ],
        )
        favorite_take_button.click(
            fn=favorite_candidate,
            inputs=[candidate_choice, candidate_paths_state, library_search, library_filter],
            outputs=[
                generation_audio,
                generation_file,
                library_choice,
                library_details,
                library_audio,
                library_file,
                library_status,
                library_cards,
                sticky_player_info,
                sticky_player_audio,
                pinned_cards,
                favorite_selected_button,
                training_selected_button,
                archive_selected_button,
                delete_selected_button,
                candidate_compare,
                candidate_choice,
            ],
        )

        save_draft_button.click(
            fn=save_generate_drafts,
            inputs=[lyrics, tags],
            outputs=[editor_status],
        )
        optimize_lyrics_button.click(
            fn=optimize_lyrics_for_length,
            inputs=[lyrics, max_audio_length_s, tags, preset_name],
            outputs=[lyrics, editor_status, lyrics_guidance, prompt_guidance],
        )
        reload_draft_button.click(
            fn=reload_generate_drafts_with_guidance,
            inputs=[max_audio_length_s, preset_name],
            outputs=[lyrics, tags, editor_status, lyrics_guidance, prompt_guidance],
        )
        reset_defaults_button.click(
            fn=reset_generate_defaults_with_guidance,
            inputs=[max_audio_length_s, preset_name],
            outputs=[lyrics, tags, editor_status, lyrics_guidance, prompt_guidance],
        )

        library_refresh_button.click(
            fn=load_dashboard,
            inputs=[library_choice, library_search, library_filter],
            outputs=[
                library_choice,
                library_details,
                library_audio,
                library_file,
                library_status,
                library_cards,
                sticky_player_info,
                sticky_player_audio,
                pinned_cards,
                favorite_selected_button,
                training_selected_button,
                archive_selected_button,
                delete_selected_button,
                render_history_html,
            ],
        )
        library_choice.change(
            fn=library_snapshot,
            inputs=[library_choice, library_search, library_filter],
            outputs=[
                library_choice,
                library_details,
                library_audio,
                library_file,
                library_status,
                library_cards,
                sticky_player_info,
                sticky_player_audio,
                pinned_cards,
                favorite_selected_button,
                training_selected_button,
                archive_selected_button,
                delete_selected_button,
            ],
        )
        library_search.change(
            fn=library_snapshot,
            inputs=[library_choice, library_search, library_filter],
            outputs=[
                library_choice,
                library_details,
                library_audio,
                library_file,
                library_status,
                library_cards,
                sticky_player_info,
                sticky_player_audio,
                pinned_cards,
                favorite_selected_button,
                training_selected_button,
                archive_selected_button,
                delete_selected_button,
            ],
        )
        library_filter.change(
            fn=library_snapshot,
            inputs=[library_choice, library_search, library_filter],
            outputs=[
                library_choice,
                library_details,
                library_audio,
                library_file,
                library_status,
                library_cards,
                sticky_player_info,
                sticky_player_audio,
                pinned_cards,
                favorite_selected_button,
                training_selected_button,
                archive_selected_button,
                delete_selected_button,
            ],
        )
        favorite_selected_button.click(
            fn=toggle_library_favorite,
            inputs=[library_choice, library_search, library_filter],
            outputs=[
                library_choice,
                library_details,
                library_audio,
                library_file,
                library_status,
                library_cards,
                sticky_player_info,
                sticky_player_audio,
                pinned_cards,
                favorite_selected_button,
                training_selected_button,
                archive_selected_button,
                delete_selected_button,
            ],
        )
        training_selected_button.click(
            fn=toggle_library_training_candidate,
            inputs=[library_choice, library_search, library_filter],
            outputs=[
                library_choice,
                library_details,
                library_audio,
                library_file,
                library_status,
                library_cards,
                sticky_player_info,
                sticky_player_audio,
                pinned_cards,
                favorite_selected_button,
                training_selected_button,
                archive_selected_button,
                delete_selected_button,
                training_harvest_status,
                training_harvest_log,
            ],
        )
        archive_selected_button.click(
            fn=toggle_library_archive,
            inputs=[library_choice, library_search, library_filter],
            outputs=[
                library_choice,
                library_details,
                library_audio,
                library_file,
                library_status,
                library_cards,
                sticky_player_info,
                sticky_player_audio,
                pinned_cards,
                favorite_selected_button,
                training_selected_button,
                archive_selected_button,
                delete_selected_button,
            ],
        )
        delete_selected_button.click(
            fn=delete_library_track,
            inputs=[library_choice, library_search, library_filter],
            outputs=[
                library_choice,
                library_details,
                library_audio,
                library_file,
                library_status,
                library_cards,
                sticky_player_info,
                sticky_player_audio,
                pinned_cards,
                favorite_selected_button,
                training_selected_button,
                archive_selected_button,
                delete_selected_button,
            ],
        )

        nav_create_button.click(
            fn=lambda: switch_workspace("create"),
            outputs=[
                workspace_state,
                workspace_intro,
                nav_create_button,
                nav_library_button,
                nav_training_button,
                nav_transcribe_button,
                nav_settings_button,
                create_workspace,
                library_workspace,
                training_workspace,
                transcribe_workspace,
                settings_workspace,
            ],
        )
        nav_library_button.click(
            fn=lambda: switch_workspace("library"),
            outputs=[
                workspace_state,
                workspace_intro,
                nav_create_button,
                nav_library_button,
                nav_training_button,
                nav_transcribe_button,
                nav_settings_button,
                create_workspace,
                library_workspace,
                training_workspace,
                transcribe_workspace,
                settings_workspace,
            ],
        )
        nav_training_button.click(
            fn=lambda: switch_workspace("training"),
            outputs=[
                workspace_state,
                workspace_intro,
                nav_create_button,
                nav_library_button,
                nav_training_button,
                nav_transcribe_button,
                nav_settings_button,
                create_workspace,
                library_workspace,
                training_workspace,
                transcribe_workspace,
                settings_workspace,
            ],
        )
        nav_transcribe_button.click(
            fn=lambda: switch_workspace("transcribe"),
            outputs=[
                workspace_state,
                workspace_intro,
                nav_create_button,
                nav_library_button,
                nav_training_button,
                nav_transcribe_button,
                nav_settings_button,
                create_workspace,
                library_workspace,
                training_workspace,
                transcribe_workspace,
                settings_workspace,
            ],
        )
        nav_settings_button.click(
            fn=lambda: switch_workspace("settings"),
            outputs=[
                workspace_state,
                workspace_intro,
                nav_create_button,
                nav_library_button,
                nav_training_button,
                nav_transcribe_button,
                nav_settings_button,
                create_workspace,
                library_workspace,
                training_workspace,
                transcribe_workspace,
                settings_workspace,
            ],
        )

        demo.load(
            fn=load_dashboard,
            outputs=[
                library_choice,
                library_details,
                library_audio,
                library_file,
                library_status,
                library_cards,
                sticky_player_info,
                sticky_player_audio,
                pinned_cards,
                favorite_selected_button,
                training_selected_button,
                archive_selected_button,
                delete_selected_button,
                render_history_html,
            ],
        )
        demo.load(
            fn=render_training_harvest_status,
            outputs=[training_harvest_status],
        )
        demo.load(
            fn=lambda: switch_workspace("create"),
            outputs=[
                workspace_state,
                workspace_intro,
                nav_create_button,
                nav_library_button,
                nav_training_button,
                nav_transcribe_button,
                nav_settings_button,
                create_workspace,
                library_workspace,
                training_workspace,
                transcribe_workspace,
                settings_workspace,
            ],
        )
        demo.load(
            fn=load_studio_preferences_for_ui,
            outputs=[
                model_path,
                output_name,
                finetune_checkpoint,
                finetune_checkpoint_choice,
                codec_num_steps,
                codec_guidance_scale,
                mac_safe,
                keep_models_loaded,
                transcribe_model_path,
                transcribe_mac_safe,
                settings_status,
                settings_overview,
            ],
        )
        transcribe_button.click(
            fn=transcribe_music,
            inputs=[
                uploaded_audio,
                music_path,
                transcribe_model_path,
                transcribe_mac_safe,
            ],
            outputs=[transcription_status, transcription_text],
        )

    return demo


def parse_args():
    parser = argparse.ArgumentParser(description="Launch mark's Studio.")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--inbrowser", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ensure_render_queue_worker()
    app = build_app()
    app.queue(default_concurrency_limit=1).launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        inbrowser=args.inbrowser,
    )
