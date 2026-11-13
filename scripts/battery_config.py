"""Shared processing conventions; provisional limits are not physical impossibilities."""
import hashlib
import json
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent / 'battery_thresholds.json'
CONFIG = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
CONFIG_SHA256 = hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()
