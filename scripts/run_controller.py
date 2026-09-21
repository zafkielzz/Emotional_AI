#!/usr/bin/env python3
"""Start the controller on HTTP/WebSocket port 8000 and legacy TCP relay 9100."""

from pathlib import Path
import sys

import uvicorn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

uvicorn.run("controller.phonefarm_controller.main:app", host="0.0.0.0", port=8000, reload=False)
