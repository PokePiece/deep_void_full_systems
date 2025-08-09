
#!/usr/bin/env python3
"""
service_main.py
Daemon wrapper for the Void intelligence runtime.
- Runs life() in a background thread (keeps your existing code intact)
- Bridges results to an asyncio queue
- Exposes a small control HTTP API for start/stop/status on localhost
- Gracefully handles SIGTERM for systemd
"""

import asyncio
import signal
import threading
import queue
import time
import logging
from typing import Optional
from fastapi import FastAPI
import uvicorn

# Import your existing functions
# Adjust import paths if necessary
# from your_package.life_module import life, action, intelligence  <-- example
# In your posted code, life is defined in the same file; if so import accordingly
from pathlib import Path
import os
import sys
from core_intelligence import life, action, intelligence

# If your code is in another file, adapt the import. For inline quick test:
# from void_runtime import life, action, ... or if in same file, import nothing.

# ---- CONFIG ----
HOST = "127.0.0.1"
PORT = 8700  # local control port
LOG_FILE = "/var/log/void_runtime.log" if os.geteuid() == 0 else "./void_runtime.log"
SHUTDOWN_TIMEOUT = 30  # seconds to wait for clean shutdown

# ---- logging ----
logger = logging.getLogger("void_runtime")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

fh = logging.FileHandler(LOG_FILE)
fh.setFormatter(formatter)
logger.addHandler(fh)

ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(formatter)
logger.addHandler(ch)

# ---- Async bridge ----
async_result_queue: asyncio.Queue = asyncio.Queue()
_thread_result_queue: queue.Queue = queue.Queue()

# We'll import the life() function at runtime to avoid circular imports
# If life() is defined in this same file, just use it; else import properly.
try:
    # if life is in the same module
    from __main__ import life  # only works when running as script
except Exception:
    try:
        # try common module name 'void_life'
        from core_intelligence import life  # replace with your actual module name
    except Exception:
        # fallback: assume life exists in global namespace (for dev)
        life = globals().get("life")
        if life is None:
            logger.error("Could not import life(). Place service_main in same package or adjust import.")
            raise SystemExit(1)

# ---- Control flags ----
_shutdown_event = threading.Event()
_life_thread: Optional[threading.Thread] = None
_life_running = threading.Event()

def _life_thread_target(output_q: queue.Queue):
    """
    Run the existing life() function in a thread and push events to output_q.
    This keeps user code unchanged and bridges its queue->asyncio queue.
    """
    global _life_running
    try:
        logger.info("life() thread starting")
        _life_running.set()
        # The life() in your code expects output_queue parameter.
        life(prime_dir=None, output_queue=output_q)
    except Exception as e:
        logger.exception("Exception in life(): %s", e)
    finally:
        logger.info("life() thread exiting")
        _life_running.clear()

async def _bridge_thread_queue_to_asyncio():
    """
    Move items from the thread-safe queue into asyncio queue for HTTP endpoints.
    """
    while not _shutdown_event.is_set():
        try:
            item = _thread_result_queue.get(timeout=0.5)
            await async_result_queue.put(item)
        except queue.Empty:
            await asyncio.sleep(0.1)
        except Exception as e:
            logger.exception("Bridge error: %s", e)

# ---- FastAPI control API ----
app = FastAPI(title="Void Runtime Control API")

@app.on_event("startup")
async def startup_event():
    logger.info("Control API startup: launching bridge and starting life")
    # start bridge task
    app.state.bridge_task = asyncio.create_task(_bridge_thread_queue_to_asyncio())
    # start life thread
    start_life()

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Control API shutdown: stopping life")
    stop_life()
    # cancel bridge
    if hasattr(app.state, "bridge_task"):
        app.state.bridge_task.cancel()
        try:
            await app.state.bridge_task
        except asyncio.CancelledError:
            pass

@app.get("/health")
async def health():
    return {"status": "ok", "life_running": _life_running.is_set()}

@app.get("/status")
async def status():
    # return last n items from queue non-destructively would require caching; here we pop a single item if available
    items = []
    try:
        while not async_result_queue.empty():
            items.append(await async_result_queue.get())
            # don't await heavy processing here
    except Exception as e:
        logger.exception("Error reading status queue: %s", e)
    return {"life_running": _life_running.is_set(), "recent_events_count": len(items), "recent": items[:10]}

@app.post("/start")
async def api_start():
    if _life_running.is_set():
        return {"result": "already_running"}
    start_life()
    return {"result": "started"}

@app.post("/stop")
async def api_stop():
    if not _life_running.is_set():
        return {"result": "not_running"}
    stop_life()
    return {"result": "stopped"}

def start_life():
    global _life_thread
    if _life_thread and _life_thread.is_alive():
        logger.info("life thread already running")
        return
    _life_thread = threading.Thread(target=_life_thread_target, args=(_thread_result_queue,), daemon=True)
    _life_thread.start()
    logger.info("life thread launched")

def stop_life():
    logger.info("stop_life requested: setting shutdown flag")
    _shutdown_event.set()
    # if life() polls output_queue and checks for termination, push a sentinel
    try:
        _thread_result_queue.put({"type": "control", "message": "shutdown"})
    except Exception:
        pass
    # wait for thread to stop
    if _life_thread:
        _life_thread.join(timeout=SHUTDOWN_TIMEOUT)
        if _life_thread.is_alive():
            logger.warning("life thread did not exit in time")

def _handle_sigterm():
    logger.info("SIGTERM received, initiating shutdown")
    _shutdown_event.set()

def serve():
    # register signals for graceful shutdown
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _handle_sigterm)

    config = uvicorn.Config(app, host=HOST, port=PORT, log_level="info")
    server = uvicorn.Server(config)

    # Run server until shutdown flag
    async def server_main():
        # Run uvicorn until shutdown event
        await server.serve()

    try:
        loop.run_until_complete(server_main())
    except Exception as e:
        logger.exception("Server exception: %s", e)
    finally:
        logger.info("Server shutting down")
        stop_life()

if __name__ == "__main__":
    logger.info("Starting void_runtime service_main")
    serve()
