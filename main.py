import os
import json
import queue
import threading
import time

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from dotenv import load_dotenv

from core_intelligence import (
    life,
    action,
    prime_directive,
)

load_dotenv() 

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8000",
        "https://void.dilloncarey.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

life_output_queue = queue.Queue()

life_thread = None

def run_life_in_background():
    global life_thread
    if life_thread is None or not life_thread.is_alive():
        print("Starting life function in background thread...")
        life_thread = threading.Thread(target=life, kwargs={'output_queue': life_output_queue, 'prime_dir_val': prime_directive})
        life_thread.daemon = True
        life_thread.start()
        print("Life thread started.")
    else:
        print("Life function is already running.")

@app.post("/start_life")
async def start_life_process(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_life_in_background)
    return {"message": "Life process started in the background."}

@app.get("/life_stream")
async def life_stream():
    async def event_generator():
        while True:
            try:
                item = life_output_queue.get(timeout=30)
                yield f"data: {json.dumps(item)}\n\n"
                if item.get("type") == "death":
                    print("Death message sent, closing SSE stream.")
                    break
            except queue.Empty:
                yield ":keep-alive\n\n"
            except Exception as e:
                print(f"Error in SSE event_generator: {e}")
                yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
                break

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/interact")
async def interact_with_ai(prompt: dict):
    user_input = prompt.get("text")
    if not user_input:
        raise HTTPException(status_code=400, detail="Prompt text is required.")
    
    response = action(user_input)
    return {"response": response}

@app.on_event("startup")
async def startup_event():
    print("FastAPI application startup completed.")

@app.on_event("shutdown")
async def shutdown_event():
    print("FastAPI application shutdown initiated.")
    if life_thread and life_thread.is_alive():
        print("Waiting for life thread to finish (if not daemon)...")
        print("Life thread should terminate soon.")