from typing import List, Dict
import uuid
import json
import os
import datetime

MEMORY_FILE = "memory.json"

memory: List[Dict] = []

def load_memory():
    global memory
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            try:
                memory = json.load(f)
            except json.JSONDecodeError:
                memory = []
    else:
        memory = []

def save_memory():
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2)

def add_memory(text: str, category: str = "general", tags: List[str] = None):
    node = {
        "id": str(uuid.uuid4()),
        'datetime': str(datetime.datetime.now()),
        "text": text,
        "category": category,
        "tags": tags or []
    }
    memory.append(node)
    save_memory()

def get_memory(category: str = None, tag: str = None) -> List[Dict]:
    results = memory
    if category:
        results = [m for m in results if m.get("category") == category]
    if tag:
        results = [m for m in results if tag in m.get("tags", [])]
    return results

def clear_memory():
    global memory
    memory.clear()
    save_memory()
    
def update_memory(node_id: str, text: str = None, category: str = None, tags: List[str] = None) -> bool:
    for node in memory:
        if node["id"] == node_id:
            if text is not None:
                node["text"] = text
            if category is not None:
                node["category"] = category
            if tags is not None:
                node["tags"] = tags
            save_memory()
            return True
    return False

load_memory()

save_memory()


