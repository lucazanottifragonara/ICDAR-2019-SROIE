import ollama
import os
import json
import time

IMG_DIR = "./data/img"
KEY_DIR = "./data/key"
MODEL = "llama3.2-vision" # Try "qwen2-vl" for a 3x speed boost
LOG_FILE = "ocr_debug_log.txt"

def run_debug_benchmark(limit=1):
    client = ollama.Client(host='http://localhost:11434')
    images = [f for f in os.listdir(IMG_DIR) if f.lower().endswith('.jpg')][:limit]
    
    with open(LOG_FILE, "w", encoding="utf-8") as log:
        log.write(f"--- Benchmark Start: {time.ctime()} ---\n\n")

        for img_name in images:
            img_path = os.path.join(IMG_DIR, img_name)
            print(f"Testing {img_name}...", end=" ", flush=True)
            
            start = time.time()
            response = client.chat(
                model=MODEL,
                messages=[{'role': 'user', 'content': 'Extract Vendor, Date, and Total. Return JSON.', 'images': [img_path]}],
                options={"num_thread": 12} 
            )
            elapsed = time.time() - start
            raw_output = response['message']['content']

            # --- DEBUG LOGGING ---
            log.write(f"FILE: {img_name} ({elapsed:.1f}s)\n")
            log.write(f"RAW OUTPUT:\n{raw_output}\n")
            log.write("-" * 30 + "\n")
            # ---------------------

            # Attempt to parse
            try:
                clean = raw_output.strip().replace("```json", "").replace("```", "")
                data = json.loads(clean)
                print(f"✅ (Total: {data.get('total')})")
            except:
                print(f"❌ (Parsing Failed - Check {LOG_FILE})")

if __name__ == "__main__":
    run_debug_benchmark(limit=5)