import ollama
import os
import json
import time
import re
from decimal import Decimal, InvalidOperation

IMG_DIR = "./data/img"
KEY_DIR = "./data/key"
MODEL = "llama3.2-vision" # Try "qwen2-vl" for a 3x speed boost
LOG_FILE = "ocr_debug_log.txt"

def run_debug_benchmark(limit=2):
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
            except Exception:
                log.write(f"FILE: {img_name} (PARSE FAIL)\n")
                log.write(f"RAW OUTPUT:\n{raw_output}\n")
                log.write("-" * 30 + "\n")
                print(f"❌ (Parsing Failed - Check {LOG_FILE})")
                continue

            # Load ground truth for this image (prefer .json)
            base = os.path.splitext(img_name)[0]
            gt_path_json = os.path.join(KEY_DIR, f"{base}.json")
            gt = None
            if os.path.exists(gt_path_json):
                try:
                    with open(gt_path_json, 'r', encoding='utf-8') as gf:
                        gt = json.load(gf)
                except Exception as e:
                    log.write(f"Failed to read ground truth {gt_path_json}: {e}\n")
            else:
                log.write(f"Ground truth not found for {base} (expected {gt_path_json})\n")

            # Normalize monetary amounts to Decimal
            def normalize_amount(s):
                if s is None:
                    return None
                s = str(s).strip()
                if not s:
                    return None
                negative = False
                if s.startswith('(') and s.endswith(')'):
                    negative = True
                    s = s[1:-1].strip()
                m = re.search(r"[-+]?\d[\d,]*\.?\d*", s)
                if not m:
                    return None
                num = m.group(0).replace(',', '')
                try:
                    val = Decimal(num)
                except InvalidOperation:
                    return None
                return -val if negative else val

            ai_total_raw = data.get('total')
            gt_total_raw = None if gt is None else gt.get('total')

            ai_val = normalize_amount(ai_total_raw)
            gt_val = normalize_amount(gt_total_raw)

            if ai_val is not None and gt_val is not None:
                match = abs(ai_val - gt_val) <= Decimal('0.01')
                print(f"{'✅ MATCH' if match else '❌ FAIL'} (AI: {ai_val:.2f} GT: {gt_val:.2f})")
                log.write(f"FILE: {img_name} ({'MATCH' if match else 'FAIL'})\n")
                log.write(f"AI: {ai_val}  GT: {gt_val}\n")
            else:
                # Fallback to string compare
                ai_s = str(ai_total_raw).replace('$', '').strip()
                gt_s = str(gt_total_raw).replace('$', '').strip()
                match = (ai_s == gt_s)
                print(f"{'✅ MATCH' if match else '❌ FAIL'} (AI: {ai_s} GT: {gt_s})")
                log.write(f"FILE: {img_name} ({'MATCH' if match else 'FAIL'})\n")
                log.write(f"AI RAW: {ai_total_raw}  GT RAW: {gt_total_raw}\n")
            log.write("-" * 30 + "\n")

if __name__ == "__main__":
    run_debug_benchmark(limit=2)