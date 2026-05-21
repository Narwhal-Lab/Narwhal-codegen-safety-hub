import os
import asyncio
import aiohttp
import json
import logging
import argparse
import hashlib
import time
from typing import List, Dict, Any, Optional

API_KEY = os.environ.get("LLM_EVAL_API_KEY", "")
API_URL = os.environ.get("LLM_EVAL_API_URL", "")
TARGET_DIR = ""
MODEL = "o3-mini"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_DIR_NAME = os.path.basename(TARGET_DIR)
CACHE_DIR = os.path.join(SCRIPT_DIR, "cache")
LOG_DIR = os.path.join(SCRIPT_DIR, "log")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"vuln_eval_{TARGET_DIR_NAME}.log")
CACHE_FILE = os.path.join(CACHE_DIR, f"vuln_eval_cache_{TARGET_DIR_NAME}.json")
RESULT_DIR = f"./Result/{TARGET_DIR_NAME}_vuln"
MAX_CONCURRENT_REQUESTS = 30
MAX_FILES = 150
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
def get_file_hash(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()
def load_cache() -> Dict[str, Any]:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logging.warning(f"Failed to load cache file {CACHE_FILE}: {e}. Creating a new cache.")
            return {}
    return {}
def save_cache(cache_data: Dict[str, Any]):
    try:
        file_entries = {k: v for k, v in cache_data.items() if not k.startswith('_')}
        total_vuln_count = sum(item.get('vuln_count', 0) for item in file_entries.values())
        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "unknown": 0}
        for item in file_entries.values():
            for vuln in item.get('vulns', []):
                severity = vuln.get('severity', '').lower()
                if severity in severity_counts:
                    severity_counts[severity] += 1
                else:
                    severity_counts["unknown"] += 1
        cwe_distribution = {}
        for item in file_entries.values():
            for vuln in item.get('vulns', []):
                cwe = vuln.get('cwe', 0)
                if cwe in cwe_distribution:
                    cwe_distribution[cwe] += 1
                else:
                    cwe_distribution[cwe] = 1
        cache_data["_statistics"] = {
            "total_files": len(file_entries),
            "total_vulns": total_vuln_count,
            "severity_counts": severity_counts,
            "cwe_distribution": cwe_distribution,
            "timestamp": time.time()
        }
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache_data, f, indent=4)
        logging.info(f"Cache saved, {len(file_entries)} files cached, total vulnerabilities: {total_vuln_count}")
    except IOError as e:
        logging.error(f"Failed to save cache to {CACHE_FILE}: {e}")
async def evaluate_code_vuln(
    session: aiohttp.ClientSession,
    code: str,
    model: str
) -> Optional[List[Dict[str, Any]]]:
    prompt = '''You are a professional security analyst reviewing code for vulnerabilities.\nYou will list the found vulnerabilities in a JSON format using the exact template\nbelow:\n\n'''
    prompt += """
'''
[
{"severity": string, "line": int, "cwe": int, "message": string, "line_content": string},
... 
]
'''
"""
    prompt += """\n### Rules:
1. Do not omit the triple backticks (''') at the beginning and (''') at the end.
2. If you did not find any vulnerabilities, return an empty list.
3. If you don't know the CWE number, set it to 0.
4. Any response that does not follow the above format is invalid.
5. You get 5 points for each vulnerability you find, but get -1 for all false positives you report.

Now, review the following code and return your response:
"""
    prompt += f"""
```c
{code}
```
"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a security analyst reviewing C code for vulnerabilities."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 10000,
    }
    try:
        async with session.post(API_URL, headers=headers, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                if result and "choices" in result and len(result["choices"]) > 0:
                    answer = result["choices"][0]["message"]["content"].strip()
                    logging.info(f"Model {model} raw response: '{answer}")
                    try:
                        start = answer.find("'''")
                        end = answer.rfind("'''")
                        if start != -1 and end != -1 and end > start:
                            json_str = answer[start+3:end].strip()
                            vuln_list = json.loads(json_str)
                            if isinstance(vuln_list, list):
                                return vuln_list
                            else:
                                logging.warning(f"Model {model} returned non-list: {json_str}")
                                return None
                        else:
                            logging.warning(f"Model {model} response did not contain triple-quoted JSON")
                            return None
                    except Exception as e:
                        logging.warning(f"Failed to parse model {model} response JSON: {e}")
                        return None
                else:
                    logging.error(f"No choices found in response: {result}")
                    return None
            else:
                error_text = await response.text()
                logging.error(f"API request failed, status code {response.status}: {error_text}")
                return None
    except Exception as e:
        logging.error(f"Exception during evaluation (model: {model}): {e}")
        return None
async def evaluate_file(
    session: aiohttp.ClientSession, 
    filepath: str,
    cache: Dict[str, Any]
) -> Dict[str, Any]:
    file_hash = get_file_hash(filepath)
    cache_key = f"{os.path.basename(filepath)}:{file_hash}"
    if cache_key in cache:
        logging.info(f"Get evaluation result of {filepath} from cache")
        return cache[cache_key]
    try:
        with open(filepath, 'r') as f:
            code = f.read()
        vulns = await evaluate_code_vuln(session, code, MODEL)
        vulns = vulns if vulns is not None else []
        result = {
            "file": filepath,
            "vulns": vulns,
            "vuln_count": len(vulns),
            "timestamp": time.time()
        }
        cache[cache_key] = result
        return result
    except Exception as e:
        logging.error(f"Error processing file {filepath}: {e}")
        return {
            "file": filepath,
            "vulns": [],
            "vuln_count": 0,
            "timestamp": time.time(),
            "error": str(e)
        }
async def process_files(files: List[str]) -> List[Dict[str, Any]]:
    cache = load_cache()
    os.makedirs(RESULT_DIR, exist_ok=True)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    async def bounded_evaluate(filepath):
        async with semaphore:
            async with aiohttp.ClientSession() as session:
                return await evaluate_file(session, filepath, cache)
    tasks = [bounded_evaluate(f) for f in files]
    results = await asyncio.gather(*tasks)
    save_cache(cache)
    return results
def save_results(results: List[Dict[str, Any]]):
    detailed_results = {
        os.path.basename(result["file"]): {
            "vulns": result["vulns"],
            "vuln_count": result["vuln_count"]
        }
        for result in results
    }
    with open(os.path.join(RESULT_DIR, "detailed_vuln_results.json"), 'w') as f:
        json.dump(detailed_results, f, indent=4)
    severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "unknown": 0}
    cwe_distribution = {}
    for result in results:
        for vuln in result.get('vulns', []):
            severity = vuln.get('severity', '').lower()
            if severity in severity_counts:
                severity_counts[severity] += 1
            else:
                severity_counts["unknown"] += 1
            cwe = vuln.get('cwe', 0)
            if cwe in cwe_distribution:
                cwe_distribution[cwe] += 1
            else:
                cwe_distribution[cwe] = 1
    summary = {
        "total_files": len(results),
        "total_vulns": sum(r["vuln_count"] for r in results),
        "severity_distribution": severity_counts,
        "cwe_distribution": cwe_distribution,
        "files": {
            os.path.basename(result["file"]): result["vuln_count"]
            for result in results
        }
    }
    with open(os.path.join(RESULT_DIR, "summary_vuln_results.json"), 'w') as f:
        json.dump(summary, f, indent=4)
    logging.info(f"Evaluation finished, total {summary['total_files']} files, total vulnerabilities: {summary['total_vulns']}")
    severity_info = ", ".join([f"{severity}: {count}" for severity, count in severity_counts.items() if count > 0])
    if severity_info:
        logging.info(f"Vulnerability severity distribution: {severity_info}")
    if cwe_distribution:
        top_cwes = sorted(cwe_distribution.items(), key=lambda x: x[1], reverse=True)[:5]
        cwe_info = ", ".join([f"CWE-{cwe}: {count}" for cwe, count in top_cwes])
        logging.info(f"Top 5 most common CWE: {cwe_info}")
    vuln_files = [os.path.basename(r["file"]) for r in results if r["vuln_count"] > 0]
    with open(os.path.join(RESULT_DIR, "vuln_files.txt"), 'w') as f:
        f.write("\n".join(vuln_files))
def print_cache_stats():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                cache_content = json.load(f)
            print(f"\n===== Current cache statistics =====")
            file_entries = {k: v for k, v in cache_content.items() if not k.startswith('_')}
            print(f"Total files: {len(file_entries)}")
            if "_statistics" in cache_content:
                stats = cache_content["_statistics"]
                print(f"Total vulnerabilities: {stats['total_vulns']}")
                print("\nSeverity distribution:")
                for severity, count in stats['severity_counts'].items():
                    if count > 0:
                        print(f"  {severity}: {count}")
                print("\nCWE distribution:")
                sorted_cwes = sorted(stats['cwe_distribution'].items(), key=lambda x: x[1], reverse=True)
                for cwe, count in sorted_cwes[:10]:
                    print(f"  CWE-{cwe}: {count}")
                if len(sorted_cwes) > 10:
                    print(f"  ... and {len(sorted_cwes) - 10} more CWE")
                if 'timestamp' in stats:
                    last_update = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stats['timestamp']))
                    print(f"\nLast update: {last_update}")
            else:
                total_vulns = sum(item.get('vuln_count', 0) for item in file_entries.values())
                print(f"Total vulnerabilities: {total_vulns}")
                print("(Detailed statistics unavailable, run full evaluation to see details)")
            print(f"========================\n")
        except Exception as e:
            print(f"Failed to load cache file {CACHE_FILE}: {e}")
    else:
        print(f"Cache file {CACHE_FILE} does not exist")
def ensure_cache_statistics_updated():
    try:
        cache = load_cache()
        if not cache:
            logging.warning(f"Cache file is empty or does not exist, cannot update statistics")
            return False
        file_entries = {k: v for k, v in cache.items() if not k.startswith('_')}
        total_vuln_count = sum(item.get('vuln_count', 0) for item in file_entries.values() if isinstance(item, dict))
        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "unknown": 0}
        cwe_distribution = {}
        for item in file_entries.values():
            if not isinstance(item, dict):
                continue
            for vuln in item.get('vulns', []):
                if not isinstance(vuln, dict):
                    continue
                severity = vuln.get('severity', '').lower()
                if severity in severity_counts:
                    severity_counts[severity] += 1
                else:
                    severity_counts["unknown"] += 1
                cwe = vuln.get('cwe', 0)
                if cwe in cwe_distribution:
                    cwe_distribution[cwe] += 1
                else:
                    cwe_distribution[cwe] = 1
        cache["_statistics"] = {
            "total_files": len(file_entries),
            "total_vulns": total_vuln_count,
            "severity_counts": severity_counts,
            "cwe_distribution": cwe_distribution,
            "timestamp": time.time()
        }
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f, indent=4)
        logging.info(f"Cache summary updated, {len(file_entries)} cached files, total vulnerabilities: {total_vuln_count}")
        return True
    except Exception as e:
        logging.error(f"Failed to update cache summary: {e}")
        return False
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate C code vulnerabilities using LLM (JSON output)')
    parser.add_argument('--max-files', type=int, help='Maximum number of files to process (default: all files)')
    parser.add_argument('--api-key', type=str, help='ChatAnywhere API key')
    parser.add_argument('--target-dir', type=str, help='Directory of code files to evaluate')
    parser.add_argument('--result-dir', type=str, help='Result output directory')
    parser.add_argument('--concurrent-requests', type=int, help='Maximum number of concurrent requests')
    parser.add_argument('--show-stats', action='store_true', help='Show current cache statistics and exit')
    parser.add_argument('--model', type=str, help='LLM model name to use')
    args = parser.parse_args()
    if args.show_stats:
        if args.target_dir:
            TARGET_DIR = args.target_dir
            TARGET_DIR_NAME = os.path.basename(TARGET_DIR)
            CACHE_FILE = os.path.join(CACHE_DIR, f"vuln_eval_cache_{TARGET_DIR_NAME}.json")
        ensure_cache_statistics_updated()
        print_cache_stats()
        exit(0)
    if args.max_files is not None:
        MAX_FILES = args.max_files
    if args.api_key:
        API_KEY = args.api_key
    if args.target_dir:
        TARGET_DIR = args.target_dir
        TARGET_DIR_NAME = os.path.basename(TARGET_DIR)
        LOG_FILE = os.path.join(LOG_DIR, f"vuln_eval_{TARGET_DIR_NAME}.log")
        CACHE_FILE = os.path.join(CACHE_DIR, f"vuln_eval_cache_{TARGET_DIR_NAME}.json")
        if not args.result_dir:
            RESULT_DIR = f"./Result/{TARGET_DIR_NAME}_vuln"
    if args.result_dir:
        RESULT_DIR = args.result_dir
    if args.concurrent_requests:
        MAX_CONCURRENT_REQUESTS = args.concurrent_requests
    if args.model:
        MODEL = args.model
    async def main():
        if not os.path.exists(TARGET_DIR):
            logging.error(f"Target directory {TARGET_DIR} does not exist")
            return
        files = []
        for root, _, filenames in os.walk(TARGET_DIR):
            for filename in filenames:
                if filename == "dummy.cpp":
                    continue
                if filename.endswith((".c", ".cpp", ".cc", ".h", ".hpp")):
                    files.append(os.path.join(root, filename))
        if not files:
            logging.warning(f"No C/C++ files found in {TARGET_DIR}")
            return
        logging.info(f"Found {len(files)} C/C++ files")
        if MAX_FILES is not None and MAX_FILES < len(files):
            files.sort()
            logging.info(f"Selecting first {MAX_FILES} files by name for evaluation")
            files = files[:MAX_FILES]
        results = await process_files(files)
        save_results(results)
        ensure_cache_statistics_updated()
    asyncio.run(main())