import argparse
import json
import os
from typing import Any, Dict, List, Tuple, DefaultDict
from collections import defaultdict


def _is_setup_event(evt: Dict[str, Any]) -> bool:
    name = evt.get("name", "")
    if not isinstance(name, str):
        return False
    return "setup" in name


def _collect_setup_intervals(events: List[Dict[str, Any]]) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
    """
    Build merged time intervals for setup events per (pid, tid).
    Handles 'X' complete events and 'B'/'E' begin/end pairs.
    Returns dict: {(pid, tid): [(start_ts, end_ts), ... merged]}
    """
    intervals_by_thread: DefaultDict[Tuple[int, int], List[Tuple[int, int]]] = defaultdict(list)

    # First pass: collect 'X' complete events
    for e in events:
        if not _is_setup_event(e):
            continue
        ph = e.get("ph")
        pid = e.get("pid")
        tid = e.get("tid")
        if pid is None or tid is None:
            continue
        if ph == "X":
            ts = int(e.get("ts", 0))
            dur = int(e.get("dur", 0))
            if dur <= 0:
                continue
            intervals_by_thread[(pid, tid)].append((ts, ts + dur))

    # Second pass: handle 'B'/'E' nesting for setup names
    stacks: DefaultDict[Tuple[int, int], List[int]] = defaultdict(list)
    for e in events:
        name = e.get("name", "")
        ph = e.get("ph")
        pid = e.get("pid")
        tid = e.get("tid")
        if pid is None or tid is None:
            continue
        key = (pid, tid)
        if ph == "B" and isinstance(name, str) and name.startswith("setup/"):
            stacks[key].append(int(e.get("ts", 0)))
        elif ph == "E" and stacks.get(key):
            start_ts = stacks[key].pop()
            end_ts = int(e.get("ts", start_ts))
            if end_ts > start_ts:
                intervals_by_thread[key].append((start_ts, end_ts))

    # Merge intervals per thread
    merged_by_thread: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for key, ivals in intervals_by_thread.items():
        if not ivals:
            continue
        ivals.sort()
        merged: List[Tuple[int, int]] = []
        cur_s, cur_e = ivals[0]
        for s, e in ivals[1:]:
            if s <= cur_e:
                cur_e = max(cur_e, e)
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append((cur_s, cur_e))
        merged_by_thread[key] = merged
    return merged_by_thread


def _event_within_any_interval(evt: Dict[str, Any], merged_by_thread: Dict[Tuple[int, int], List[Tuple[int, int]]]) -> bool:
    pid = evt.get("pid")
    tid = evt.get("tid")
    if pid is None or tid is None:
        return False
    key = (pid, tid)
    ivals = merged_by_thread.get(key)
    if not ivals:
        return False
    ph = evt.get("ph")
    ts = int(evt.get("ts", 0))
    if ph == "X":
        dur = int(evt.get("dur", 0))
        end_ts = ts + max(dur, 0)
        ev_s, ev_e = ts, end_ts
    else:
        # Treat as instant at ts
        ev_s = ev_e = ts
    # Binary search could be used; linear is fine for moderate sizes
    return any(ev_s >= s and ev_e <= e for s, e in ivals)

def filter_trace(input_path: str, output_path: str) -> None:
    with open(input_path, "r") as f:
        data = json.load(f)

    # Chrome trace formats we may see:
    # 1) {"traceEvents": [...], ...}
    # 2) [ ... events ... ]
    if isinstance(data, dict) and "traceEvents" in data and isinstance(data["traceEvents"], list):
        events = data["traceEvents"]
        original_len = len(events)
        merged = _collect_setup_intervals(events)
        filtered = [e for e in events if not _is_setup_event(e) and not _event_within_any_interval(e, merged)]
        removed = original_len - len(filtered)
        data["traceEvents"] = filtered
    elif isinstance(data, list):
        events = data
        original_len = len(events)
        merged = _collect_setup_intervals(events)
        data = [e for e in events if not _is_setup_event(e) and not _event_within_any_interval(e, merged)]
        removed = original_len - len(data)
    else:
        raise ValueError("Unsupported trace JSON structure. Expected a list or an object with 'traceEvents'.")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f)

    print(f"Filtered trace written to {output_path} (removed {removed} setup events)")


def main():
    parser = argparse.ArgumentParser(description="Filter out setup/* events from a Chrome trace JSON file")
    parser.add_argument("input", type=str, help="Path to input Chrome trace JSON (e.g., .pt.trace.json)")
    parser.add_argument("output", type=str, nargs="?", default=None, help="Path to output filtered JSON")
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    output_path = args.output
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = base + ".filtered" + ext

    filter_trace(input_path, os.path.abspath(output_path))


if __name__ == "__main__":
    main()


