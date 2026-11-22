#!/usr/bin/env python3
"""
Batch generate synthetic body renders.

This script orchestrates the generation of multiple synthetic bodies
by calling Blender in background mode for each one.

Usage:
    python scripts/batch_generate.py --count 100 --output data/renders
"""

import argparse
import subprocess
import os
import sys
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def find_blender():
    """Find Blender executable."""
    # Common locations
    candidates = [
        "blender",  # In PATH
        "/Applications/Blender.app/Contents/MacOS/Blender",  # macOS
        "/usr/bin/blender",  # Linux
        "C:/Program Files/Blender Foundation/Blender/blender.exe",  # Windows
    ]
    
    for candidate in candidates:
        try:
            result = subprocess.run(
                [candidate, "--version"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                version = result.stdout.split('\n')[0]
                print(f"Found Blender: {candidate}")
                print(f"  Version: {version}")
                return candidate
        except (FileNotFoundError, PermissionError):
            continue
    
    return None


def generate_single(args):
    """Generate a single body. Called by process pool."""
    blender_path, script_path, output_prefix, seed, use_mpfb2 = args
    
    cmd = [
        blender_path,
        "--background",
        "--python", script_path,
        "--",
        "--output", output_prefix,
        "--seed", str(seed),
    ]
    
    if use_mpfb2:
        cmd.append("--use-mpfb2")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout per render
        )
        
        if result.returncode == 0:
            return {"seed": seed, "status": "success", "output": output_prefix}
        else:
            return {
                "seed": seed, 
                "status": "error", 
                "error": result.stderr[-500:] if result.stderr else "Unknown error"
            }
    except subprocess.TimeoutExpired:
        return {"seed": seed, "status": "timeout"}
    except Exception as e:
        return {"seed": seed, "status": "error", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Batch generate synthetic bodies")
    parser.add_argument("--count", "-n", type=int, default=10, 
                        help="Number of bodies to generate")
    parser.add_argument("--output", "-o", default="data/renders",
                        help="Output directory")
    parser.add_argument("--start-seed", type=int, default=0,
                        help="Starting random seed")
    parser.add_argument("--workers", "-w", type=int, default=1,
                        help="Number of parallel workers (Blender instances)")
    parser.add_argument("--use-mpfb2", action="store_true",
                        help="Use MPFB2 for body generation")
    
    args = parser.parse_args()
    
    # Find Blender
    blender_path = find_blender()
    if not blender_path:
        print("Error: Blender not found. Please install Blender and add to PATH.")
        print("  macOS: brew install --cask blender")
        return 1
    
    # Find generation script
    script_dir = Path(__file__).parent
    script_path = script_dir / "generate_body.py"
    if not script_path.exists():
        print(f"Error: Generation script not found: {script_path}")
        return 1
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating {args.count} synthetic bodies...")
    print(f"Output directory: {output_dir}")
    print(f"Workers: {args.workers}")
    print()
    
    # Prepare tasks
    tasks = []
    for i in range(args.count):
        seed = args.start_seed + i
        output_prefix = str(output_dir / f"body_{seed:05d}")
        tasks.append((blender_path, str(script_path), output_prefix, seed, args.use_mpfb2))
    
    # Run generation
    results = []
    
    if args.workers == 1:
        # Sequential execution with progress bar
        for task in tqdm(tasks, desc="Generating"):
            result = generate_single(task)
            results.append(result)
            if result["status"] != "success":
                tqdm.write(f"  Warning: {result}")
    else:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(generate_single, task): task for task in tasks}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating"):
                result = future.result()
                results.append(result)
                if result["status"] != "success":
                    tqdm.write(f"  Warning: {result}")
    
    # Summary
    success = sum(1 for r in results if r["status"] == "success")
    errors = sum(1 for r in results if r["status"] == "error")
    timeouts = sum(1 for r in results if r["status"] == "timeout")
    
    print(f"\n{'='*50}")
    print("GENERATION COMPLETE")
    print(f"{'='*50}")
    print(f"  Success:  {success}/{args.count}")
    print(f"  Errors:   {errors}")
    print(f"  Timeouts: {timeouts}")
    
    # Save results log
    log_path = output_dir / "generation_log.json"
    with open(log_path, 'w') as f:
        json.dump({
            "config": {
                "count": args.count,
                "start_seed": args.start_seed,
                "use_mpfb2": args.use_mpfb2,
            },
            "results": results
        }, f, indent=2)
    print(f"\nLog saved to: {log_path}")
    
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
