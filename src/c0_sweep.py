#!/usr/bin/env python3
import argparse, os, sys, subprocess, itertools, time, re
from pathlib import Path

def parse_float_list(s):
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def parse_int_list(s):
    return [int(x.strip()) for x in s.split(",") if x.strip()]

# Heuristic: detect tqdm-ish progress bar lines and skip them in logs
_PBAR_RES = [
    re.compile(r'^\s*\d{1,3}%\|[#\s]+\|\s*\d+/\d+\s*\[.*?it/s\].*$'),
    re.compile(r'^\s*\d{1,3}%\|[#\s]+\|\s*\d+/\d+.*$'),
    re.compile(r'^\s*\d+/\d+\s+\[.*?it/s\].*$'),
]

def is_progress_line(line: str) -> bool:
    if "it/s" in line and "|" in line:
        return True
    for rg in _PBAR_RES:
        if rg.match(line):
            return True
    return False

def main():
    ap = argparse.ArgumentParser(description="Run C0p pruning parameter sweep with deterministic logging.")
    ap.add_argument("--project_root", type=str, default=".", help="Repo root where src/aron_main.py lives")
    ap.add_argument("--python", type=str, default=sys.executable, help="Python executable to use")
    ap.add_argument("--dataset", type=str, default="Cora")
    ap.add_argument("--ver", type=str, default="aron_desc_intra", choices=["aron_desc_intra","aron_desc_inter"])
    ap.add_argument("--seeds", type=str, default="0,1,2", help="Comma-separated seeds")
    ap.add_argument("--alphas", type=str, default="0.6,0.7,0.8", help="Comma-separated restrict_alpha values")
    ap.add_argument("--gammas", type=str, default="1.0", help="Comma-separated restrict_gamma values")
    ap.add_argument("--prune_fracs", type=str, default="0.00,0.05,0.10,0.15,0.20", help="Comma-separated c0p_prune_frac values")
    ap.add_argument("--gmm_k", type=int, default=16)
    ap.add_argument("--gmm_tau", type=float, default=0.55)
    ap.add_argument("--aug_ratio", type=float, default=0.10)
    ap.add_argument("--aug_bound", type=float, default=0.10)
    ap.add_argument("--epochs", type=int, default=700)
    ap.add_argument("--date_tag", type=str, default=time.strftime("%m%d"), help="Folder under logs/ to store logs")
    ap.add_argument("--extra", type=str, default="", help="Extra args appended to aron_main.py (e.g., --cuda 0)")
    ap.add_argument("--dry_run", action="store_true", help="Print commands only, do not execute")
    ap.add_argument("--continue_on_error", action="store_true", help="Keep sweeping if a job fails")
    # NEW: control log cleanliness from *this* script only
    ap.add_argument("--add_sweep_mode", action="store_true",
                    help="Append --sweep_mode to training cmd and disable tqdm via env.")
    ap.add_argument("--filter_progress", action="store_true",
                    help="Filter tqdm-like progress lines from logs (harmless if already disabled).")
    args = ap.parse_args()

    seeds = parse_int_list(args.seeds)
    alphas = parse_float_list(args.alphas)
    gammas = parse_float_list(args.gammas)
    prune_fracs = parse_float_list(args.prune_fracs)

    # Ensure log dir
    logdir = Path(args.project_root) / "logs" / args.date_tag
    logdir.mkdir(parents=True, exist_ok=True)

    combos = list(itertools.product(seeds, alphas, gammas, prune_fracs))
    print(f"[SWEEP] dataset={args.dataset} ver={args.ver} | total runs={len(combos)}")
    print(f"[SWEEP] seeds={seeds} alphas={alphas} gammas={gammas} prune_fracs={prune_fracs}")
    print(f"[SWEEP] logs -> {logdir}")

    # Base env: silence tqdm if requested
    base_env = os.environ.copy()
    if args.add_sweep_mode:
        base_env["TQDM_DISABLE"] = "1"      # many tqdm installs honor this
    base_env["PYTHONUNBUFFERED"] = "1"      # ensure real-time line flushing

    for (seed, alpha, gamma, prune) in combos:
        # --- filename built ONLY from loop vars (what collector expects) ---
        a_str   = f"{alpha:.2f}"
        g_str   = f"{gamma:.2f}"
        c0p_str = f"{prune:.2f}"
        fname = f"{args.dataset}_{args.ver}_a{a_str}_g{g_str}_c0p{c0p_str}_seed{seed}.log"
        log_path = logdir / fname

        cmd = [
            args.python, "-u", str(Path(args.project_root) / "src" / "aron_main.py"),
            "--dataset", args.dataset,
            "--ver", args.ver,
            "--cluster_method", "gmm",
            "--gmm_k", str(args.gmm_k),
            "--gmm_tau", str(args.gmm_tau),
            "--restricted",
            "--restrict_alpha", str(alpha),
            "--restrict_gamma", str(gamma),
            "--aug_ratio", str(args.aug_ratio),
            "--aug_bound", str(args.aug_bound),
            "--epochs", str(args.epochs),
            "--seed", str(seed),
            "--date", args.date_tag,
            "--c0p_prune_frac", str(prune),
            # tip: if you didn't patch aron_main.py with --sweep_mode,
            # drop "--logging" to avoid internal redirection.
            "--logging",
        ]
        if args.add_sweep_mode:
            cmd += ["--sweep_mode"]
        if args.extra:
            cmd += args.extra.strip().split()

        print("="*120)
        print("[RUN]", " ".join(cmd))
        print("[LOG]", log_path)
        print("="*120)

        if args.dry_run:
            continue

        with open(log_path, "w", buffering=1) as f:
            proc = subprocess.Popen(
                cmd, cwd=args.project_root,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, env=base_env
            )
            try:
                for raw in proc.stdout:
                    line = raw.rstrip("\n")
                    if args.filter_progress and is_progress_line(line):
                        continue
                    sys.stdout.write(line + "\n")
                    f.write(line + "\n")
                ret = proc.wait()
            except KeyboardInterrupt:
                proc.terminate()
                ret = proc.wait()
            finally:
                print(f"[RUN-END] returncode={ret} | log={log_path}")
                if ret != 0 and not args.continue_on_error:
                    print("[SWEEP] Aborting on error (use --continue_on_error to proceed).")
                    sys.exit(ret)


    print("[SWEEP] Completed all runs.")
    print(f"[SWEEP] Next: python collect_c0p_results.py --log_glob '{logdir}/{args.dataset}_{args.ver}_*.log'")

if __name__ == "__main__":
    main()
