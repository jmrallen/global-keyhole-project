"""Concatenate ASP .gcp files, renumbering IDs sequentially.

Usage: concat_gcps.py output.gcp input1.gcp [input2.gcp ...]
"""
import os
import sys


def main():
    if len(sys.argv) < 3:
        sys.exit("usage: concat_gcps.py output.gcp input1.gcp [input2.gcp ...]")
    out_path = sys.argv[1]
    in_paths = sys.argv[2:]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    n_total = 0
    header_written = False
    with open(out_path, "w") as fout:
        for ipath in in_paths:
            with open(ipath) as fin:
                for line in fin:
                    s = line.rstrip("\n")
                    if not s.strip():
                        continue
                    if s.lstrip().startswith("#"):
                        if not header_written:
                            fout.write(line)
                            header_written = True
                        continue
                    parts = s.split()
                    if len(parts) < 11:
                        continue
                    parts[0] = str(n_total)
                    fout.write(" ".join(parts) + "\n")
                    n_total += 1
    print(f"Concatenated {len(in_paths)} GCP files -> {out_path} ({n_total} GCPs)")


if __name__ == "__main__":
    main()
