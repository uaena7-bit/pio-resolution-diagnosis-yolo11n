#!/usr/bin/env python3
"""Apply the v1.2.0 PigDetect repository update to an existing repository root."""
from pathlib import Path
import argparse, shutil, sys

EXCLUDE_DELETE = {'.git', '.github'}

def copy_tree(src: Path, dst: Path):
    for p in src.rglob('*'):
        rel = p.relative_to(src)
        out = dst / rel
        if p.is_dir():
            out.mkdir(parents=True, exist_ok=True)
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo-root', required=True)
    ap.add_argument('--keep-visdrone', action='store_true')
    args = ap.parse_args()
    repo = Path(args.repo_root).expanduser().resolve()
    payload = Path(__file__).resolve().parents[1]
    if not (repo / '.git').exists():
        print(f'WARNING: {repo} does not contain .git; files will still be copied.', file=sys.stderr)
    old_readme = repo / 'README.md'
    if old_readme.exists():
        shutil.copy2(old_readme, repo / 'README_v1.1.0_backup.md')
    # Copy selected payload contents; skip installers and manifest at root.
    for name in ['README.md','release_notes_v1.2.0.md','audit_summaries','supplementary_source_data',
                 'figure_source_data','main_figures','external_validation','repository_audit','scripts']:
        src = payload / name
        dst = repo / name
        if src.is_dir(): copy_tree(src, dst)
        elif src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    removed = []
    if not args.keep_visdrone:
        for p in sorted(repo.rglob('*'), reverse=True):
            if any(part in EXCLUDE_DELETE for part in p.parts):
                continue
            if 'visdrone' in p.name.lower():
                try:
                    if p.is_dir():
                        shutil.rmtree(p)
                    else:
                        p.unlink()
                    removed.append(str(p.relative_to(repo)))
                except FileNotFoundError:
                    pass
    print(f'Installed PigDetect v1.2.0 update into: {repo}')
    if removed:
        print(f'Removed {len(removed)} active VisDrone-named paths:')
        for x in removed: print('  -', x)
    print('\nNext commands:')
    print(f'  cd "{repo}"')
    print('  python scripts/verify_pigdetect_v1_2_alignment.py --repo-root .')
    print('  git status')
    print('  git add -A')
    print('  git commit -m "Replace VisDrone check with PigDetect livestock validation"')
    print('  git tag -a v1.2.0 -m "PigDetect external-validation update"')
    print('  git push origin HEAD --follow-tags')

if __name__ == '__main__': main()
