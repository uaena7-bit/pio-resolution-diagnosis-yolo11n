#!/usr/bin/env python3
from pathlib import Path
import argparse, csv, sys

EXPECTED = [
 'README.md', 'release_notes_v1.2.0.md',
 'audit_summaries/PigDetect_clean_split_audit_summary.csv',
 'audit_summaries/PigDetect_source_to_table_map.csv',
 'supplementary_source_data/Table_S17a_PigDetect_clean_split_audit.csv',
 'supplementary_source_data/Table_S17b_PigDetect_training_and_locked_test_performance.csv',
 'supplementary_source_data/Table_S17c_PigDetect_scale_stratified_AP.csv',
 'supplementary_source_data/Table_S17d_PigDetect_density_stratified_AP.csv',
 'supplementary_source_data/Table_S17e_PigDetect_paired_bootstrap_summary.csv',
 'supplementary_source_data/Table_S17f_PigDetect_locked_weight_deployment_benchmark.csv',
 'supplementary_source_data/Table_S17g_PigDetect_source_to_manuscript_map.csv',
 'figure_source_data/Table12_PigDetect_main_summary_source_data.csv',
 'figure_source_data/Fig6_PigDetect_qualitative_evidence_metadata.json',
 'main_figures/Fig6_PigDetect_external_validation.pdf',
 'main_figures/Fig6_PigDetect_external_validation.png',
 'external_validation/PigDetect/protocol.json'
]

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--repo-root',default='.'); a=ap.parse_args()
 root=Path(a.repo_root).resolve(); errors=[]
 for rel in EXPECTED:
  if not (root/rel).exists(): errors.append('Missing: '+rel)
 readme=(root/'README.md').read_text(encoding='utf-8',errors='replace') if (root/'README.md').exists() else ''
 if 'v1.2.0' not in readme: errors.append('README does not identify v1.2.0')
 if 'PigDetect' not in readme: errors.append('README does not describe PigDetect')
 # Check locked Table 12 rows
 p=root/'figure_source_data/Table12_PigDetect_main_summary_source_data.csv'
 if p.exists():
  rows=list(csv.DictReader(p.open(encoding='utf-8')))
  expected={'800':'0.7673','960':'0.7757','1280':'0.7844'}
  got={r['Input (px)']:r['mAP50-95'] for r in rows}
  if got!=expected: errors.append(f'Table 12 mAP mismatch: {got}')
 if errors:
  print('ALIGNMENT CHECK: FAIL')
  for e in errors: print(' -',e)
  sys.exit(1)
 print('ALIGNMENT CHECK: PASS')
 print(f'Checked {len(EXPECTED)} expected files and locked Table 12 values.')

if __name__=='__main__': main()
