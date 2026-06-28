# Frontiers submission notes

## Final display structure

- Main manuscript: Tables 1-10 and Figures 1-4.
- Supplementary material: Tables S1-S11 and Figures S1-S4.
- Final files are generated in `results/frontiers_submission`.
- Figures are exported as 300 dpi RGB PNG and LZW-compressed TIFF files.

## Interpretation boundaries and limitations

1. Outcomes are severity categories derived from complete questionnaire scores,
   not independent clinical diagnoses.
2. The source questionnaires were administered in a fixed order. The analysis is
   an offline observed-prefix simulation, not a prospective or deployment-ready
   computerized adaptive test.
3. RT score adjustment does not improve primary severity classification, and RT
   calibration does not outperform strong generic post-hoc calibration controls.
4. The pooled short-form error and overconfident-error results are strongly
   PSS-influenced. Scale-specific thresholds may be preferable.
5. High-risk false negatives are rare, so their detection and escalation capture
   estimates are unstable.
6. Leave-one-scale-out transfer is negative; the RT audit increment is not
   scale-invariant.
7. Audit-detector training uses deterministic subsampling for computational
   tractability. Validation and test evaluation use the complete disjoint splits.

## Submission checks

- Keep the wording focused on RT as process-audit evidence for selective
  full-form escalation.
- Do not describe RT as a clinical diagnostic enhancer, generic calibrator, or
  proven adaptive-testing rule.
- Upload main and supplementary figures separately using the numbered TIFF files
  in `results/frontiers_submission`.
