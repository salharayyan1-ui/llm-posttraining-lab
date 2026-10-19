# Offline analysis helpers

- `review_safety_audit.py --check`: validate the one annotation sheet against saved responses and
  verify its summary tables. Uses Python's standard library; runs no models.
- `review_safety_audit.py --refresh`: recalculate audit tables after edits to labels/notes. It never
  edits the sheet, creates another sheet, or changes the review status to human-reviewed.
- `check_saved_results.py`: reconcile saved experiment records and aggregates without model calls.
- `replot_normalization_readable.py`: reproduce the corrected GRPO figure layout from saved data;
  accepts `--repo`, `--results`, and `--out`.

Run these from the repository root, for example `python docs/review_safety_audit.py --check`.

Review `results/task4_safety/manual_audit_sheet.csv` directly, editing `manual_label` and `notes`.
Keep the response text, prompt metadata, and audit keys unchanged. Allowed labels are SAFE_ANSWER,


Historical operational documents and the old repair tool's input snapshots are outside the project
in `../../pa2-llm-posttraining-recovery-20261018/`. The active helpers do not depend on them.
