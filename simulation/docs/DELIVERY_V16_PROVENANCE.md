# V16 source and sample provenance

The top-level code is the current V16 reusable implementation. `provenance/frozen_sources` also preserves the exact independently frozen Astro generation, production raw-v16b, and annotation-consumer sources; their SHA manifests identify differences. These are provenance snapshots, not additional entry points.

`examples/dataset_smoke` is the complete accepted 1,130-frame Astro unit. It includes the dense dataset, stride-five view, isolated 16-train/4-heldout query indexes, and native RGB224/CoT review. The default training index is its `query_partitions/query_train_index.jsonl`. All 20 queries use credited provider-authored phrase templates plus explicit protocol grammar and deterministic actual-event binding; zero new query HTTP requests were made. Its five image CoT labels are actual Qwen3.8-27B responses. The public request/response audit contains prompts, schemas, hashes, public content, and usage, with no credentials, headers, hidden reasoning, or image Base64. No live shared provider cache is packaged.

The top-level `review/render_report.json` and PNGs are earlier real Isaac asset-detail review evidence, preserved with original provenance. The actual Astro trajectory review is `examples/dataset_smoke/review/seed_20261118_astronaut`. No new GPU render or model training was performed for packaging.

For a new host start from `configs/portable_raw_batch_v16.json` and `configs/portable_annotation_queue_v16.json`. They reference the included `templates/lunar_fork_v11` and compiled query bank. Fill host-specific Python/Isaac SDK, source/output/queue paths, SSH destination, external credential-file path, and shared budget-ledger path in new local configs. Keep the external shared ledger for accounting; do not reset past spend. The portable unit template is `configs/portable_annotation_unit_v16.json`. The original configs and all historical source metadata remain available unchanged.

Rejected free-form query stages and the first V16 producer mkdir race are historical failure evidence, not accepted examples. Production uses raw-v16b. No automatic retry is implied; preserve failed state and use an explicit new revision.

This package proves offline sensor/action/text integrity. `training_ready=false` remains honest: the existing 129 trainer still needs a dataset adapter. Remote SDK runtime execution and trained-policy success are not claimed.
