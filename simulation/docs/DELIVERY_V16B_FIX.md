# V16b consumer layout-identity correction

The raw close-pass flag trajectory contains capture RNG seed 20261110 and source_layout_seed 20261118. Its episode identity, targets.seed and all source-layout SHA values bind the true layout 20261118. The prior consumer incorrectly compared the RNG field with layout identity and rejected this complete unit before any provider call.

V16b checks explicit non-null source_layout_seed against queue layout identity; it does not treat capture RNG as layout identity. Episode, target, targets.seed, every scene-file hash, dense completion, controls, raw payload, ready and remote receipt checks remain required. Historical legacy data without an explicit layout field use those independent bindings. The exact fixed consumer and CPU positive/negative proof are included.

This seal is a new directory. The previous V16 seal, actual Astro observations, compiled queries, real CoT, source generation provenance, and all old failure state are preserved unchanged. Full SHA inventory is repeated. Previous dense/stride/query relocation and USD dependency closure results are inherited only for byte-identical payloads; they are not represented as newly run. The old credential scan is inherited for identical byte content, and new unique content is confidentially scanned again. No API/GPU/remote operation is performed for this revision.
