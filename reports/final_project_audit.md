# Final Project Audit

Generated: 2026-08-31T22:54:28+00:00

## Evidence-based status

- Model/checkpoint records: 197
- Benchmark isolation: `BLOCKED_OFFICIAL_BENCHMARK_HASH_LIST_REQUIRED`
- Public privacy findings: 17
- Unsafe training-script findings: 59

## Frontend

The active application is served from `app/static`. The current homepage must remain a targeted polish pass, not a redesign.

## Backend

Two API implementations are present (`app/server.py` and `deployment/api.py`). They expose overlapping but incompatible contracts and need consolidation before public deployment.

## Model pipeline and history

The frozen production report describes a 735,038,561-parameter model. Separate reports also describe multi-expert and high-capacity candidates. These are competing artifacts, not proof of one final model, until the checkpoint, inference class, and independent held-out evaluation are reconciled.

## Datasets, training, and validation

Training manifests contain real record-level hashes, but the official organizer benchmark hash list is not present in this repository. Isolation therefore remains blocked rather than assumed.

## Robustness, provenance, and reporting

Spatial and provenance modules exist. Public claims must be limited to metrics and artifacts that the final selected inference pipeline can reproduce.

## Deployment and GitHub readiness

A root public Git repository is not initialized. The nested `app` repository has independent history. Consolidate deliberately before publishing so the public repository contains the complete application without secrets or private paths.

## Blocking findings

- Public privacy scan: `app/static/index.html` contains `buildabot`.
- Public privacy scan: `app/static/index.html` contains `cuda`.
- Public privacy scan: `app/static/index.html` contains `rtx`.
- Public privacy scan: `app/static/app.js` contains `buildabot`.
- Public privacy scan: `app/static/app.js` contains `cuda`.
- Public privacy scan: `app/static/app.js` contains `rtx`.
- Public privacy scan: `app/static/app.js` contains `100.69.`.
- Public privacy scan: `frontend/index.html` contains `buildabot`.
- Public privacy scan: `frontend/index.html` contains `cuda`.
- Public privacy scan: `frontend/index.html` contains `rtx`.
- Public privacy scan: `frontend/app.js` contains `buildabot`.
- Public privacy scan: `frontend/app.js` contains `cuda`.
- Public privacy scan: `frontend/app.js` contains `rtx`.
- Public privacy scan: `frontend/app.js` contains `100.69.`.
- Public privacy scan: `README.md` contains `buildabot`.
- Public privacy scan: `README.md` contains `ssh root@`.
- Public privacy scan: `README.md` contains `/home/manan`.
- Training safety: `scripts/benchmark_io_and_pilot_training.py` contains `zero tensor fallback`.
- Training safety: `scripts/benchmark_io_and_pilot_training.py` contains `synthetic fusion logits`.
- Training safety: `scripts/execute_actual_large_scale_training_and_feedback.py` contains `random tensor fallback`.
- Training safety: `scripts/execute_authoritative_final_training.py` contains `random tensor fallback`.
- Training safety: `scripts/execute_authoritative_final_training.py` contains `synthetic fusion logits`.
- Training safety: `scripts/execute_final_training_and_feedback_v6.py` contains `zero tensor fallback`.
- Training safety: `scripts/execute_final_training_and_feedback_v6.py` contains `random tensor fallback`.
- Training safety: `scripts/execute_fresh_master_training_protocol.py` contains `random tensor fallback`.
- Training safety: `scripts/execute_master_remediation_pipeline.py` contains `random tensor fallback`.
- Training safety: `scripts/execute_master_remediation_pipeline.py` contains `synthetic fusion logits`.
- Training safety: `scripts/execute_master_training_pipeline_final.py` contains `synthetic fusion logits`.
- Training safety: `scripts/execute_optimized_full_training_v6.py` contains `zero tensor fallback`.
- Training safety: `scripts/execute_optimized_full_training_v6.py` contains `random tensor fallback`.
- Training safety: `scripts/execute_true_full_training.py` contains `zero tensor fallback`.
- Training safety: `scripts/execute_true_full_training.py` contains `random tensor fallback`.
- Training safety: `scripts/execute_true_full_training_v6.py` contains `zero tensor fallback`.
- Training safety: `scripts/execute_true_full_training_v6.py` contains `random tensor fallback`.
- Training safety: `scripts/fast_v3_fusion_audit_and_feedback.py` contains `zero tensor fallback`.
- Training safety: `scripts/final_master_training_pipeline.py` contains `random tensor fallback`.
- Training safety: `scripts/final_pretraining_reconciliation_engine.py` contains `random tensor fallback`.
- Training safety: `scripts/final_pretraining_reconciliation_v2.py` contains `random tensor fallback`.
- Training safety: `scripts/fresh_definitive_master_training.py` contains `random tensor fallback`.
- Training safety: `scripts/master_production_remediation_pipeline_v2.py` contains `random tensor fallback`.
- Training safety: `scripts/master_production_remediation_pipeline_v2.py` contains `synthetic fusion logits`.
- Training safety: `scripts/phase4_master_execution_pipeline.py` contains `random tensor fallback`.
- Training safety: `scripts/phase5_master_execution_engine.py` contains `random tensor fallback`.
- Training safety: `scripts/phase6_master_validation_engine.py` contains `random tensor fallback`.
- Training safety: `scripts/phase7_master_pretraining_validation_engine.py` contains `random tensor fallback`.
- Training safety: `scripts/phase_b_and_c_specialist_and_fusion.py` contains `zero tensor fallback`.
- Training safety: `scripts/run_all_models_fusion_master_experiment.py` contains `synthetic fusion logits`.
- Training safety: `scripts/run_definitive_master_training.py` contains `random tensor fallback`.
- Training safety: `scripts/run_fresh_master_execution.py` contains `random tensor fallback`.
- Training safety: `scripts/run_fresh_master_execution.py` contains `synthetic fusion logits`.
- Training safety: `scripts/run_master_protocol_sections_8_to_16.py` contains `synthetic fusion logits`.
- Training safety: `scripts/run_master_raw_image_training_session.py` contains `zero tensor fallback`.
- Training safety: `scripts/run_master_raw_image_training_session.py` contains `random tensor fallback`.
- Training safety: `scripts/run_stage1_master_evaluation.py` contains `random tensor fallback`.
- Training safety: `scripts/sanity_test_trainable_vision.py` contains `zero tensor fallback`.
- Training safety: `scripts/sanity_test_trainable_vision.py` contains `random tensor fallback`.
- Training safety: `scripts/train_buildabot_gpu_fusion.py` contains `synthetic fusion logits`.
- Training safety: `scripts/train_final_highres_specialists_master.py` contains `zero tensor fallback`.
- Training safety: `scripts/train_full_260k_corpus.py` contains `random tensor fallback`.
- Training safety: `scripts/train_full_260k_corpus.py` contains `synthetic fusion logits`.
- Training safety: `scripts/train_full_3epoch_specialists_and_3epoch_fusion.py` contains `zero tensor fallback`.
- Training safety: `scripts/train_heavy_production_specialist.py` contains `zero tensor fallback`.
- Training safety: `scripts/train_large_scale_robust_detector.py` contains `synthetic fusion logits`.
- Training safety: `scripts/train_live_all_models_c0_c7_unified_engine.py` contains `zero tensor fallback`.
- Training safety: `scripts/train_master_production_specialists.py` contains `zero tensor fallback`.
- Training safety: `scripts/train_phase2_detector.py` contains `synthetic PIL fallback`.
- Training safety: `scripts/train_phase3_fusion_challenge.py` contains `zero tensor fallback`.
- Training safety: `scripts/train_phase3_fusion_challenge.py` contains `random tensor fallback`.
- Training safety: `scripts/train_phase3_fusion_challenge.py` contains `synthetic fusion logits`.
- Training safety: `scripts/train_portrait_rem_1.py` contains `zero tensor fallback`.
- Training safety: `scripts/train_portrait_rem_1.py` contains `synthetic fusion logits`.
- Training safety: `scripts/train_quad_hybrid_gating.py` contains `zero tensor fallback`.
- Training safety: `scripts/train_sequential_fp32_full_precision_pipeline.py` contains `zero tensor fallback`.
- Training safety: `scripts/train_sequential_fp32_full_precision_pipeline.py` contains `synthetic fusion logits`.
- Training safety: `scripts/train_v3_gating_fusion_and_audit.py` contains `zero tensor fallback`.
- Training safety: `scripts/train_v3_master_pipeline.py` contains `synthetic PIL fallback`.
