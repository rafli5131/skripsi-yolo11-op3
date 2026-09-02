# Implementasi QAT dengan DetectionTrainer

## Tujuan

Merekam integrasi NNCF QAT ke lifecycle Ultralytics DetectionTrainer.

## Status

**DONE**.

`best.pt` FP32 dimuat sebagai arsitektur YOLO, parameter trainability dipulihkan mengikuti trainer Ultralytics, lalu satu compressed structure dibuat dengan `create_compressed_model`. Detection loss YOLO tetap digunakan. Kebijakan DFL membekukan 16 parameter DFL; parameter floating-point lain dipulihkan untuk training. Optimizer AdamW dibuat oleh DetectionTrainer terhadap model compressed tersebut.

Range init memakai `YoloTrainInitializingDataLoader` dan `init_range(PTRangeInitParams)` dari subset TRAIN deterministik. Checkpoint QAT menyimpan `model_state_dict`, `compression_state`, dan `nncf_config`; checkpoint training juga menyimpan epoch, optimizer/scheduler/scaler/EMA bila tersedia. Restore: FP32 architecture -> compressed structure dengan config/state -> `nncf.torch.load_state(..., is_resume=True)`.

Ekspektasi tervalidasi: 184 `SymmetricQuantizer`, 1.323 state entries NNCF.

## Artefak terkait

[`scripts/train_qat.py`](../scripts/train_qat.py), [`nncf_checkpoint_format.json`](../artifacts/checkpoints/qat/nncf_checkpoint_format.json), [QAT final](07_final_qat_training.md).
