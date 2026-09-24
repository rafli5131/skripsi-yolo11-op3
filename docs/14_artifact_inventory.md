# Inventaris Artifact Penting

Daftar ini membedakan artifact final, comparator, hasil evaluasi, dan diagnostic historis. Hash lengkap dan perintah audit ada di [provenance](26_provenance_dan_verifikasi.md).

| Artifact | Lokasi | Status saat ini |
| --- | --- | --- |
| FP32 checkpoint final | `artifacts/checkpoints/fp32/best.pt` | Beku; SHA256 `e05b2b21...d420b8c` |
| QAT checkpoint final | `artifacts/checkpoints/qat/qat_best.pt` | Beku; SHA256 `b9e520cb...fc35430d` |
| QAT selection manifest | `artifacts/checkpoints/qat/selection_manifest.json` | Winner dipilih dari VAL; TEST/PTQ metrics tidak dipakai |
| QAT search candidates | `artifacts/checkpoints/qat_search/` | Riwayat pemilihan VAL; bukan file final untuk deployment |
| FP32 OpenVINO IR | `artifacts/openvino/fp32/model.{xml,bin}` | Baseline valid |
| PTQ OpenVINO IR canonical | `experiments/ptq/backend_models/ptq_openvino_model/model.{xml,bin}` | Existing comparator valid |
| QAT OpenVINO IR final | `artifacts/openvino/qat_int8/model.{xml,bin}` | Direct OpenVINO; valid setelah audit numerical/provenance |
| Final TEST PyTorch | `experiments/final_test/` | Hasil held-out yang telah dibekukan |
| PTQ VAL diagnostic | `experiments/ptq/` | Evaluasi pembanding, bukan TEST |
| Server benchmark | `experiments/server_openvino/` | CPU host diagnostic FP32/PTQ/QAT |
| Hardware OP3 | `experiments/hardware_op3/` | Bukti pengukuran hardware terpisah |
| Diagnostic export | `artifacts/openvino/*diagnostic.json` | Bukti accepted maupun rejected; baca status masing-masing |

Salinan valid `artifacts/openvino/ptq_int8/` memiliki hash IR yang sama dengan comparator PTQ canonical. File model kandidat ONNX dan IR QAT lama yang rejected telah dibersihkan dari worktree; JSON diagnosticnya tetap disimpan. Baca [analisis kegagalan](25_kegagalan_dan_pembersihan.md) untuk daftar penghapusan dan alasan ilmiahnya.
