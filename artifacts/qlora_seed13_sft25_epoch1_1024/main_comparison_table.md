| system | base | four_module | accuracy | macro_f1 | ece | tie_recall |
| --- | --- | --- | --- | --- | --- | --- |
| Raw M-Prometheus-3B | frozen | no | 0.5624 | 0.4079 |  |  |
| Current BEA-Judge | frozen | yes | 0.7512 | 0.673 | 0.0558 | 0.5231 |
| QLoRA-M-Prometheus-3B | qlora | no | 0.760684 | 0.5948 | 0.2403 | 0.130769 |
| QLoRA-BEA-Judge | qlora | yes | 0.793 | 0.7072 | 0.0249 | 0.4846 |
