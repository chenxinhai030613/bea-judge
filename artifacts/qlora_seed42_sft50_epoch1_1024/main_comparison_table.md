| system | base | four_module | accuracy | macro_f1 | ece | tie_recall |
| --- | --- | --- | --- | --- | --- | --- |
| Raw M-Prometheus-3B | frozen | no | 0.5624 | 0.4079 |  |  |
| Current BEA-Judge | frozen | yes | 0.7512 | 0.673 | 0.0558 | 0.5231 |
| QLoRA-M-Prometheus-3B | qlora | no | 0.786325 | 0.6319 | 0.2165 | 0.161538 |
| QLoRA-BEA-Judge | qlora | yes | 0.8025 | 0.714 | 0.0205 | 0.4692 |
