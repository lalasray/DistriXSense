# Pretrained-core synthetic cost results

7,749 completed CPU/GPU/mixed combinations; 21 methods/ablations, three cores, 41 dataset/modality settings.
No dataset files downloaded. Random sensor/adapter weights; pretrained frozen language cores. These are cost/integration tests, not accuracy results.
BF16 core, FP32 sensor/adapter, batch1, eight CPU threads, three repetitions plus one warm-up, eight generated tokens.
GPU: NVIDIA RTX PRO 6000 Blackwell (96GB), CUDA PyTorch2.14.1+cu130. Mixed: CPU edges, GPU core. GPU access requires execution outside the sandbox. Same-shape core timing reuse is labeled in summary.csv.

|Input profile|Streams|Known decoded numeric MiB / 3s|Feature MiB / 3s|Unknown raw-size streams|
|---|---:|---:|---:|---:|
|nymeria|41|1550.955|9.876|1|
|openmarcie_bicycle|13|24.517|0.469|0|
|openmarcie_printer|19|48.996|0.899|0|
|opportunity++|5|26.459|0.114|0|

Input amounts use the explicit assumptions in modality_inventory.csv; decoded volumes are not compressed corpus sizes. Media preprocessing costs are excluded.

|All-family proposed model|Core|Target|Deployed parameters (B)|Weights GiB|Counted GFLOPs|Wire KiB/window|Core generation ms|Summed pipeline ms|
|---|---|---|---:|---:|---:|---:|---:|---:|
|opportunity++|gemma3_1b|cpu|1.0015|1.868|155.418|0.410|415.81|418.32|
|openmarcie_bicycle|gemma3_1b|cpu|1.0020|1.870|269.720|1.079|455.61|460.75|
|openmarcie_printer|gemma3_1b|cpu|1.0026|1.873|356.354|1.596|480.07|487.15|
|nymeria|gemma3_1b|cpu|1.0092|1.897|680.860|3.483|610.92|625.32|
|opportunity++|gemma3_1b|cuda|1.0015|1.868|155.418|0.410|80.93|84.91|
|opportunity++|gemma3_1b|mixed|1.0015|1.868|155.418|0.410|80.93|84.03|
|openmarcie_bicycle|gemma3_1b|cuda|1.0020|1.870|269.720|1.079|81.06|90.63|
|openmarcie_bicycle|gemma3_1b|mixed|1.0020|1.870|269.720|1.079|81.06|87.68|
|openmarcie_printer|gemma3_1b|cuda|1.0026|1.873|356.354|1.596|80.76|93.55|
|openmarcie_printer|gemma3_1b|mixed|1.0026|1.873|356.354|1.596|80.76|91.85|
|nymeria|gemma3_1b|cuda|1.0092|1.897|680.860|3.483|80.78|105.13|
|nymeria|gemma3_1b|mixed|1.0092|1.897|680.860|3.483|80.78|98.17|
|opportunity++|qwen3_4b|cpu|4.0293|7.518|820.190|0.410|1434.71|1437.73|
|openmarcie_bicycle|qwen3_4b|cpu|4.0299|7.520|1416.329|1.079|1572.10|1578.28|
|openmarcie_printer|qwen3_4b|cpu|4.0305|7.522|1868.402|1.596|1718.28|1727.04|
|nymeria|qwen3_4b|cpu|4.0370|7.547|3562.622|3.483|2253.00|2269.99|
|opportunity++|qwen3_4b|cuda|4.0293|7.518|820.190|0.410|80.56|84.62|
|opportunity++|qwen3_4b|mixed|4.0293|7.518|820.190|0.410|80.56|83.73|
|openmarcie_bicycle|qwen3_4b|cuda|4.0299|7.520|1416.329|1.079|83.26|92.57|
|openmarcie_bicycle|qwen3_4b|mixed|4.0299|7.520|1416.329|1.079|83.26|90.08|
|openmarcie_printer|qwen3_4b|cuda|4.0305|7.522|1868.402|1.596|84.35|97.46|
|openmarcie_printer|qwen3_4b|mixed|4.0305|7.522|1868.402|1.596|84.35|94.18|
|nymeria|qwen3_4b|cuda|4.0370|7.547|3562.622|3.483|90.72|115.34|
|nymeria|qwen3_4b|mixed|4.0370|7.547|3562.622|3.483|90.72|108.52|
|opportunity++|qwen3_8b|cpu|8.2079|15.320|1560.110|0.410|2576.94|2581.11|
|openmarcie_bicycle|qwen3_8b|cpu|8.2084|15.322|2687.556|1.079|2842.50|2851.04|
|openmarcie_printer|qwen3_8b|cpu|8.2090|15.325|3538.109|1.596|3057.44|3069.35|
|nymeria|qwen3_8b|cpu|8.2156|15.349|6693.425|3.483|4012.25|4041.82|
|opportunity++|qwen3_8b|cuda|8.2079|15.320|1560.110|0.410|109.90|113.99|
|opportunity++|qwen3_8b|mixed|8.2079|15.320|1560.110|0.410|109.90|113.08|
|openmarcie_bicycle|qwen3_8b|cuda|8.2084|15.322|2687.556|1.079|112.33|121.74|
|openmarcie_bicycle|qwen3_8b|mixed|8.2084|15.322|2687.556|1.079|112.33|119.07|
|openmarcie_printer|qwen3_8b|cuda|8.2090|15.325|3538.109|1.596|115.67|128.63|
|openmarcie_printer|qwen3_8b|mixed|8.2090|15.325|3538.109|1.596|115.67|125.06|
|nymeria|qwen3_8b|cuda|8.2156|15.349|6693.425|3.483|124.95|149.18|
|nymeria|qwen3_8b|mixed|8.2156|15.349|6693.425|3.483|124.95|142.35|

286 core-shape benchmarks measured (including remeasurements after resume); 7463 combinations reuse a matched core stage. Each sensor/adapter configuration is separately profiled.
Transport validation passed. 3 cases show CPU/GPU re-encoding differences above1e-5 (maximum8.1e-05). Transport compares the same edge outputs before and after serialization. These separate checks remain visible in the CSV and JSON.
Full comparisons: summary.csv; each physical/aggregate edge and core: devices.csv; detailed stage and unsupported-FLOP data: individual core directories.
Transfer estimates assume10Mbps links with2ms latency and28-byte packet overhead. FLOPs omit unsupported operators. Three timing samples do not establish reliable p95 performance.
ImageBind is excluded because pretrained ImageBind encoder/features were not tested. Corpus compressed modality sizes and energy consumption are unknown.
