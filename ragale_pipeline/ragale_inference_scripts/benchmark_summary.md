# Ragale Benchmark Results

Generated at: 2026-04-08 16:20:17

| Model              | Strategy          | Valid | Total | Valid% | Avg Time | Avg Tokens | Avg Masks | Avg Backtracks |
|:-------------------|:------------------|------:|------:|-------:|---------:|-----------:|----------:|---------------:|
| gemma3-1b-base     | masking_only      |     3 |    20 |  15.0% |     3.6s |         74 |        72 |            0.0 |
| gemma3-1b-base     | masking_backtrack |     3 |    20 |  15.0% |     3.6s |         74 |        72 |            0.0 |
| gemma3-1b-base     | hybrid            |    13 |    20 |  65.0% |     1.5s |         34 |        32 |            0.1 |
| gemma3-1b-lora     | masking_only      |    20 |    20 | 100.0% |     1.4s |         26 |        24 |            0.0 |
| gemma3-1b-lora     | masking_backtrack |    20 |    20 | 100.0% |     1.4s |         26 |        24 |            0.0 |
| gemma3-1b-lora     | hybrid            |    20 |    20 | 100.0% |     1.5s |         26 |        24 |            0.0 |
| gemma4-e2b-base    | masking_only      |     7 |    20 |  35.0% |     3.1s |         29 |        27 |            0.0 |
| gemma4-e2b-base    | masking_backtrack |     7 |    20 |  35.0% |     3.0s |         29 |        27 |            0.0 |
| gemma4-e2b-base    | hybrid            |    10 |    20 |  50.0% |     2.9s |         28 |        26 |            0.0 |
