# Cognitive-Level Adaptive Content Generation via LLMs with Multi-Layer Knowledge Graphs and Preference-Aware Learning
> The rapid advancement of Large Language Models (LLMs) presents transformative opportunities for education, yet significant challenges persist in aligning with learners' cognitive levels. We identify a critical cognitive misalignment issue in LLMs: (1) inability to recognize learners' knowledge boundaries, leading to responses that either exceed or fall below appropriate cognitive levels; (2) lack of adaptive teaching strategies for different cognitive levels, resulting in suboptimal learning efficiency. Existing approaches struggle to systematically address these challenges due to inadequate cognitive hierarchy classification and inherent preference biases. To bridge this gap, we propose a Hierarchical Paired QA dataset (HPQA) containing tri-level cognitive annotations to capture diverse learning needs. Building upon HPQA, we design a Cognitive-Level Alignment Framework (CLAF) with three synergistic modules: (1) A capability-aware retrieval module leveraging multi-level knowledge graphs for cognitive-level content alignment; (2) A knowledge controllable generation module employing latent vector constraints to enhance output consistency; (3) A adaptive language style optimization module adapting teaching strategies across cognitive levels. Experimental results demonstrate that CLAF achieves significant improvements in cognitive alignment, validating the effectiveness of hierarchical knowledge modeling and strategic adaptation.



###demo

ALSO

``` shell
cd OpenRLHF
# train for sfr
python ./examples/scripts/train_sft_llama.sh
# train for dpo
python ./examples/scripts/train_dpo_llama.sh
```

CARAG
``` shell

cd CARAG
python demo.py

```