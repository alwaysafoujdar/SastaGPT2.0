# SastaGPT2.0
This final iteration of SastaGPT introduces several architectural advancements drawn from the DeepSeek-V3 technical report.

## Abstract

This project presents SastaGPT 2.0, the successor to SastaGPT (v1), an autoregressive character-level language model. Building upon the foundation of the Transformer architecture, SastaGPT 2.0 incorporates key architectural enhancements inspired by the DeepSeek V3 technical report. These include the implementation of Mixture-of-Experts (MoE) layers, Key-Value (KV) caching, and gating mechanisms. These modifications led to significant improvements in both training efficiency and the quality of generated text, even under limited computational resources. A subsequent minor iteration, SastaGPT 2.1 — Gotham Protocol (Codename “Alfred”), further refined these techniques through fine-tuning. The results demonstrate the potential of optimized modular transformer architectures to achieve competitive performance compared to much larger models.

## Introduction

The initial SastaGPT model was a basic character-level Transformer model trained on literary texts, facing limitations due to compute and memory constraints. This final project introduces SastaGPT 2.0, an evolution incorporating state-of-the-art techniques like MoE, KV Caching, and simple expert gating. Two variants were developed: SastaGPT 2.0 and SastaGPT 2.1 — Gotham Protocol (Codename “Alfred”). Both leverage modular design principles to significantly enhance training speed and the coherence of the generated output. Notably, SastaGPT 2.1 employed 8 experts with a top-2 routing mechanism, achieving comparable quality to the original v1 model with only 10% of the original training budget.

## Literature Review

This project draws significant inspiration from the following works:

-   **DeepSeek V3 Technical Report (2024):** Provided insights into efficient training of large models through expert sparsity and caching mechanisms.
-   **Vaswani et al. (2017), 'Attention Is All You Need':** The foundational paper introducing the Transformer architecture.
-   **Shazeer et al. (2017), 'Outrageously Large Neural Networks':** Introduced the concept of Mixture-of-Experts routing.
-   **SastaGPT (v1):** Served as the baseline model for comparison and the initial framework for this project.

## Dataset Source and Description

The training data for this project is the same as that used in the midterm project, consisting of two primary datasets:

-   **Shakespeare’s Works:** A compilation of all plays and poems by William Shakespeare, sourced from the Internet Archive and combined into a single text file.
-   **Stoic Philosophy Texts:** Included "Meditations" by Marcus Aurelius, "Letters from a Stoic" by Seneca, and "Enchiridion" and "Discourses" by Epictetus, obtained as pre-processed text files and combined.

Both datasets offer rich grammatical structures suitable for character-level sequence modeling, requiring minimal preprocessing.

## Data Exploration and Important Features

The data was processed into sequences of individual characters for this character-level language model. Key aspects considered include:

-   **Contextual Dependencies:** The `CONTEXT_WINDOW` size determined the amount of preceding text the model considered when generating the next character, significantly impacting output coherence.
-   **Character Encoding:** A simple mapping of each unique character in the combined corpus to a unique integer index was used to create the vocabulary. This ensured each character could be represented as a distinct token for the model.

## New Techniques Implemented

-   **KV Cache:** During inference, the computed key and value tensors from the attention mechanism are cached. This avoids redundant computations for previously processed tokens, leading to improved runtime memory efficiency and reduced latency, particularly beneficial for generating longer sequences.
-   **Mixture-of-Experts (MoE):** A sparse MoE layer with 8 parallel expert networks was implemented. For each input token, a routing mechanism selected the top-2 experts to process the input. Although explicit expert specialization was not enforced during this stage, this implementation demonstrated significant gains in generalization and learning speed.
-   **Gating Mechanisms:** The output of each expert in the MoE layer is weighted by a softmax gating vector. This vector is computed based on the hidden representation of the input token, allowing the model to dynamically assign importance to different experts for different inputs, using a relatively simple routing strategy.

## Model Variants and Observations

-   **SastaGPT 2.0:** This variant incorporated 2 experts with a gating mechanism inspired by decision tree ensemble techniques. Instead of traditional gating weights, the selection and combination of expert outputs were influenced by a tree-like structure. Even without Rotational Position Embeddings (RoPE) and explicit expert specialization, SastaGPT 2.0 outperformed the original v1 model within just 20% of the training steps.
-   **SastaGPT 2.1 — Gotham Protocol (Codename “Alfred”):** This iteration focused on hyperparameter tuning and leveraged 8 experts with top-2 gating, KV Cache, and dropout applied to the gating mechanism. Remarkably, Alfred achieved the quality of SastaGPT v1 in only 500 training iterations, compared to the original 5000 iterations. This 10x improvement highlights the substantial benefits of sparse modular computation. Even with limitations in context length and batch size, a noticeable improvement in output quality was observed.

## Final Hyperparameters
BATCH_SIZE = 32
CONTEXT_WINDOW = 256
EPOCHS = 2500
CHECKPOINT_INTERVAL = 500
LR = 3e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EVAL_ITERS = 200
EMBEDDING_DIM = 512
HEADS = 8
LAYERS = 8
DROPOUT_RATE = 0.1

## Results

The integration of KV Caching and Mixture-of-Experts significantly accelerated convergence and enhanced the final output quality. The following table summarizes the performance comparison between the model versions:

| Model Version             | Parameters | Training Epochs | Performance vs v1                     |
| :------------------------ | :--------- | :---------------- | :------------------------------------ |
| SastaGPT (v1)             | ~10M       | 5000              | Baseline                              |
| SastaGPT 2.0              | ~13M       | 5000              | Beats v1 in just 20% training episodes |
| SastaGPT 2.1              | ~60M       | 2500              | Beats v1 in just 10% training episodes |

## Conclusion

This project demonstrates the effectiveness of employing sophisticated architectural choices like Mixture-of-Experts and Key-Value Caching to achieve significant performance gains in smaller language models. The improvements observed in SastaGPT 2.0 and the remarkable efficiency of SastaGPT 2.1 underscore the potential of modularity and sparsity in enabling efficient training, even with limited computational resources. Future work could explore the integration of Rotational Position Embeddings (RoPE), more sophisticated expert routing strategies, and the adoption of sub-word tokenization methods like Byte-Pair Encoding (BPE) to further enhance the model's capabilities.

## References

-   Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). "Attention Is All You Need."
-   DeepSeek V3 Technical Report, 2024.
-   Shazeer et al., 'Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer', ICLR 2017.
-   Andrej Karpathy Neural Networks Series
-   Internet Archive: Shakespeare’s Works and Stoic Philosophy Texts
