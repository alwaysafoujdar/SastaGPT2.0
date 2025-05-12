import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

# Define model hyperparameters
BATCH_SIZE = 32
CONTEXT_WINDOW = 256 # Max context length
EPOCHS = 5000
CHECKPOINT_INTERVAL = 500
LR = 3e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EVAL_ITERS = 200
EMBEDDING_DIM = 512
HEADS = 8
LAYERS = 8
DROPOUT_RATE = 0.1

# --- MoE Hyperparameters ---
NUM_EXPERTS = 8
TOP_K_EXPERTS = 2

torch.manual_seed(457)

# Load dataset
try:
    with open('stoic.txt', 'r', encoding='utf-8') as file:
        corpus = file.read()
except FileNotFoundError:
    print("Error: 'stoic.txt' not found. Please create this file with your training data.")
    exit()


# Character encoding setup
char_list = sorted(list(set(corpus)))
VOCAB_SIZE = len(char_list)
char_to_index = {ch: i for i, ch in enumerate(char_list)}
index_to_char = {i: ch for i, ch in enumerate(char_list)}

encode_text = lambda s: [char_to_index[c] for c in s]
decode_text = lambda l: ''.join([index_to_char[i] for i in l])

# Train-validation split
data_tensor = torch.tensor(encode_text(corpus), dtype=torch.long)
split_idx = int(0.9 * len(data_tensor))
train_data, val_data = data_tensor[:split_idx], data_tensor[split_idx:]

# Function to generate mini-batches
def get_batch(mode):
    dataset = train_data if mode == 'train' else val_data
    max_start_index = len(dataset) - CONTEXT_WINDOW - 1
    if max_start_index <= 0:
         raise ValueError("Dataset is too small for the given CONTEXT_WINDOW.")
    idxs = torch.randint(max_start_index, (BATCH_SIZE,))
    x_batch = torch.stack([dataset[i:i + CONTEXT_WINDOW] for i in idxs])
    y_batch = torch.stack([dataset[i + 1:i + CONTEXT_WINDOW + 1] for i in idxs])
    return x_batch.to(DEVICE), y_batch.to(DEVICE)

@torch.no_grad()
def compute_loss():
    losses = {}
    model.eval()
    for mode in ['train', 'val']:
        batch_losses = torch.zeros(EVAL_ITERS)
        for i in range(EVAL_ITERS):
            x, y = get_batch(mode)
            _, loss, _ = model(x, targets=y) # Pass targets directly and unpack the 3 return values
            if loss is not None:
                 batch_losses[i] = loss.item()
            else:
                 batch_losses[i] = float('nan')
        losses[mode] = batch_losses[~torch.isnan(batch_losses)].mean()
    model.train()
    return losses


# Define attention heads with KV Cache handling
class Head(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.key = nn.Linear(EMBEDDING_DIM, head_dim, bias=False)
        self.query = nn.Linear(EMBEDDING_DIM, head_dim, bias=False)
        self.value = nn.Linear(EMBEDDING_DIM, head_dim, bias=False)
        self.dropout = nn.Dropout(DROPOUT_RATE)
        # causal mask is created dynamically in MultiHeadAttention

    def forward(self, x, mask, cache_k=None, cache_v=None):
        B, T, C = x.shape # Batch, Time (current), Channels
        k = self.key(x)   # (B, T, head_dim)
        q = self.query(x) # (B, T, head_dim)
        v = self.value(x) # (B, T, head_dim)

        # --- KV Cache Logic ---
        if cache_k is not None and cache_v is not None:
            # Concatenate past keys/values (from cache) with current key/value
            k = torch.cat([cache_k, k], dim=1) # (B, T_prev + T_curr, head_dim)
            v = torch.cat([cache_v, v], dim=1) # (B, T_prev + T_curr, head_dim)

        # Update cache with the *new* combined k, v for the next iteration
        updated_cache_k = k
        updated_cache_v = v

        # --- Attention Calculation ---
        # Use the full key/value sequence (cached + current)
        attention_scores = (q @ k.transpose(-2, -1)) * (k.shape[-1] ** -0.5) # (B, T_curr, T_prev + T_curr)

        if mask is not None:
             # Ensure mask aligns with the attention scores dimensions (query len x key len)
             # Mask shape should be (T_curr, T_prev + T_curr)
             current_mask = mask[:T, :k.size(1)]
             attention_scores = attention_scores.masked_fill(current_mask == 0, float('-inf'))

        attention_probs = F.softmax(attention_scores, dim=-1) # (B, T_curr, T_prev + T_curr)
        attention_probs = self.dropout(attention_probs)

        # Attend to the full value sequence
        out = attention_probs @ v # (B, T_curr, head_dim)

        return out, updated_cache_k, updated_cache_v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, EMBEDDING_DIM)
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.num_heads = num_heads
        # Register buffer for causal mask (reusable)
        self.register_buffer('causal_mask', torch.tril(torch.ones(CONTEXT_WINDOW, CONTEXT_WINDOW, dtype=torch.bool)).view(1, 1, CONTEXT_WINDOW, CONTEXT_WINDOW))


    def forward(self, x, kv_cache=None):
        B, T, C = x.shape
        head_outputs = []
        updated_kv_cache = [] if kv_cache is not None else None

        # Determine the causal mask based on current sequence length T
        # This mask prevents attention to future tokens within the current processing window
        mask = self.causal_mask[:, :, :T, :T].squeeze(0).squeeze(0) # Get (T, T) mask

        for i, h in enumerate(self.heads):
            cache_k, cache_v = None, None
            # --- Extract cache for this head ---
            if kv_cache is not None and kv_cache[i] is not None:
                 cache_k, cache_v = kv_cache[i]
                 # Create the appropriate mask for when cache is present
                 # It needs to be causal within the current query tokens (T x T)
                 # And allow attention to all previous key tokens (T x T_prev)
                 T_prev = cache_k.shape[1]
                 # Mask should be (T_curr, T_prev + T_curr)
                 full_mask = torch.ones(T, T_prev + T, dtype=torch.bool, device=x.device)
                 causal_part = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
                 full_mask[:, T_prev:] = causal_part # Apply causal mask only to the current part
                 mask = full_mask # Override the simple causal mask
            # --- Pass relevant part of cache and mask to the head ---
            out_h, updated_k, updated_v = h(x, mask, cache_k=cache_k, cache_v=cache_v)
            head_outputs.append(out_h)
            if updated_kv_cache is not None:
                updated_kv_cache.append((updated_k, updated_v))

        # Concatenate heads outputs
        out = torch.cat(head_outputs, dim=-1) # (B, T, num_heads * head_size)
        out = self.dropout(self.proj(out))
        return out, updated_kv_cache


# --- Mixture of Experts (Unchanged) ---

class Expert(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(DROPOUT_RATE),
        )
    def forward(self, x):
        return self.net(x)

class Gate(nn.Module):
    def __init__(self, input_dim: int, num_experts: int, top_k: int):
        super().__init__()
        self.gate_linear = nn.Linear(input_dim, num_experts, bias=False)
        self.top_k = top_k

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.gate_linear(x)
        weights = F.softmax(logits, dim=-1, dtype=torch.float)
        top_k_weights, top_k_indices = torch.topk(weights, self.top_k, dim=-1)
        top_k_weights_norm = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        return top_k_weights_norm.type_as(x), top_k_indices

class MixtureOfExperts(nn.Module):
    def __init__(self, num_experts: int, top_k: int, embedding_dim: int):
        super().__init__()
        self.gate = Gate(embedding_dim, num_experts, top_k)
        self.experts = nn.ModuleList([Expert(embedding_dim) for _ in range(num_experts)])
        self.top_k = top_k
        self.num_experts = num_experts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        x_flat = x.view(-1, C)
        top_k_weights, top_k_indices = self.gate(x_flat)
        final_output = torch.zeros_like(x_flat)
        expert_outputs_buffer = torch.zeros(x_flat.size(0), self.top_k, C, device=x.device, dtype=x.dtype)

        for i in range(self.num_experts):
            token_indices, k_pos = torch.where(top_k_indices == i)
            if token_indices.numel() > 0:
                 expert_input = x_flat[token_indices]
                 expert_output = self.experts[i](expert_input)
                 expert_outputs_buffer[token_indices, k_pos] = expert_output * top_k_weights[token_indices, k_pos].unsqueeze(1)

        final_output = expert_outputs_buffer.sum(dim=1)
        return final_output.view(B, T, C)


# --- Transformer Block (No RoPE) ---

class Block(nn.Module):
    """ Transformer block: Communication followed by Computation (MoE) """
    def __init__(self, embedding_dim, num_heads, num_experts, top_k):
        super().__init__()
        head_size = embedding_dim // num_heads
        self.mha = MultiHeadAttention(num_heads, head_size)
        self.moe = MixtureOfExperts(num_experts, top_k, embedding_dim)
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)

    def forward(self, x, kv_cache=None):
        # Multi-Head Attention part (with residual connection)
        attn_output, updated_kv_cache = self.mha(self.ln1(x), kv_cache=kv_cache) # No freqs_cis needed
        x = x + attn_output
        # Mixture of Experts part (with residual connection)
        moe_output = self.moe(self.ln2(x))
        x = x + moe_output
        return x, updated_kv_cache


# Define the language model
class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        # --- Add standard Positional Embeddings ---
        self.position_embedding_table = nn.Embedding(CONTEXT_WINDOW, EMBEDDING_DIM)
        # --- RoPE frequencies removed ---

        self.blocks = nn.ModuleList([Block(EMBEDDING_DIM, HEADS, NUM_EXPERTS, TOP_K_EXPERTS) for _ in range(LAYERS)])
        self.ln_f = nn.LayerNorm(EMBEDDING_DIM)
        self.lm_head = nn.Linear(EMBEDDING_DIM, VOCAB_SIZE)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 / (LAYERS**0.5)) # Scale std dev
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, kv_cache_list=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) # (B, T, C)

        # --- Add Positional Embeddings ---
        # Create position indices (0 to T-1)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0) # Shape (1, T)
        pos_emb = self.position_embedding_table(pos) # Shape (1, T, C)
        # Add token and position embeddings (pos_emb broadcasts across batch B)
        x = tok_emb + pos_emb
        # --- RoPE application removed ---


        # Initialize new cache list if needed
        new_kv_cache_list = [None] * LAYERS if kv_cache_list is not None else None

        # Pass through transformer blocks
        for i, block in enumerate(self.blocks):
            layer_cache = kv_cache_list[i] if kv_cache_list is not None else None
            # Pass x and layer cache (no freqs_cis)
            x, updated_layer_cache = block(x, kv_cache=layer_cache)
            if new_kv_cache_list is not None:
                 new_kv_cache_list[i] = updated_layer_cache

        x = self.ln_f(x)
        logits = self.lm_head(x) # (B, T, VocabSize)

        loss = None
        if targets is not None:
            B_logits, T_logits, C_logits = logits.shape
            logits_flat = logits.view(B_logits * T_logits, C_logits)
            targets_flat = targets.view(B_logits * T_logits)
            loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=-1)

        return logits, loss, new_kv_cache_list # Return updated cache

    @torch.no_grad() # Ensure no gradients are computed during generation
    def generate(self, idx, max_new_tokens):
        """
        Generates tokens autoregressively using the KV cache.
        idx: (B, T_initial) tensor of initial context tokens
        """
        self.eval() # Set model to evaluation mode

        kv_cache = [None] * LAYERS # List of caches, one per layer
        generated_tokens = idx

        for _ in range(max_new_tokens):
            # --- Prepare input for this step ---
            # Use only the last token if cache is present, otherwise use context
            # Crop context if it exceeds window size before feeding to model
            idx_cond = generated_tokens[:, -CONTEXT_WINDOW:]
            is_generating = kv_cache[0] is not None and kv_cache[0][0] is not None

            # If we have cache, only process the *last* token of the current sequence
            if is_generating:
                 idx_for_forward = idx_cond[:, -1:] # Shape (B, 1)
            else:
                 idx_for_forward = idx_cond # Shape (B, T_initial or CONTEXT_WINDOW)

            # --- Forward pass with the current token(s) and the cache ---
            logits, _, kv_cache = self(idx_for_forward, targets=None, kv_cache_list=kv_cache)

            # --- Cache Pruning (Important!) ---
            # Prune cache if its sequence length dimension exceeds CONTEXT_WINDOW
            if kv_cache is not None and kv_cache[0] is not None and kv_cache[0][0].shape[1] > CONTEXT_WINDOW:
                 for i in range(LAYERS):
                     if kv_cache[i] is not None:
                         k_cache, v_cache = kv_cache[i]
                         # Keep only the most recent CONTEXT_WINDOW - 1 tokens in cache
                         # This allows space for the *next* token's K/V to be added
                         kv_cache[i] = (k_cache[:, -(CONTEXT_WINDOW - 1):, :], v_cache[:, -(CONTEXT_WINDOW - 1):, :])
            # ---------------------------------

            # Get logits for the very last token prediction
            logits_last = logits[:, -1, :] # (B, VocabSize)

            # Apply softmax to get probabilities
            probs = F.softmax(logits_last, dim=-1)

            # Sample the next token index
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)

            # Append the sampled token to the sequence
            generated_tokens = torch.cat((generated_tokens, idx_next), dim=1) # (B, T+1)

        self.train() # Set model back to training mode if needed later
        return generated_tokens
# --- Training Setup ---
model = LanguageModel()
m = model.to(DEVICE)

print(f"{sum(p.numel() for p in m.parameters()) / 1e6:.2f} M parameters")

# Use AdamW optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# Training loop
print(f"Starting training on {DEVICE}...")
print(f"Vocab size: {VOCAB_SIZE}")
print(f"Context window: {CONTEXT_WINDOW}")
print(f"Embedding dim: {EMBEDDING_DIM}")
print(f"Layers: {LAYERS}, Heads: {HEADS}")
print(f"Experts: {NUM_EXPERTS}, Top-K: {TOP_K_EXPERTS}")

for epoch in range(EPOCHS):
    # Print loss periodically
    if epoch == 0 or epoch % CHECKPOINT_INTERVAL == 0 or epoch == EPOCHS - 1:
        losses = compute_loss()
        print(f"Epoch {epoch}: Train Loss {losses.get('train', float('nan')):.4f}, Val Loss {losses.get('val', float('nan')):.4f}")

        # Generate Sample Text
        m.eval()
        start_context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
        print("--- Generating Sample ---")
        generated_sequence = m.generate(start_context, max_new_tokens=100)
        generated_text = decode_text(generated_sequence[0].tolist())
        print(generated_text)
        print("-------------------------")
        m.train()

    # Get a batch of data
    xb, yb = get_batch('train')

    # Forward pass, calculate loss
    logits, loss, _ = model(xb, targets=yb, kv_cache_list=None) # No cache during training

    if loss is None:
        print(f"Warning: Loss is None at epoch {epoch}. Skipping step.")
        continue

    # Backpropagation
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("Training finished.")
torch.save(model.state_dict(), 'model_weights.pkl')

# Final generation example
print("\n--- Final Generation Example ---")
model.eval()
start_context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
generated_sequence = model.generate(start_context, max_new_tokens=500)
print(decode_text(generated_sequence[0].tolist()))
print("-----------------------------")