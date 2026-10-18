import torch
import torch.nn.functional as F


def nonefficient_greedy_decoding(model, x: torch.Tensor, max_new_tokens: int = 64) -> torch.Tensor:
    """Greedy autoregressive decoding without KV caching (recomputes full sequence at each step)."""
    model.eval()
    previous_gen = x
    for _ in range(max_new_tokens):
        with torch.no_grad():
            _, logits, _, _ = model(previous_gen)
            next_token_logits = logits[:, -1, :]
            next_token_id = next_token_logits.argmax(dim=-1, keepdim=True)
            previous_gen = torch.cat([previous_gen, next_token_id], dim=1)
    return previous_gen


def greedy_decoding(
    model,
    x: torch.Tensor,
    max_new_tokens: int = 64,
    max_cache_size: int = 512,
) -> torch.Tensor:
    """
    Efficient greedy autoregressive decoding with Key-Value (KV) caching.
    
    1. Passes the initial prompt through the model to prime the KV cache.
    2. Sequentially feeds only the latest predicted token id, appending to the cache.
    """
    model.eval()
    previous_gen = x
    k_cache, v_cache = None, None

    with torch.no_grad():
        start_cache = previous_gen.size(1) - 1
        _, logits, k_cache, v_cache = model(
            previous_gen, k_cache=k_cache, v_cache=v_cache, start_cache=start_cache
        )

    for _ in range(max_new_tokens):
        next_token_logits = logits[:, -1, :]
        next_token_id = next_token_logits.argmax(dim=-1, keepdim=True)
        previous_gen = torch.cat([previous_gen, next_token_id], dim=1)
        start_cache = previous_gen.size(1) - 1

        _, logits, k_cache, v_cache = model(
            next_token_id,
            k_cache=k_cache,
            v_cache=v_cache,
            start_cache=start_cache,
        )

        if max_cache_size is not None:
            k_cache = [k[:, :, -max_cache_size:, :] for k in k_cache]
            v_cache = [v[:, :, -max_cache_size:, :] for v in v_cache]

    return previous_gen


def sampling_decoding(
    model,
    x: torch.Tensor,
    max_new_tokens: int = 64,
    max_cache_size: int = 512,
    temperature: float = 0.7,
) -> torch.Tensor:
    """
    Stochastic sampling decoding with temperature and KV caching.
    """
    if temperature <= 0:
        raise ValueError("temperature must be strictly positive")

    model.eval()
    previous_gen = x
    k_cache, v_cache = None, None

    with torch.no_grad():
        start_cache = previous_gen.size(1) - 1
        _, logits, k_cache, v_cache = model(
            previous_gen, k_cache=k_cache, v_cache=v_cache, start_cache=start_cache
        )

    for _ in range(max_new_tokens):
        next_token_logits = logits[:, -1, :]
        scaled_logits = next_token_logits / temperature
        probs = torch.softmax(scaled_logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)
        previous_gen = torch.cat([previous_gen, next_token_id], dim=1)
        start_cache = previous_gen.size(1) - 1

        _, logits, k_cache, v_cache = model(
            next_token_id,
            k_cache=k_cache,
            v_cache=v_cache,
            start_cache=start_cache,
        )

        if max_cache_size is not None:
            k_cache = [k[:, :, -max_cache_size:, :] for k in k_cache]
            v_cache = [v[:, :, -max_cache_size:, :] for v in v_cache]

    return previous_gen


def evaluate_perplexity(model, input_ids: torch.Tensor):
    """
    Computes cross-entropy loss and Perplexity (PPL = exp(loss))
    for autoregressive language modeling.
    """
    model.eval()
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    with torch.no_grad():
        _, logits, _, _ = model(input_ids)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        perplexity = torch.exp(loss)

    return loss.item(), perplexity.item()
