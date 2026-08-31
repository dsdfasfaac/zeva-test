# Zeva checkpoint compatibility

The public source tree and APIs use the paper terminology: CTE, BIT, PIM,
Phase-Conditioned Retrieval, Causal Prompt, and Policy Injection.

Released DCP policy checkpoints still contain a small set of historical
`behavior_*` parameter prefixes. Those strings are a serialized weight schema,
not public module names. Renaming registered PyTorch attributes would prevent
distributed checkpoints from resolving their tensors, so the policy network
keeps those storage keys and exposes paper-named properties in code.

The released CTE checkpoint uses two historical key fragments. The loader calls
`remap_legacy_cte_state_dict()` to map them to `interaction_state_token` and
`interaction_state` before strict loading. The mapping changes keys only and
does not copy, reshape, or modify tensor values.

All new code should import from `cosmos_framework.model.zeva`. The former
`cosmos_framework.model.behavior` package is intentionally not shipped.
