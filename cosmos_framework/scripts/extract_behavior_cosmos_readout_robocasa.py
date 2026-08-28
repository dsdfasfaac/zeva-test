#!/usr/bin/env python3
"""Extract clean image+instruction Cosmos readouts for RoboCasa Stage 3."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

from cosmos_framework.data.generator.action.json_formatter import ActionPromptJsonFormatter
from cosmos_framework.data.generator.sequence_packing import SequencePlan
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean
from cosmos_framework.scripts.action_policy_server_robocasa365 import (
    RobolabPolicyService,
    RobolabServerArgs,
    _CONCAT_VIEW_DESCRIPTION,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--experiment", default="action_policy_robocasa365_atomic5_cosmos_behavior_stage2")
    parser.add_argument("--wan-vae-path", required=True)
    parser.add_argument("--qwen-vlm-path", required=True)
    parser.add_argument("--first-latent-shard", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optionally extract only the first N samples (for a disposable smoke test).",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="First sample index to extract; combines with --max-samples for sharded extraction.",
    )
    return parser.parse_args()


def format_action_prompt(instruction: str) -> str:
    sample = {
        "ai_caption": instruction.strip(),
        "video": torch.empty((3, 33, 1, 1), dtype=torch.uint8),
        "conditioning_fps": torch.tensor(20.0),
        "image_size": torch.tensor([480, 832, 480, 832]),
        "viewpoint": "concat_view",
        "additional_view_description": _CONCAT_VIEW_DESCRIPTION,
        "mode": "wam",
    }
    formatted = ActionPromptJsonFormatter(caption_key="ai_caption")(sample)["ai_caption"]
    return json.dumps(formatted) if isinstance(formatted, dict) else str(formatted)


@torch.inference_mode()
def extract_batch(model, latent_batch: torch.Tensor, prompts: list[str]) -> torch.Tensor:
    batch_size = len(prompts)
    # No action sample means the Stage-2 packer reserves no oracle behavior
    # prefix. This is the clean VLM pass used by BehaviorVLA Stage 3.
    plans = [SequencePlan(has_text=True, has_vision=True, condition_frame_indexes_vision=[0]) for _ in prompts]
    token_ids = model._tokenize_captions(
        prompts,
        use_system_prompt=bool(model.vlm_config.use_system_prompt),
        system_prompt=None,
        is_video=False,
    )
    clean = GenerationDataClean(
        batch_size=batch_size,
        is_image_batch=True,
        x0_tokens_vision=[latent_batch[i].unsqueeze(0).unsqueeze(2) for i in range(batch_size)],
        fps_vision=torch.full((batch_size,), 20.0),
    )
    packed = model._pack_input_sequence(
        plans,
        token_ids,
        clean,
        torch.zeros(batch_size, dtype=torch.float32),
        include_end_of_generation_token=model._derive_include_end_of_generation_token(),
    )
    if packed.behavior_indexes.numel() != 0:
        raise RuntimeError("Clean Stage-3 extraction unexpectedly contains behavior prefix tokens")
    packed.to_cuda()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model.net(packed_seq=packed)
    hidden = output["last_hidden_state"]
    indexes = packed.vision.sequence_indexes
    readouts = []
    sample_start = 0
    for sample_length in packed.sample_lens:
        sample_end = sample_start + sample_length
        sample_indexes = indexes[(indexes >= sample_start) & (indexes < sample_end)]
        readouts.append(hidden.index_select(0, sample_indexes).mean(dim=0))
        sample_start = sample_end
    return torch.stack(readouts).float().cpu()


def main() -> None:
    args = parse_args()
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_main = rank == 0
    shard = torch.load(args.first_latent_shard, map_location="cpu", weights_only=True, mmap=True)
    if shard.get("format") != "cosmos_behavior_first_latent_v1":
        raise ValueError(f"Unsupported shard: {args.first_latent_shard}")
    total_count = len(shard["instructions"])
    if args.start_index < 0 or args.start_index >= total_count:
        raise ValueError(f"--start-index must be in [0, {total_count}), got {args.start_index}")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    stop_index = total_count
    if args.max_samples is not None:
        stop_index = min(total_count, args.start_index + args.max_samples)
    count = stop_index - args.start_index
    prompts = [format_action_prompt(str(text)) for text in shard["instructions"][args.start_index:stop_index]]
    service = RobolabPolicyService(RobolabServerArgs(
        checkpoint_path=args.checkpoint_path,
        allow_dcp_checkpoint=True,
        experiment=args.experiment,
        experiment_overrides=[
            f"model.config.tokenizer.vae_path={args.wan_vae_path}",
            f"model.config.vlm_config.tokenizer.pretrained_model_name={args.qwen_vlm_path}",
            "model.config.tokenizer.encode_exact_durations=[33]",
        ],
        output_dir=args.runtime_output / f"rank{rank}",
        deterministic_seed=True,
        seed=0,
    ))
    model = service.model
    model.eval()
    readouts: torch.Tensor | None = None
    started = time.monotonic()
    for batch_index, start in enumerate(range(0, count, args.batch_size), 1):
        end = min(count, start + args.batch_size)
        absolute_start = args.start_index + start
        absolute_end = args.start_index + end
        result = extract_batch(model, shard["latents"][absolute_start:absolute_end].contiguous(), prompts[start:end])
        if readouts is None:
            readouts = torch.empty((count, result.shape[-1]), dtype=torch.bfloat16)
        readouts[start:end].copy_(result.to(torch.bfloat16))
        if args.log_every and (batch_index % args.log_every == 0 or end == count):
            rate = end / max(time.monotonic() - started, 1e-6)
            if is_main:
                print(f"readouts={end}/{count} rate={rate:.2f}_samples_s", flush=True)
    if not is_main:
        # Parameter sharding requires every rank to execute the same forwards,
        # but only rank zero owns the final readouts artifact.
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        return
    assert readouts is not None
    payload = {
        "format": "cosmos_behavior_cosmos_readout_v1",
        "checkpoint_path": args.checkpoint_path,
        "source_first_latent_shard": str(args.first_latent_shard),
        "pooling": "mean_clean_initial_vision_tokens_after_text_conditioning",
        "prompt_format": "robocasa365_action_json_480x832_20hz",
        "oracle_behavior_prefix_tokens": 0,
        "readouts": readouts,
        "source_indices": shard["source_indices"][args.start_index:stop_index].clone(),
        "episode_ids": shard["episode_ids"][args.start_index:stop_index].clone(),
        "instructions": shard["instructions"][args.start_index:stop_index],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, args.output)
    print(
        f"wrote={args.output} samples={count} shape={tuple(readouts.shape)} "
        f"world_size={world_size}",
        flush=True,
    )
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    main()
