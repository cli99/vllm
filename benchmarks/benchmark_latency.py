"""Benchmark the latency of processing a single batch of requests."""
import argparse
import time

import numpy as np
import torch
from tqdm import tqdm
from utils import add_model_hooks, remove_model_hooks, set_model_stage
from vllm import LLM, SamplingParams


def main(args: argparse.Namespace):
    print(args)

    # Process all the requests in a single batch if possible.
    # NOTE(woosuk): If the request cannot be processed in a single batch,
    # the engine will automatically process the request in multiple batches.
    llm = LLM(
        model=args.model,
        tokenizer=args.tokenizer,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_seqs=args.batch_size,
        max_num_batched_tokens=args.batch_size * args.input_len,
        trust_remote_code=args.trust_remote_code,
        use_dummy_weights=args.use_dummy_weights,
        dtype="float16",
    )

    model = llm.llm_engine.workers[0].model

    add_model_hooks(model)

    sampling_params = SamplingParams(
        n=args.n,
        temperature=0.0 if args.use_beam_search else 1.0,
        top_p=1.0,
        use_beam_search=args.use_beam_search,
        ignore_eos=True,
        max_tokens=args.output_len,
    )
    print(sampling_params)
    dummy_prompt_token_ids = [[0] * args.input_len] * args.batch_size

    def run_to_completion(profile: bool = False):
        if profile:
            torch.cuda.cudart().cudaProfilerStart()

        set_model_stage(model, "prefill")

        start_time = time.time()
        llm.generate(prompt_token_ids=dummy_prompt_token_ids,
                     sampling_params=sampling_params,
                     use_tqdm=False)

        torch.cuda.synchronize()
        end_time = time.time()
        latency = end_time - start_time
        if profile:
            torch.cuda.cudart().cudaProfilerStop()
        return latency, model.__duration__


    print("Warming up...")
    try:
        run_to_completion(profile=False)
    except RuntimeError as e:
        print(e)
        exit(1)

    # Benchmark.
    total_latencies = []
    prompt_latencies = []
    for _ in tqdm(range(args.num_iters), desc="Profiling iterations"):
        t1, t2 = run_to_completion(profile=False)
        total_latencies.append(t1)
        prompt_latencies.append(t2)

    remove_model_hooks(model)

    avg_total_latency = np.mean(total_latencies)
    avg_prompt_latency = np.mean(prompt_latencies)
    if args.output_file:
        with open(args.output_file, 'a') as f:
            f.write(f'{args.batch_size}, {args.input_len}, {args.output_len}, {avg_total_latency}, {avg_prompt_latency}\n')
    print(f'Avg total latency: {avg_total_latency} seconds, Avg prompt latency: {np.mean(avg_prompt_latency)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Benchmark the latency of processing a single batch of '
                    'requests till completion.')
    parser.add_argument('--model', type=str, default='facebook/opt-125m')
    parser.add_argument('--tokenizer', type=str, default=None)
    parser.add_argument('--tensor-parallel-size', '-tp', type=int, default=1)
    parser.add_argument('--input-len', type=int, default=32)
    parser.add_argument('--output-len', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--n', type=int, default=1,
                        help='Number of generated sequences per prompt.')
    parser.add_argument('--use-beam-search', action='store_true')
    parser.add_argument('--num-iters', type=int, default=3,
                        help='Number of iterations to run.')
    parser.add_argument('--trust-remote-code', action='store_true',
                        help='trust remote code from huggingface')
    parser.add_argument('--use-dummy-weights',
                        action='store_true',
                        help='use dummy values for model weights')
    parser.add_argument('--output-file',
                        type=str, default=None,
                        help='output to file')
    args = parser.parse_args()
    main(args)
