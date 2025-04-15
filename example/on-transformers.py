#!/usr/bin/env python3

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import os
import time
import sys
import argparse
import logging
from vua.core import VUA, VUAConfig
from transformers.cache_utils import DynamicCache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-stored-kvcache", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--stop-after-prefix", action="store_true")
    args = parser.parse_args()

    kvcache_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vua.tmp")

    try: os.mkdir(kvcache_path)
    except OSError: pass

    print("VUA path", kvcache_path)
    cache = VUA(VUAConfig, kvcache_path)

    # Load a larger model using quantization to fit within ~20GB GPU memory
    model_name = "mistralai/Mistral-7B-v0.1"
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.float16
    )
    device = 'cuda'

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map='auto',
        quantization_config=quantization_config,
    )

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s.%(msecs)03d [%(levelname)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # Prepare prefix
    prefix = open("samples/Makefile.inc1").read()[:16000]
    prefix_ids = tokenizer(prefix, return_tensors="pt").input_ids.cuda()
    print(f"prefix_ids: {prefix_ids.size()}")

    # Either compute or load past_key_values from VUA for the prefix
    past_key_values = None
    with torch.no_grad():
        start_time = time.time()
        if args.no_cache:
            print("Not using cache at all, computing instead")
            outputs = model(prefix_ids, use_cache=True)
            past_key_values = outputs.past_key_values
        else:
            prefix_trimmed = VUAConfig.trim_to_split_factor(prefix_ids)
            res = None
            if not args.skip_stored_kvcache:
                res = cache.get_closest(prefix_trimmed, device)
                print("Timing after get_closest: ", time.time() - start_time)
            if not res:
                print("Not using stored kvcache, computing instead")
                prefix_trimmed_sq = prefix_trimmed.unsqueeze(0)
                if len(prefix_trimmed_sq[0]) > 0:
                    print(f"Computing for trimmed prefix {prefix_trimmed_sq.size()}")
                    outputs = model(prefix_trimmed_sq, use_cache=True)
                    past_key_values_cache = outputs.past_key_values.to_legacy_cache()
                    print(f"Model returns past_key_values_cache {past_key_values_cache[0][0].size()}")
                    print("Prefix compute at:", time.time() - start_time)
                    cache.put(prefix_trimmed, past_key_values_cache)
                    past_key_values = outputs.past_key_values
                    print("Prefix save at:", time.time() - start_time)
                remaining = prefix_ids[:, len(prefix_trimmed_sq[0]):]
            else:
                print("Using kvcache from storage")
                closest_prefix = res.tokens
                # we need to make up for the difference
                past_key_values = DynamicCache.from_legacy_cache(res.data)
                print("Timing after DynamicCache.from_legacy_cache: ", time.time() - start_time)
                remaining = prefix_ids[:, len(closest_prefix):]

            if len(remaining[0]) > 0:
                print(f"Computing for remaining prefix {remaining.size()}")
                outputs = model(remaining, past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values

        end_time = time.time()
        print("Prefix load/compute took:", end_time - start_time)

    if args.stop_after_prefix:
        return

    # Generate additional tokens manually using cached past_key_values
    max_new_tokens = 1000
    generated = prefix_ids

    print(prefix, end='')
    all_printed = prefix

    for i in range(max_new_tokens):
        input_ids = generated[:, -1].unsqueeze(-1)

        with torch.no_grad():
            outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values

        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(-1)
        generated = torch.cat([generated, next_token], dim=-1)

        full_ids = torch.cat([generated], dim=-1)
        full_text = tokenizer.decode(full_ids[0], skip_special_tokens=True)

        # Print extra text
        print(f"\033[1m{full_text[len(all_printed):]}\033[0m", end='')
        sys.stdout.flush()
        all_printed = full_text

main()
