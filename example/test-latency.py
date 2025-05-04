#!/usr/bin/env python3

from openai import OpenAI
import sys
import time
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')


def req(index):
    base = sys.argv[1]
    client = OpenAI(
        base_url=f"http://{base}/v1",
        api_key="fke9dfkjw9rjqw94rtj29",
    )

    # long_prompt = open("example/vllm_example_query.txt").read()
    # long_prompt = str(time.time()) + "\n" + long_prompt[len(long_prompt)//4*3:]
    # print(long_prompt)
    # # long_prompt = "How much is 2+2?"
    model = "meta-llama/Llama-3.2-3B-Instruct"
    long_prompt = open("example/vllm_example_query.txt").read()*11

    logger.info(f"Thread {index}: issuing request")
    before = time.time()
    completion = client.chat.completions.create(
        model=model,
        max_completion_tokens=1,
        messages=[
            {"role": "user", "content": long_prompt},
        ],
        temperature=0,  # <-- deterministic, no creativity
    )
    after = time.time()
    logger.info("Response time: {:.3f} secs".format(after - before))


def main():
    import threading
    n = 1
    threads = []
    for i in range(0, n):
        t1 = threading.Thread(target=req, args=(i, ))
        threads.append(t1)
        t1.start()

    for thread in threads:
        thread.join()

main()
