"""Write a realistic-shape prompt dataset for speculative-decoding benchmarks.

`vllm bench throughput --dataset-name random` synthesizes prompts as an
arithmetic ramp over the vocabulary, which no draft model can predict: measured
DFlash2 acceptance was 1.4-3.2% purely because of the workload, making the
speculative numbers meaningless.

This writes a JSONL file of natural-language prompts with requested output
lengths, in the format `--dataset-name hf` expects, so draft acceptance reflects
real text. Prompts vary by topic/instruction to avoid trivially duplicated
prefixes (which would flatter the drafter via prefix caching).
"""

import argparse
import json
import random

TOPICS = [
    "the water cycle",
    "how vaccines work",
    "the history of the printing press",
    "why tides follow the moon",
    "how a compiler turns code into machine instructions",
    "the difference between TCP and UDP",
    "how plants convert light into sugar",
    "the causes of the 2008 financial crisis",
    "how GPS satellites determine position",
    "the structure of a sonnet",
    "how antibiotics interact with bacteria",
    "the basics of supply and demand",
    "why the sky changes colour at sunset",
    "how neural networks learn from data",
    "the role of enzymes in digestion",
    "how a refrigerator moves heat",
    "the origins of the calendar",
    "how noise-cancelling headphones work",
    "the physics of a curveball",
    "how elections are administered",
]

TASKS = [
    "Write a clear, detailed explanation of {topic} for an interested adult.",
    "Explain {topic} step by step, including the key principles involved.",
    "Describe {topic} in a structured way, with examples where useful.",
    "Summarize the most important points about {topic} and why they matter.",
    "Walk through {topic}, covering both the basics and one subtle detail.",
]

COMPLETION_PLACEHOLDER = (
    "Here is a careful explanation covering the key ideas, how they relate, "
    "and a concrete example that makes the mechanism easy to follow."
)

OPENERS = [
    "I have been reading about {topic} and would like a careful overview.",
    "Could you explain {topic}? I understand the basics but want more depth.",
    "I need a solid briefing on {topic} for a project I am working on.",
    "Please help me understand {topic} properly, not just at a high level.",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-prompts", type=int, default=64)
    ap.add_argument("--output-len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = []
    for i in range(args.num_prompts):
        topic = TOPICS[i % len(TOPICS)]
        task = rng.choice(TASKS).format(topic=topic)
        opener = rng.choice(OPENERS).format(topic=topic)
        prompt = f"{opener}\n\n{task}\n\nAim for roughly {args.output_len} tokens of substantive prose."
        rows.append(
            {
                "prompt": prompt,
                "output_tokens": args.output_len,
                # Distinct per row so prefix caching cannot mask drafter quality.
                "id": f"bench-{i:04d}",
            }
        )

    # ShareGPT layout: `vllm bench --dataset-name sharegpt --dataset-path` reads a
    # JSON list of entries with >= 2 conversation turns and uses turn 0 as the
    # prompt. The completion text is only used for length calibration, so the
    # prompts stay natural language while output length comes from --output-len.
    data = [
        {
            "id": row["id"],
            "conversations": [
                {"from": "human", "value": row["prompt"]},
                # Non-empty: the sampler drops turns it cannot tokenize, and the
                # completion text only calibrates length here (output length is
                # pinned by --output-len).
                {"from": "gpt", "value": COMPLETION_PLACEHOLDER},
            ],
        }
        for row in rows
    ]
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(data, f)
    print(f"wrote {len(data)} prompts to {args.out} (sharegpt format)")


if __name__ == "__main__":
    main()
