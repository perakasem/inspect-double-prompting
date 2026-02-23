from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import match
from inspect_ai.solver import (
    generate, prompt_template, system_message
)
from inspect_ai.model import GenerateConfig


def record_to_sample(record):
    DELIM = "####"
    input = record["question"]
    answer = record["answer"].split(DELIM)
    target = answer.pop().strip()
    reasoning = DELIM.join(answer)
    return Sample(
        input=input,
        target=target,
        metadata={"reasoning": reasoning.strip()}
    )


def sample_to_fewshot(sample):
    return (
        f"{sample.input}\n\nReasoning:\n"
        + f"{sample.metadata['reasoning']}\n\n"
        + f"ANSWER: {sample.target}"
    )


@task
def gsm8k(fewshot=10, fewshot_seed=42):
    strict_config = GenerateConfig(max_tokens=2000, temperature=0)

    solver = [
        system_message("Output only the numeric answer. No explanations."),
        prompt_template("Problem: {prompt}\nAnswer:"),
        generate(),
    ]

    return Task(
        dataset=hf_dataset(
            path="gsm8k", name="main", split="test", sample_fields=record_to_sample
        ),
        plan=solver,
        scorer=match(numeric=True),
        config=strict_config,
    )
