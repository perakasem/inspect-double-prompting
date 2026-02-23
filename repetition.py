"""Evaluation of the effect of prompt repetition on gsm8k performance."""

import re
from dotenv import load_dotenv
from inspect_ai import eval, Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import match
from inspect_ai.solver import generate, prompt_template, system_message
from inspect_ai.analysis import evals_df
from inspect_ai.model import GenerateConfig

load_dotenv()


def gsm8k_record_to_sample(record):
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

def math_record_to_sample(record):
    input_text = record["problem"]
    solution = record["solution"]

    match = re.findall(r"\\boxed\{(.*?)\}", solution)
    target = match[-1] if match else solution

    return Sample(input=input_text, target=target, metadata={"solution": solution})


def arc_record_to_sample(record):
    question = record["question"]
    choices = record["choices"]

    standard_labels = ["A", "B", "C", "D", "E"]
    choice_string = "\n".join(
        [f"({standard_labels[i]}) {text}" for i, text in enumerate(choices["text"])]
    )

    label_map = {
        str(original): standard_labels[i] for i, original in enumerate(choices["label"])
    }
    target = label_map.get(record["answerKey"], record["answerKey"])

    return Sample(
        input=f"{question}\n\nChoices:\n{choice_string}",
        target=target,
    )


MATH_PROMPT_TEMPLATE = """
{prompt}

Remember to only respond with the final answer, do not include any reasoning.

Answer:
""".strip()


ARC_PROMPT_TEMPLATE = """
{prompt}

Select the best answer from the choices above. Output ONLY the letter (A, B, C, or D).

Answer:
""".strip()


def repeat_entire_prompt(num_repetitions, prompt_template):
    return "\n\n".join(
        [f"Repetition {i+1}:\n{prompt_template}" for i in range(num_repetitions)]
    )


@task
def gsm8k(num_reps=1):
    return Task(
        dataset=hf_dataset(
            path="gsm8k",
            name="main",
            split="test",
            sample_fields=gsm8k_record_to_sample
        ),
        plan=[
            system_message("""Solve the following math problem. Your response should be of the form '$ANSWER', 
                           "where $ANSWER is the answer to the problem. Only respond with the solution, do not 
                           include any reasoning."""),
            prompt_template(repeat_entire_prompt(num_reps, MATH_PROMPT_TEMPLATE)),
            generate(),
        ],
        scorer=match(numeric=True),
        config=GenerateConfig(max_tokens=256, temperature=0),
    )


@task
def math_eval(num_reps=1):
    return Task(
        dataset=hf_dataset(
            path="HuggingFaceH4/MATH-500", split="test", sample_fields=math_record_to_sample
        ),
        plan=[
            system_message("""Solve the following math problem. Your response should be of the form '$ANSWER', 
                           "where $ANSWER is the answer to the problem. Only respond with the solution, do not 
                           include any reasoning."""),
            prompt_template(repeat_entire_prompt(num_reps, MATH_PROMPT_TEMPLATE)),
            generate(),
        ],
        scorer=match(),
        config=GenerateConfig(max_tokens=256, temperature=0),
    )


@task
def arc_challenge(num_reps=1):
    return Task(
        dataset=hf_dataset(
            "ai2_arc",
            name="ARC-Challenge",
            split="test",
            sample_fields=arc_record_to_sample,
        ),
        plan=[
            prompt_template(repeat_entire_prompt(num_reps, ARC_PROMPT_TEMPLATE)),
            generate()
        ],
        scorer=match(location="any", ignore_case=True),
        config=GenerateConfig(max_tokens=10, temperature=0)
    )


if __name__ == "__main__":
    models = [
        "openrouter/anthropic/claude-3-7-sonnet-20250219",
        "openrouter/openai/gpt-4o",
        "openrouter/google/gemini-2.0-flash-001",
    ]

    repetition_counts = [1, 2, 3]
    tasks_to_run = [gsm8k, math_eval, arc_challenge]
    log_dir = "./logs/repetition_sweep"

    for task_func in tasks_to_run:
        for reps in repetition_counts:
            print(f"\nRunning {task_func.__name__} | Reps: {reps}")
            eval(
                task_func(num_reps=reps),
                model=models,
                limit=50,
                log_dir=log_dir,
            )
