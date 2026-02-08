import argparse
import functools

from inference import *
from reward_model import *
from utils.logger import *
from utils.utils import *
from utils.model import *
from utils.reward import *


def parse_key_value(arg):
    """Parses key=value string into a (key, value) pair, converting value to int/float if needed."""
    if '=' not in arg:
        raise argparse.ArgumentTypeError(
            "Arguments must be in key=value format")
    key, value = arg.split('=', 1)
    try:
        # Try to cast to int or float
        if '.' in value:
            value = float(value)
        else:
            value = int(value)
    except ValueError:
        pass  # Keep as string if it can't be converted
    return key, value

if __name__ == "__main__":
    from dataset import *

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Evaluation dataset")
    parser.add_argument("--model", type=str, required=True, choices=MODEL_MAP.keys(), help="Model to run inference with")
    parser.add_argument("--reward", type=str, nargs="+", required=True, choices=REWARD_MAP.keys(), help="Reward model to run inference with")
    parser.add_argument("--num-workers", type=int, default=500, help="Number of workers generating the answers")
    parser.add_argument("--queue-size", type=int, default=500, help="Queue size of data loading")
    parser.add_argument("--budget", type=int, default=32, help="Test time compute budget of generations")
    parser.add_argument("--postfix", type=str, help="Postfix added to the result file name")
    parser.add_argument("--keep", action="store_true", help="Keep the progress file")
    parser.add_argument('--config', nargs='*', type=parse_key_value,
                        help="Override model config as key=value")
    args = parser.parse_args()

    config = {
        "num_workers": args.num_workers,
        "queue_size": args.queue_size,
    }
    if args.config: config.update(dict(args.config))

    progress_path = f"results/{args.model}_{args.dataset}_progress{f"_{args.postfix}" if args.postfix else ""}.json"
    result_path = f"results/{args.model}_{args.dataset}_results{f"_{args.postfix}" if args.postfix else ""}.json"
    logger = QAProgressLogger(progress_path=progress_path)
    print(logger.processed_questions)

    loader = None
    if args.dataset.lower() == 'math':
        loader = MathDatasetLoader(
            config=config, 
            logger=logger
        )
    elif args.dataset.lower() == 'qa':
        loader = QADatasetLoader(
            config=config, 
            logger=logger
        )
    elif args.dataset.lower() == 'instruction':
        loader = InstructionDatasetLoader(
            config=config, 
            logger=logger
        )
    elif args.dataset.lower() == 'aime':
        loader = AIMEDatasetLoader(
            config=config, 
            logger=logger
        )
    else:
        raise NotImplementedError("Dataset is not supported ❌")


    lm = OpenAILanguageModel(
            api_base=API_BASE, 
            api_key=API_KEY, 
            model_name=MODEL_NAME, 
            is_chat=False,
            temperature=0.1,
            max_tokens=256,
            max_concurrency=50, #config["num_workers"],
        )
    step_token = "\n\n##" if "llama" in MODEL_NAME.lower() else "\n\n"
    sg = StepGeneration(
        lm, 
        step_token,
        max_steps=16,
    )

    rm_agg_method = "model"
    # prm = LocalVllmProcessRewardModel(
    #         model_name=RW_MODEL_NAME,
    #         aggregation_method=rm_agg_method
    #     )
    if len(args.reward) == 1 and args.reward[0].lower() == "none":
        prm = None
    else:
        prm = LocalHallucinationScoreModel(
            model_types=args.reward,
            model_name=MODEL_NAME,
            reward_model=RW_MODEL_NAME,
        )

    model = MODEL_MAP[args.model](
        sg=sg,
        orm=prm,
        prm=prm,
        config=config,
        logger=logger
    )
    # model = BeamSearch(sg, prm, beam_width=4)
    
    loader.set_processor(functools.partial(model.process_question, logger=logger))

    sg.set_system_prompt(loader._system_prompt)
    sg.set_stop_token(loader._stop_token)
    model.loader = loader
    model.evaluator = loader.eval
    
    loop = always_get_an_event_loop()
    loop.run_until_complete(
        loader.run()
    )

    results = [
        {**stat, "id": int(stat["id"])}
        for stat in logger.progress_data["stats"]
    ]

    results = sorted(results, key=lambda x: x["id"])
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    tokens_sum = 0
    score_sum = 0
    for stat in results:
        score = stat['score']
        # tokens = stat['completion_tokens']
        # tokens_sum += tokens
        score_sum += score

    score = score_sum / len(results) * 100.0
    avg_tokens = tokens_sum / len(results)
    stats = {
        "len": len(results),
        "score": score,
        "llm": MODEL_NAME
    }
    results.insert(0, stats)

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    if not args.keep:
        os.remove(progress_path)

    logger.info(stats)
    logger.info(f"Done inference in {args.dataset} dataset on {args.model}_model ✅")