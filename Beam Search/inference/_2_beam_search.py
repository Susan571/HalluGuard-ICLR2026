import copy
import time
import numpy as np
from pydantic.dataclasses import dataclass

from . import *
from utils import *
from utils.logger import *
from utils.model import *
from utils.prompt_list import *
from utils.utils import *


@dataclass
class BeamSearchResult(AbstractScalingResult):
    responses: list[str]
    scores: list[float]
    selected_index: int
    steps_used: list[int]

    @property
    def the_one(self) -> str:
        return self.responses[self.selected_index]


@dataclass
class Path:
    steps: list[str]
    is_stopped: bool
    score: float

    def deepcopy(self):
        # create a deep copy of the path object
        return Path(
            steps=copy.deepcopy(self.steps),
            is_stopped=self.is_stopped,
            score=self.score,
        )


class BeamSearch(AbstractScalingAlgorithm):
    def __init__(
        self,
        sg: StepGeneration,
        prm: AbstractProcessRewardModel,
        config: Dict[str, Any],
        logger: BaseProgressLogger = DefaultProgressLogger(),
        **kwargs
    ):
        self.name = "beam_search"
        self.sg = sg
        self.prm = prm
        self.budget = config.get("budget", 32)
        self.beam_width = config.get("beam_width", 2)
        self.logger = logger

    async def _search_one_level(
        self,
        candidates: list[Path],
        prompt: str,
    ) -> list[Path]:
        is_stopped_in_the_beginning = [c.is_stopped for c in candidates]

        # collect batch inputs
        prompts, steps_so_far = [], []
        for c, is_stopped in zip(candidates, is_stopped_in_the_beginning):
            if is_stopped:
                continue

            prompts.append(prompt)
            steps_so_far.append(c.steps)

        # collect batch outputs
        # print("Prompts:", prompts)
        # print("Steps so far:", steps_so_far)
        sg_forward_results = await self.sg.generate(prompts, steps_so_far)

        # update candidates
        i = 0
        for c, is_stopped in zip(candidates, is_stopped_in_the_beginning):
            if is_stopped:
                continue

            next_step, is_stopped = sg_forward_results[i]
            c.steps.append(next_step.text)
            c.is_stopped = is_stopped
            i += 1

        # collect batch inputs for scoring
        steps_so_far = []
        for c, is_stopped in zip(candidates, is_stopped_in_the_beginning):
            if is_stopped:
                continue

            steps_so_far.append(c.steps)
        # print("Steps so far for scoring:", steps_so_far)

        # collect batch outputs for scoring
        scores = self.prm.score(
            prompt,
            [
                self.sg._post_process(steps_so_far_per_prompt, stopped=True)
                for steps_so_far_per_prompt in steps_so_far
            ],
        )

        # update candidates
        i = 0
        for c, is_stopped in zip(candidates, is_stopped_in_the_beginning):
            if is_stopped:
                continue

            c.score = scores[i]
            i += 1

        return candidates

    async def infer(
        self,
        prompt: str,
        budget: int,
        return_response_only: bool = True,
    ) -> str | BeamSearchResult:
        assert budget % self.beam_width == 0, "budget must be divisible by beam_width"
        assert budget >= self.beam_width, (
            "budget must be greater than or equal to beam_width"
        )

        num_beams = budget // self.beam_width

        candidates = [
            Path(steps=[], is_stopped=False, score=0) for _ in range(num_beams)
        ]

        while not all(c.is_stopped for c in candidates):
            candidates = await self._search_one_level(candidates, prompt)

            # get the top beam_width candidates
            candidates.sort(key=lambda x: x.score, reverse=True)
            candidates = candidates[: self.beam_width]

            # duplicate the candidates with the highest score
            new_candidates = []
            for _ in range(num_beams):
                for c in candidates:
                    new_candidates.append(c.deepcopy())
            candidates = new_candidates

        scores = [c.score for c in candidates]
        steps_used = [len(c.steps) for c in candidates]
        result = BeamSearchResult(
            responses=[
                self.sg._post_process(c.steps, stopped=True) for c in candidates
            ],
            scores=scores,
            selected_index=int(np.argmax(scores)),
            steps_used=steps_used,
        )
        return result.the_one if return_response_only else result

    async def process_question(
        self, 
        id: str,
        query: str,
        answer: str = "",
        logger=DefaultProgressLogger(),
        **kwargs
    ):
        start_time = time.time()
        
        output = await self.infer(query, budget=self.budget)
        # output, tokens = "", 0

        token_stats = {} # extract_token_stats(usage)
        score = self.evaluator(query, output, answer)
        self.logger.add_stat({
            "id": id,
            "query": query,
            "answer": answer,
            "prediction": output,
            **token_stats,
            "score": score
        })
        self.logger.update_progress({"last_question_total": round(time.time() - start_time, 2)})
