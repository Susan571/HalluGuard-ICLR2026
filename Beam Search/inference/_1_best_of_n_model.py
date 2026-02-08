import time
from pydantic.dataclasses import dataclass

from . import *
from utils import *
from utils.logger import *
from utils.model import *
from utils.prompt_list import *
from utils.utils import *


@dataclass
class BestOfNResult(AbstractScalingResult):
    responses: list[str]
    scores: list[float]
    selected_index: int

    @property
    def the_one(self) -> str:
        return self.responses[self.selected_index]


class BestOfN_Model:
    def __init__(
        self, 
        sg: StepGeneration,
        orm: AbstractOutcomeRewardModel,
        config: Dict[str, Any],
        logger: BaseProgressLogger = DefaultProgressLogger(),
        **kwargs
    ):
        self.name = "best_of_n"
        self.sg = sg
        self.orm = orm
        self.budget = config.get("budget", 32)
        self.logger = logger

    async def infer(
        self,
        prompt: str,
        budget: int,
        return_response_only: bool = True,
    ) -> str | BestOfNResult:
        # generate responses
        responses = await self.sg.lm.generate(
            [[{"role": "user", "content": prompt}] for _ in range(budget)],
            max_tokens=2048,
        )

        # score responses
        # TODO: make batched a configurable parameter or remove non-batched branch
        # Currently hardcoded to True, will be addressed in future PR
        batched = True
        if batched:
            scores = self.orm.score(
                prompt, 
                [response.text for response in responses]
            )
        else:
            scores = []
            for r in responses:
                scores.append(self.orm.score(prompt, r))

        # select the best response
        selected_index = scores.index(max(scores))
        print(len(scores), selected_index, scores)

        # return the result
        result = BestOfNResult(
            responses=[response.text for response in responses],
            scores=scores,
            selected_index=selected_index,
        )
        return result.the_one if return_response_only else result
    
    async def process_question(
        self, 
        id: str,
        query: str,
        steps: List[str],
        credits: List[int],
        reference: str = "",
        solution: str = "",
        answer: str = "",
        level: int = None,
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