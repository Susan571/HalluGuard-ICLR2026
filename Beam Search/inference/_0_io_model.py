import time

from . import *
from utils import *
from utils.logger import *
from utils.model import *
from utils.prompt_list import *
from utils.utils import *

class IO_Model:
    def __init__(
        self,
        sg: StepGeneration,
        prm: AbstractProcessRewardModel,
        config: Dict[str, Any],
        logger: BaseProgressLogger = DefaultProgressLogger(),
        **kwargs
    ):
        self.name = "io"
        self.sg = sg
        self.prm = prm
        self.logger = logger

    # @llm_retry(max_retries=10, default_output=("", 0))
    async def io_reasoning(self, query: Query):
        # Given a query and its solving route, ask LLM to solve it by using the given budget.
        user_message = query.description

        responses = await self.sg.lm.generate(
            [
                [{"role": "user", "content": user_message}]
            ], 
            temperatures=[self.sg.temperature],
            max_tokens=2048
        )
        output = responses[0].text
        usage = responses[0].usage

        self.logger.debug(str(self.sg.lm._prepare_messages([{"role": "user", "content": user_message}])) + "\n" + output)
        
        return output, usage

    async def process_question(
        self, 
        id: str,
        query: str,
        reference: str = "",
        answer: str = "",
        level: int = None,
        logger=DefaultProgressLogger(),
        **kwargs
    ):
        start_time = time.time()
        
        q = Query(
            description=query,
            reference=reference,
            answer=answer,
            level=level
        )
        output, usage = await self.io_reasoning(q)
        # output, tokens = "", 0

        token_stats = extract_token_stats(usage)
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
