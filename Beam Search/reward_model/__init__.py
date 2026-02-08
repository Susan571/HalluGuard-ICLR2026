from reward_model.perplexity_reward import PerplexityScoreModel
from reward_model.energy_reward import EnergyScoreModel
from reward_model.eigenscore_reward import EigenScoreModel
from reward_model.mind_reward import MindScoreModel
from reward_model.entropy_reward import EntropyScoreModel
from reward_model.semantic_entropy_reward import SemanticEntropyScoreModel
from reward_model.selfcheckgpt_reward import SelfCheckGPTScoreModel
from reward_model.race_reward import RaceScoreModel
from reward_model.ptrue_reward import PTrueScoreModel
from reward_model.factscore_reward import FactScoreModel
from reward_model.ntk_reward import NTKS3ScoreModel
from reward_model.rm_reward import VllmProcessRewardModel

REWARD_MAP = {
    "none": None,
    "perplexity": PerplexityScoreModel,
    "energy": EnergyScoreModel,
    "entropy": EntropyScoreModel,
    "eigenscore": EigenScoreModel,
    "mind": MindScoreModel,
    "semantic_entropy": SemanticEntropyScoreModel,
    "selfcheckgpt": SelfCheckGPTScoreModel,
    "race": RaceScoreModel,
    "ptrue": PTrueScoreModel,
    "factscore": FactScoreModel,
    "ntk": NTKS3ScoreModel,
    "rm": VllmProcessRewardModel,
}
