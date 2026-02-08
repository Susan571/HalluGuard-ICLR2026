COMPLETION_LLM = {

}
CHAT_LLM = {
    "meta-llama/Llama-3.2-1B-Instruct",
}
REASONING_LLM = {

}

from inference._0_io_model import IO_Model
from inference._1_best_of_n_model import BestOfN_Model
from inference._2_beam_search import BeamSearch

MODEL_MAP = {
    "io": IO_Model,
    "best_of_n": BestOfN_Model,
    "beam_search": BeamSearch,
}
