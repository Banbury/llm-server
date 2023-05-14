import os
import requests
from typing import Any, List, Mapping, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

PORT=8000
if os.getenv("PORT"):
    PORT=os.environ.get("PORT")

class LocalLLM(LLM):
    host = f'localhost:{PORT}'
    uri = f'http://{host}/api/generate'

    options = {
        'max_new_tokens': 500,
        'temperature': 0.01,
        # 'do_sample': True,
        # 'top_p': 0.5,
        # 'typical_p': 1,
        # 'repetition_penalty': 1.2,
        # 'top_k': 40,
        # 'min_length': 0,
        # 'no_repeat_ngram_size': 0,
        # 'num_beams': 1,
        # 'penalty_alpha': 0,
        # 'length_penalty': 1,
        # 'early_stopping': False,
        # 'seed': -1,
        # 'add_bos_token': True,
        # 'truncation_length': 2048,
        # 'ban_eos_token': False,
        # 'skip_special_tokens': True,
        # 'stopping_strings': []
    }
        
    @property
    def _llm_type(self) -> str:
        return "custom"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        self.options['prompt'] = prompt
        # if stop:
        #     self.options['stopping_strings'].extend(stop)
        response = requests.post(self.uri, json={ "prompt": prompt, "options": self.options })
        if response.status_code == 200:
            s = response.json()['result']
            return s
        return ""
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return self.options
