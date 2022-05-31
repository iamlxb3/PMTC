import os
import json
from transformers import (AutoConfig, AutoModelWithLMHead)


def load_save_json(json_path, mode, verbose=1, encoding='utf-8', data=None):
    if mode == 'save':
        assert data is not None
        with open(json_path, 'w', encoding=encoding) as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            if verbose >= 1:
                print(f"save json data to {json_path}")
    elif mode == 'load':
        if os.path.isfile(json_path):
            with open(json_path, 'r') as f:
                response = json.load(f)
            if verbose >= 1:
                print(f"load json from {json_path} success")
        else:
            raise Exception(f"{json_path} does not exist!")
        return response
    else:
        raise NotImplementedError

def load_model_from_dir(model_dir):
    model_config_path = os.path.join(model_dir, 'config.json')
    model_config = AutoConfig.from_pretrained(model_config_path)
    model_bin_path = os.path.join(model_dir, 'pytorch_model.bin')
    model = AutoModelWithLMHead.from_pretrained(
        model_bin_path,
        config=model_config,
    )
    print(f"Load model form {model_dir} done!")
    print(model)
    return model
