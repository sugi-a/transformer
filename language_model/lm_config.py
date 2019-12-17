import os, json
"""
Configuration for a model.

This file is loaded by python3
`exec(this_file_content, None, {'model_dir': dirname_of_this_file})`

This file must contain:
    params : a dict storing configuration of Transformer and training.
This file can contain:
    IDs2text: method to convert token IDs in the target language into sentences.
        Args:
            IDs: list of list of int
        Returns:
            List of str. Just replace indivisual IDs by tokens. (Do not spm_decode)
    validation_metric: validation method
"""


# -------- `params` --------
# By default, automatically built from `model_config.json`

with open(model_dir + '/' + 'lm_config.json') as f:
    params = json.load(f)

# Add prefix to the dataset paths. Absolute prefix is recommended.
if 'basedir' in params: 
    _p = params["basedir"]
    for i in range(len(params["train"]["data"]["train"])):
        params["train"]["data"]["train"][i] = os.path.join(_p, params["train"]["data"]["train"][i])
    params["train"]["data"]["dev"] = os.path.join(_p, params["train"]["data"]["dev"])
    params["vocab"]["dict"] = os.path.join(_p, params["vocab"]["source_dict"])


# -------- methods --------
# Following two functions can be customly defined
"""
IDs2text(IDs)
validation_metric(global_step, inference)
"""
