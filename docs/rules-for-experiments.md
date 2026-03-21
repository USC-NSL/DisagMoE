Every experiment (every single benchmark client launch) need to have a unique experiment id, for disagmoe, it should be amoe-<idnumber>, for sglang, it should be sgl-<idnumber>

All the log files, metric logging, stats, should be aggregated at the head node's <PROJECT_ROOT>/experiments dir, under a subdir named by the experiment id.
Whenever you create an experiment dir, you should also add a intention.txt to briefly describe the exp and what's the intention, using less than 60 words.
Those individual experiments dir should be ignored by git. Generated plots should be placed here.
The python plotting scripts associated with it should all be inside <PROJECT_ROOT>/experiments as well, these should not be ignored by git.

When agents running experiments that are long to run, should check for completion/errors at least every 10 minutes.

Also, before each experiment, it's recommended to clear the pycache on each node/nfs. Sometimes there are stale bytecode in the FS. 