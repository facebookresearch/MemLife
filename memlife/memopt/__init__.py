"""MemOpt: reinforcement learning for the MemLife writer, against the FIRM reward.

Three dimensions, each approximating one term of the information decomposition in the paper:

``faithfulness``    is every token of this memory supported by its source segment?
                    Token-level, from a teacher-forced minimal-correction audit.
``informativeness`` does this memory still carry the answer-relevant evidence its segment
                    supplied? Entailment of a source-grounded key fact.
``retrievability``  would this memory actually be returned at question time? Replay of the
                    reader's own cached actions.

They are combined **multiplicatively** -- ``x[i,t] = R_faith[i,t] * R_inf[i] * R_ret[i]`` -- so
a token earns credit only when it is faithful *and* its memory is informative *and*
retrievable. :mod:`memlife.memopt.advantage` turns that into a token-level group-relative
advantage; :mod:`memlife.memopt.reward` is what the trainer calls per rollout.

The pipeline, in the order it runs::

    1. memlife.memory.build          a memory bank with the current writer
    2. memlife.memopt.keyfacts       Q(c) and k_{c,q}  -- once, independent of the writer
    3. memlife.memopt.trace          cache the reader's actions -- once per epoch
    4. memlife.memopt.data           the rollout dataset
    5. memlife.memopt.train          GRPO
"""
