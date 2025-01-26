#!/usr/bin/env python

# Copyright 2023 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================


from tensorflow.python.framework.group_embedding_types import (
    DistStrategy,
    set_group_lookup_strategy,
)

import os
import contextlib
from tensorflow_estimator.python.estimator import estimator as _estimator_lib


class CollectiveStrategy:
    r"""
    A thin interface to all kinds of Synchonized training strategy.
    """

    def __init__(self):
        self._hvd = None
        self._hb = None
        strategy = os.getenv("COLLECTIVE_STRATEGY", "sok")
        if strategy == DistStrategy.SOK.value:
            try:
                import horovod.tensorflow as hvd
                hvd.init()
                from sparse_operation_kit import experiment as sok
                sok.init()
            except:
                raise ImportError(
                    "While param `strategy` in enable_distributed_strategyis given `sok`,"
                    " sok module initialize error,please double check"
                )

            self._sok = sok
            self._hvd = hvd
            set_group_lookup_strategy(strategy)
        elif strategy == DistStrategy.HB.value:
            try:
                import hybridbackend.tensorflow as hb
            except:
                raise ImportError(
                    "While param `strategy` in enable_distributed_strategyis given `hb`, hb module initialize error,please double check"
                )
            self._hb = hb
            set_group_lookup_strategy(strategy)
        else:
            raise ValueError(
                "accepted `COLLECTIVE_STRATEGY` is sok or hb, while given %s", strategy
            )

    @contextlib.contextmanager
    def scope(self, *args, **kwargs):
        if self._hvd:
            from tensorflow.python.distribute import hvd_strategy
            with hvd_strategy.scope() as context:
                yield context
        elif self._hb:
            with self._hb.scope() as context:
                yield context

    @contextlib.contextmanager
    def embedding_scope(self, **kwargs):
        if self._hvd:
            from tensorflow.python.distribute import hvd_strategy
            with hvd_strategy.embedding_scope() as context:
                yield context
        elif self._hb:
            with self._hb.embedding_scope() as context:
                yield context

    def world_size(self):
        if self._hvd:
            return self._hvd.size()
        elif self._hb:
            return self._hb.context.world_size

    def rank(self):
        if self._hvd:
            return self._hvd.rank()
        elif self._hb:
            return self._hb.context.rank

    def estimator(self, model_fn, **kwargs):
        if self._hvd:
            from tensorflow.python.distribute.hvd_strategy import wraps_estimator
            _estimator = wraps_estimator(_estimator_lib.Estimator)
        elif self._hb:
            _estimator = self._hb.estimator.Estimator
        
        return _estimator(model_fn, **kwargs)

    def export_saved_model(
        self,
        savedmodel_dir,
        checkpoint_dir=None,
        signature_def_fn=None,
        assets_extra=None,
        as_text=False,
        clear_devices=True,
        strip_default_attrs=True,
    ):
        if self._hvd:
            from tensorflow.python.distribute import hvd_strategy
            hvd_strategy.export(
                savedmodel_dir,
                checkpoint_dir,
                signature_def_fn,
                assets_extra,
                as_text,
                clear_devices,
                strip_default_attrs,
            )
        elif self._hb:
            self._hb.train.export(
                savedmodel_dir,
                checkpoint_dir,
                signature_def_fn,
                assets_extra,
                as_text,
                clear_devices,
                strip_default_attrs,
            )
