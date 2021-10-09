#!/bin/bash
# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.
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

set -eo pipefail

PODNAME=$1
DISTDIR=$2

mkdir -p ${DISTDIR}
tar -czf ${DISTDIR}/archive.tar.gz .
kubectl wait --for=condition=ready pod ${PODNAME}
kubectl cp ${DISTDIR}/archive.tar.gz ${PODNAME}:/workspace/archive.tar.gz
kubectl exec -it ${PODNAME} -- tar -xzf /workspace/archive.tar.gz -C /workspace

