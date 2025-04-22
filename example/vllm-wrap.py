#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import sys
import os

print("Loading...")

from vua.core import VUA, VUAConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
import vllm_kv_connector_0
from vllm.entrypoints.cli.main import main

KVConnectorFactory.register_connector("RemoteStorageConnector",
                                      "vllm_kv_connector_0",
                                      "RemoteStorageConnector")

kvcache_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "vua.tmp")

try:
    os.mkdir(kvcache_path)
except OSError:
    pass

print("VUA path", kvcache_path)
vllm_kv_connector_0.vuacache = VUA(VUAConfig, kvcache_path)

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
    sys.exit(main())
