import os
import hashlib
import json

import logger
log = logger.get_logger(__file__, 'INFO')

def hash_config(config):
    sha256_hash = hashlib.sha256()
    temppath = '/tmp/temp_temp.json'
    json.dump(config,
              open(temppath, 'w'),
              indent=4,
              ensure_ascii=False)
    
    with open(temppath, "rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096),b""):
            sha256_hash.update(byte_block)
            
    return sha256_hash.hexdigest()

def init_config(aconfig, hpconfig):
    global config
    config = aconfig
    config['hash'] = config['hpconfig_name'] + '__' + hash_config(hpconfig)[-6:]
    os.makedirs(config['hash'], exist_ok=True)
    for k, v in config['metrics_path'].items():
        config['metrics_path'][k] = '{}/{}'.format(config['hash'], v)

    with open('{}/config.json'.format(config['hash']), 'w') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
        
    with open('{}/hpconfig.json'.format(config['hash']), 'w') as f:
        json.dump(hpconfig, f, indent=4, ensure_ascii=False)
        
