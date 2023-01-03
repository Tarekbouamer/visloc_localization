from enum import Enum


class DataConfigs(object):
    meta = {}
    
    Aachen = {
        'query': {
            'images':   "images_upright/",
            'cameras':  'queries/*_time_queries_with_intrinsics.txt'
            },
        'db': { 
            'images':   "images_upright/",
            'cameras':  None
            }
        }
        
def make_data_config(name="default"): 
    
    if name=="default":
        return DataConfigs.Default
    
    elif name=="aachen":
        return DataConfigs.Aachen
    
    else:
        NameError
    