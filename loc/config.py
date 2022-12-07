from enum import Enum


class RetrievalTypes(Enum):
    NONE        = 0 
    GF_NET      = 1 


class DetectorTypes(Enum):   
    NONE        = 0
    FAST        = 1
    SIFT        = 2 
    SURF        = 3
    ORB         = 4
    SUPERPOINT  = 5
    
    
class DescriptorTypes(Enum):
    NONE        = 0
    SIFT        = 1
    SURF        = 2
    ORB         = 3
    SUPERPOINT  = 4
   
   
class FeatureManagerConfigs(object):  
   
    GF_SUPERPOINT = dict(retrieval_type=RetrievalTypes.GF_NET,          retrieval_name="GF",
                         detector_type=DetectorTypes.SUPERPOINT,        detector_name="Superpoint",
                         descriptor_type=DescriptorTypes.SUPERPOINT,    descriptor_name="Superpoint"
                         )
    

class DataConfigs(object):
    meta = {}
    
    # Aachen = {
    #     'query': {
    #         'images':   "images/database_and_query_images/images_upright/",
    #         'cameras':  'queries/*_time_queries_with_intrinsics.txt'
    #         },
    #     'db': { 
    #         'images':   "images/database_and_query_images/images_upright/",
    #         'cameras':  None
    #         }
    #     }

    Aachen = {
        'query': {
            'images':   "images/images_upright/",
            'cameras':  'queries/*_time_queries_with_intrinsics.txt'
            },
        'db': { 
            'images':   "images/images_upright/",
            'cameras':  None
            }
        }
        
def make_data_config(name="default"): 
    
    if name=="Default":
        return DataConfigs.Default
    
    elif name=="Aachen":
        return DataConfigs.Aachen
    
    else:
        NameError
    