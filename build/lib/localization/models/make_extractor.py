from asyncio.log import logger
from os import makedirs, path
from tqdm import tqdm
import shutil

import numpy as np

import  torch 
import  torch.nn as nn 
from    torch.utils.data import DataLoader, SubsetRandomSampler

import core.backbones as body


def make_extractor(args, config):
  

    # parse params with default values
    body_config     = config["body"]
    global_config   = config["global"]
    local_config    = config["local"]
    
    # Create backbone
    print("Creating extractor model %s", body_config.get("arch"))
    print(body.__dict__)
    # body_fn = body.__dict__[body_config.get("arch")]
    # body_params = body_config.getstruct("body_params") if body_config.get("body_params") else {}

    # if body_config.getboolean("pretrained"):
    #     arch = body_config.get("arch")

    #     # vgg with bn or without
    #     if body_config.get("arch").startswith("vgg"):
    #         if body_config["normalization_mode"] != 'off':
    #             arch = body_config.get("arch") + '_bn'

    #     # Download pre trained model
    #     log_debug("Downloading pre - trained model weights %s", body_config.get("source_url"))

    #     if body_config.get("source_url") == "cvut":
            
    #         if body_config.get("arch") not in model_urls_cvut:
    #             raise ValueError(" body arch not found in cvut witch  source_url = pytorch")
            
    #         state_dict = load_state_dict_from_url(model_urls_cvut[arch])


    #         converted_model = body.convert_cvut(state_dict)

    #     elif body_config.get("source_url") == "pytorch":
    #         if body_config.get("arch") not in model_urls:
    #             raise ValueError(" body arch not found in pytorch ")
            
    #         state_dict = load_state_dict_from_url(model_urls[arch], progress=True)
   
    #         converted_model = body.convert(state_dict)
      
    #     else:
    #         raise ValueError(" try source_url = cvut  or pytorch  ")

    #     folder = args.directory + "ImageNet"

    #     if not path.exists(folder):
    #         log_debug("Create path to save pretrained backbones: %s ", folder)
    #         makedirs(folder)

    #     body_path = folder + "/" + arch + ".pth"
    #     log_debug("Saving pretrained backbones in : %s ", body_path)
    #     torch.save(converted_model, body_path)

    #     # Load  converted weights to model
    #     body.load_state_dict(torch.load(body_path, map_location="cpu"))

    #     # Freeze modules in backbone
    #     for n, m in body.named_modules():
    #         for mod_id in range(1, body_config.getint("num_frozen") + 1):
    #             if ("mod%d" % mod_id) in n:
    #                 freeze_params(m)

    # else:
    #     log_info("Initialize body to train from scratch")
    #     init_weights(body, body_config)
        
    # # Head
    # global_head = globalHead(   inp_dim=global_config.getint("inp_dim"),
    #                             global_dim=global_config.getint("global_dim"),
    #                             local_dim=global_config.getint("local_dim"),
                                
    #                             pooling=global_config.getstruct("pooling"),
    #                             do_withening=global_config.getboolean("whithening"),
                                
    #                             layer=global_config.get("type"),
    #                             norm_act=norm_act_static
    #                             )
    # # freeze head 
    # if global_config.getboolean("freeze"):
    #     log_debug("Freeze withen paramters  : %s ", body_path)

    #     for p in global_head.whiten.parameters():
    #         p.requires_grad = False
        
    #     for p in global_head.local_whiten.parameters():
    #         p.requires_grad = False
    

    # # Create a generic image retrieval network
    # net = ImageRetrievalNet(body, global_head, 
    #                         num_features=local_config.getint("num_features"))  
    
    # # Compute PCA
    # layers = ["global"]
    # compute_PCA_layer(net, train_dataloader, layers, args, config, varargs) 

       
    return 
    