# logger
import logging
from functools import partial
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Any, Dict, List, Tuple, Union

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from loc.matchers import BaseMatcher, MutualNearestNeighbor, SuperGlueMatcher
from loc.utils.io import pairs2key, read_pairs_list, remove_duplicate_pairs
from loc.utils.writers import MatchesWriter
from loc.utils.readers import LocalFeaturesLoader

logger = logging.getLogger("loc")


def make_matcher(cfg):

    matcher_name = cfg.matcher.name

    if matcher_name == "nn":
        return MutualNearestNeighbor(cfg.matcher)
    elif matcher_name == "superglue":
        return SuperGlueMatcher(cfg.matcher)
    else:
        raise KeyError(matcher_name)


class Matcher(BaseMatcher):

    def __init__(self,
                 cfg: Dict = {}
                 ) -> None:
        super().__init__(cfg=cfg)

        # call from a factory
        self.matcher = make_matcher(cfg=cfg)

        # init
        self._set_device()
        self._eval()
        self._load_weights()

        #
        logger.info(f"init {cfg.matcher.name} matcher")

    def _prepare_inputs(self,
                        data: Dict[str, Union[torch.Tensor, List, Tuple]]
                        ) -> Dict:

        for k, v in data.items():
            # device
            if isinstance(v, torch.Tensor):
                data[k] = v.to(self.device)

        return data

    def _good_matches(self,
                      preds: Dict[str, Union[torch.Tensor, List, Tuple]]
                      ) -> Dict:
        raise NotImplementedError

    @torch.no_grad()
    def match_pair(self,
                   data: Dict[str, Union[torch.Tensor, List, Tuple]]
                   ) -> Dict:

        assert "descriptors0" in data, KeyError("descriptors0 missing")
        assert "descriptors1" in data, KeyError("descriptors1 missing")

        # prepare
        data = self._prepare_inputs(data)

        # match
        preds = self.matcher(data)

        # good matches
        if self.cfg.matcher.good_matches:
            preds = self._good_matches(preds)

        return preds


class WorkQueue():
    def __init__(self, work_fn, num_workers=1):
        self.queue = Queue(num_workers)
        self.threads = [Thread(target=self.thread_fn, args=(work_fn,))
                        for _ in range(num_workers)]

        for thread in self.threads:
            thread.start()

    def join(self):

        for thread in self.threads:
            self.queue.put(None)

        for thread in self.threads:
            thread.join()

    def thread_fn(self, work_fn):
        item = self.queue.get()

        while item is not None:
            work_fn(item)
            item = self.queue.get()

    def put(self, data):
        self.queue.put(data)


class MatchSequence(Matcher):
    def __init__(self, cfg: Dict = {}) -> None:
        super().__init__(cfg)

    def _sequence_loader(self,
                         sequence
                         ):

        if isinstance(sequence, DataLoader):
            return sequence
        else:
            return DataLoader(sequence, num_workers=self.cfg.num_workers,
                              batch_size=1, shuffle=False, pin_memory=True)

    def match_sequence(self,
                       sequence: Any,
                       save_path: Path
                       ) -> Dict:
        #
        seq_dl = self._sequence_loader(sequence)

        # writer
        writer = MatchesWriter(save_path)

        # workers
        writer_queue = WorkQueue(
            partial(writer.write_matches), self.cfg.num_workers)

        # match
        for _, (src_name, dst_name, data) in enumerate(tqdm(seq_dl, total=len(seq_dl))):

            # match
            preds = self.match_pair(data)

            # get key
            pair_key = pairs2key(src_name[0], dst_name[0])

            # put
            writer_queue.put((pair_key, preds))

        writer_queue.join()
        writer.close()

        logger.info("matches saved to %s", str(save_path))

        return save_path

def wrap_keys_with_extenstion(data: Dict, ext="0"):
    new_data = {}
    
    keys = list(data.keys())
    for k in keys:
        new_data[k+ext] = data.pop(k)
        
    return new_data
        
        
class MatchQueryDatabase(Matcher):
    def __init__(self, cfg: Dict = {}) -> None:
        super().__init__(cfg)

    def match_query_database(self, q_preds, pairs):

        local_features_path     = Path(str(self.cfg.visloc_path) + '/' + 'db_local_features' + '.h5')
        local_features_loader   = LocalFeaturesLoader(save_path=local_features_path)

        for src_name, dst_name in pairs:
            db_preds = local_features_loader.load(dst_name)
            
            q_preds     = wrap_keys_with_extenstion(q_preds,    ext="0")
            db_preds    = wrap_keys_with_extenstion(db_preds,   ext="1")
            print(list(q_preds.keys()))
            print(list(db_preds.keys()))
        
            
            preds = self.match_pair({**q_preds, **db_preds})
            
            print(preds)
            print(preds.keys())
            input()
        pass
