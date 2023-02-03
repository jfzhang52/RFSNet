from typing import Dict, Any
from mmcv.utils import Config
from importlib import import_module

from src.models import BaseTester
from src.datasets import BaseLoader


class Tester(BaseTester):
    def __init__(
            self,
            ckpt: Any,
            cfg: Config,
            data_loader: BaseLoader,
            vis_options: Dict
    ) -> None:
        super(Tester, self).__init__(ckpt, cfg, data_loader, vis_options)

        if "RFSNet_" in cfg.model.name:
            RFSNet = import_module(f"src.models.RFSNet.{cfg.model.name}").RFSNet
        else:
            from src.models.RFSNet import RFSNet

        model = RFSNet(self.cfg)

        self.model = model.float().to(self.device)
        self.model.load_state_dict(self.ckpt["net"])
