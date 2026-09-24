import copy
from torch import nn


def get_backbone(args, pretrained=False):
    name = args["backbone_type"].lower()

    if name == "pretrained_vit_b16_224_vpt":
        from backbone.vpt_backbone import build_promptmodel
        vpt_type = "Shallow" if str(args.get("vpt_type", "Deep")).lower() == "shallow" else "Deep"
        model = build_promptmodel(
            modelname="vit_base_patch16_224",
            Prompt_Token_num=args["prompt_token_num"],
            VPT_type=vpt_type,
            args=args,
        )
        prompt_state_dict = model.obtain_prompt()
        model.load_prompt(prompt_state_dict)
        model.out_dim = 768
        return model.eval()
    else:
        raise NotImplementedError(f"Unknown backbone: {name}")


class BaseNet(nn.Module):
    def __init__(self, args, pretrained):
        super().__init__()
        self.backbone = get_backbone(args, pretrained)
        self.fc = None
        self._device = args["device"][0]
        self.model_type = "vit"

    @property
    def feature_dim(self):
        return self.backbone.out_dim

    def extract_vector(self, x):
        return self.backbone(x)

    def forward(self, x):
        x = self.backbone(x)
        out = self.fc(x)
        out.update({"features": x})
        return out

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self
