from .networks import *
from .lr_scheduler import ChainedScheduler
import os
from .networks.layers import *
import random
from .networks.pose_decoder import PoseDecoder
from .networks.resnet_encoder import ResnetEncoder
from .networks.depth_decoder import DepthDecoder
#from .networks.cednet import CEDNet
#from .networks.ghostnetv3 import ghostnetv3
from .networks.swiftformer import SwiftFormer_S
#from .networks.depth_encoder import LiteMono
#from .networks.depth_decoder_litemono import DepthDecoderV2
#from .networks.pose_decode_litemono import PoseDecoderV2
from .networks.intrinsics_decoder import IntrinsicsHead

from .networks.shvit import SHViTEncoder

EPS = 1e-7


class EstimateDepth():
    def __init__(self, cfgs):
        self.model_name = cfgs.get('model_name', self.__class__.__name__)
        self.height = cfgs.get('height', 256)
        self.width = cfgs.get('width', 320)
        self.batch_size = cfgs.get('batch_size', 64)

        # checking height and width are multiples of 32
        assert self.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.width % 32 == 0, "'width' must be a multiple of 32"

        self.device = cfgs.get('device', 'cpu')
        self.scales = cfgs.get('scales', [0, 1, 2, 3])
        self.num_scales = len(self.scales)
        self.frame_ids = cfgs.get('frame_ids', [0, -1, 1])
        self.num_pose_frames = 2
        self.disable_automasking = cfgs.get('disable_automasking', False)

        assert self.frame_ids[0] == 0, "frame_ids must start with 0"

        # depth
        self.min_depth = cfgs.get('min_depth', 0.1)
        self.max_depth = cfgs.get('max_depth', 100.0)
        self.min_gt_depth = cfgs.get('min_gt_depth', 0.001)
        self.max_gt_depth = cfgs.get('max_gt_depth', 1.)

        self.model_str = cfgs.get('model_str', 'monodepth2')
        self.scheduler_str = cfgs.get('scheduler_str', None)
        self.num_layers = cfgs.get('num_layers', 18)
        self.weights_init = cfgs.get('weights_init', "pretrained")
        if self.model_str == "monodepth2":
            # self.net_depth_encoder = ResnetEncoder(self.num_layers, self.weights_init == "pretrained")
            # self.net_depth_encoder = ghostnetv3(width=1.0)
            self.net_depth_encoder = SwiftFormer_S()
            self.net_depth_decoder = DepthDecoder(np.array([24, 48, 64, 168, 224]), [0, 1, 2, 3], )

            # pose
            self.net_pose_encoder = ResnetEncoder(self.num_layers, self.weights_init == "pretrained",
                                                  num_input_images=self.num_pose_frames)
            self.net_pose_decoder = PoseDecoder(self.net_pose_encoder.num_ch_enc, num_input_features=1,
                                                num_frames_to_predict_for=2)

            self.net_depth_intrinsics = IntrinsicsHead(self.net_pose_encoder.num_ch_enc)
        elif self.model_str == "shvit":
            # 1. 初始化编码器
            self.net_depth_encoder = SHViTEncoder(
                model_type='shvit_s1',
                height=self.height,
                width=self.width
            )

            # 【显式加载】直接在这里写死你的权重路径
            # 你可以将这里的字符串替换为你电脑上 shvit_s1.pth 的真实绝对路径
            shvit_weight_path = "/media/mems509/9b308a11-7150-4494-8f42-71df9385ff43/home/mems509/wjy/MonoLoT-main/shvit_s1.pth"
            self.net_depth_encoder.load_pretrained(shvit_weight_path)

            # 2. 初始化解码器
            # 注意：SHViT 只提供 3 个尺度的特征 (1/16, 1/32, 1/64)
            # 原有的 DepthDecoder 默认适配 ResNet 的 5 个尺度
            # 建议使用 LiteMono 的 DepthDecoderV2，它能完美处理 3 个尺度的输入
            # from .networks.depth_decoder_litemono import DepthDecoderV2
            # self.net_depth_decoder = DepthDecoderV2(
            #     self.net_depth_encoder.num_ch_enc,
            #     self.scales
            # )
            # 不要使用 DepthDecoderV2，因为它默认只有 3 级
            from .networks.depth_decoder import DepthDecoder
            self.net_depth_decoder = DepthDecoder(
                self.net_depth_encoder.num_ch_enc,
                self.scales
            )

            # 3. Pose 部分 (维持原样，通常用 ResNet 比较稳)
            self.net_pose_encoder = ResnetEncoder(
                self.num_layers,
                self.weights_init == "pretrained",
                num_input_images=self.num_pose_frames
            )
            self.net_pose_decoder = PoseDecoder(
                self.net_pose_encoder.num_ch_enc,
                num_input_features=1,
                num_frames_to_predict_for=2
            )
            self.net_depth_intrinsics = IntrinsicsHead(self.net_pose_encoder.num_ch_enc)
        elif self.model_str in ["lite-mono", "lite-mono-small", "lite-mono-tiny", "lite-mono-8m"]:
            self.drop_path = cfgs.get('drop_path', 0.2)
            self.net_depth_encoder = LiteMono(model=self.model_str, drop_path_rate=self.drop_path, width=self.width,
                                              height=self.height)
            self.net_depth_decoder = DepthDecoderV2(self.net_depth_encoder.num_ch_enc, self.scales)
            # pose
            self.net_pose_encoder = ResnetEncoder(self.num_layers, self.weights_init == "pretrained",
                                                  num_input_images=self.num_pose_frames)
            self.net_pose_decoder = PoseDecoderV2(self.net_pose_encoder.num_ch_enc, num_input_features=1,
                                                  num_frames_to_predict_for=2)
        elif self.model_str == 'monovit':
            self.net_depth_encoder = mpvit_small()
            self.net_depth_decoder = MonovitDecoder()
            # pose
            self.net_pose_encoder = ResnetEncoder(self.num_layers, self.weights_init == "pretrained",
                                                  num_input_images=self.num_pose_frames)
            self.net_pose_decoder = PoseDecoder(self.net_pose_encoder.num_ch_enc, num_input_features=1,
                                                num_frames_to_predict_for=2)
        else:
            raise NotImplementedError

        # optim
        self.start_epoch = cfgs.get('start_epoch', 0)
        self.num_epochs = cfgs.get('num_epochs', 20)
        self.lr = cfgs.get('lr', [0.0001, 5e-6, 36, 0.0001, 1e-5, 36])
        self.weight_decay = cfgs.get('weight_decay', 0.02)
        self.disparity_smoothness = cfgs.get('disparity_smoothness', 0.001)

        # matcher loss
        self.disable_matcher = cfgs.get('disable_matcher', False)
        self.confidence = cfgs.get('confidence', 0.9)
        self.matcher_loss_alpha = cfgs.get('matcher_loss_alpha', 0.2)
        self.half_epoch_matcher = cfgs.get('half_epoch_matcher', False)
        self.matcher_loss_delta = cfgs.get('matcher_loss_delta', 0)

        self.network_names = [k for k in vars(self) if k.startswith('net')]
        # optimizer
        if self.model_str in ['monovit', 'lite-mono', 'shvit']:  # 增加 shvit
            self.make_optimizer = lambda optim_dict: torch.optim.AdamW(
                optim_dict["parameters"], lr=optim_dict["lr"], weight_decay=self.weight_decay)
        elif self.model_str == "monodepth2":

            self.make_optimizer = lambda optim_dict: torch.optim.Adam(
                optim_dict["parameters"], lr=optim_dict["lr"])
            """
            self.make_optimizer = lambda optim_dict: torch.optim.AdamW(
                optim_dict["parameters"], lr=optim_dict["lr"], betas=(0.9, 0.999), weight_decay=self.weight_decay)
            """

        # ratio consistency
        self.ratio_consistency = cfgs.get('ratio_consistency', False)
        self.ratio_consistency_crop = cfgs.get('ratio_consistency_crop', False)
        self.ratio_consistency_normalization = cfgs.get('ratio_consistency_normalization', False)
        self.ratio_consistency_scales_normalization = cfgs.get('ratio_consistency_scales_normalization', False)
        self.weight_ratio_consistency_crop = cfgs.get('weight_ratio_consistency_crop', 1.0)
        self.align_crop_position = cfgs.get('align_crop_position', False)

        # geometry loss
        self.geometry_loss = cfgs.get('geometry_loss', False)
        self.geometry_loss_disp_mode = cfgs.get('geometry_loss_disp_mode', False)

        # load
        self.load_weights_folder = cfgs.get('load_weights_folder', None)
        self.mypretrain = cfgs.get('mypretrain', None)
        self.not_load_nets = cfgs.get('not_load_nets', ())
        self.not_load_optimizer = cfgs.get('not_load_optimizer', ())
        self.models_to_load = cfgs.get('models_to_load', [])

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        # no grad layers
        self.ssim = SSIM()
        for scale in self.scales:
            h = self.height // (2 ** scale)
            w = self.width // (2 ** scale)
            setattr(self, "backproject_depth_{}".format(scale), BackprojectDepth(self.batch_size, h, w))
            setattr(self, "project_3d_{}".format(scale), Project3D(self.batch_size, h, w))

        self.other_param_names = ['ssim']
        for scale in self.scales:
            self.other_param_names += ["backproject_depth_{}".format(scale), "project_3d_{}".format(scale)]

    def init_optimizers(self):
        # optim
        self.optimizer_names = []
        self.parameters_depth = []
        self.parameters_pose = []
        for net_name in self.network_names:
            if not any([p.requires_grad for p in getattr(self, net_name).parameters()]):
                continue
            if net_name.startswith('net_depth'):
                self.parameters_depth += list(getattr(self, net_name).parameters())
            elif net_name.startswith('net_pose'):
                self.parameters_pose += list(getattr(self, net_name).parameters())
        self.optimizer_depth = self.make_optimizer({"parameters": self.parameters_depth, "lr": self.lr[0]})
        self.optimizer_pose = self.make_optimizer({"parameters": self.parameters_pose, "lr": self.lr[3]})
        self.optimizer_names += ["optimizer_depth", "optimizer_pose"]

        # scheduler
        self.scheduler_names = []
        if self.scheduler_str is not None:
            print("scheduler_str mode...")
            if self.scheduler_str == 'cosine':
                self.scheduler_depth_lr = ChainedScheduler(
                    self.optimizer_depth,
                    T_0=int(self.lr[2]), T_mul=1, eta_min=self.lr[1], last_epoch=self.start_epoch - 1,
                    max_lr=self.lr[0], warmup_steps=0, gamma=0.9)
                self.scheduler_pose_lr = ChainedScheduler(
                    self.optimizer_pose,
                    T_0=int(self.lr[5]), T_mul=1, eta_min=self.lr[4], last_epoch=self.start_epoch - 1,
                    max_lr=self.lr[3], warmup_steps=0, gamma=0.9)
                print("optimiser cosine")
            elif self.scheduler_str == 'exp':
                self.scheduler_depth_lr = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_depth, 0.9)
                self.scheduler_pose_lr = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_pose, 0.9)
                print("optimiser exp")
            elif self.scheduler_str == 'step':
                self.scheduler_depth_lr = torch.optim.lr_scheduler.StepLR(self.optimizer_depth, self.num_epochs - 5,
                                                                          0.1)
                self.scheduler_pose_lr = torch.optim.lr_scheduler.StepLR(self.optimizer_pose, self.num_epochs - 5, 0.1)
                print("optimiser step")
        else:
            if self.model_str == 'lite-mono' or self.model_str == 'shvit': # <--- 修改这里，加上 'shvit':
                self.scheduler_depth_lr = ChainedScheduler(
                    self.optimizer_depth,
                    T_0=int(self.lr[2]), T_mul=1, eta_min=self.lr[1], last_epoch=self.start_epoch - 1,
                    max_lr=self.lr[0], warmup_steps=0, gamma=0.9)
                self.scheduler_pose_lr = ChainedScheduler(
                    self.optimizer_pose,
                    T_0=int(self.lr[5]), T_mul=1, eta_min=self.lr[4], last_epoch=self.start_epoch - 1,
                    max_lr=self.lr[3], warmup_steps=0, gamma=0.9)
                print("optimiser cosine")
            elif self.model_str == 'monovit':
                self.scheduler_depth_lr = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_depth, 0.9)
                self.scheduler_pose_lr = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_pose, 0.9)
                print("optimiser exp")
            elif self.model_str == 'monodepth2':

                self.scheduler_depth_lr = torch.optim.lr_scheduler.StepLR(self.optimizer_depth, self.num_epochs - 5,
                                                                          0.1)
                self.scheduler_pose_lr = torch.optim.lr_scheduler.StepLR(self.optimizer_pose, self.num_epochs - 5, 0.1)
                """
                self.scheduler_depth_lr = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_depth,
                                                                               [15], 0.1)
                self.scheduler_pose_lr = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_pose, [15],

                                                                           0.1)
                """

                print("optimiser step")

        self.scheduler_names += ["scheduler_depth_lr", "scheduler_pose_lr"]

    def load_model(self):
        """Load model(s) from disk
        """
        self.load_weights_folder = os.path.expanduser(self.load_weights_folder)

        assert os.path.isdir(self.load_weights_folder), \
            "Cannot find folder {}".format(self.load_weights_folder)
        print("loading model from folder {}".format(self.load_weights_folder))

        for n in self.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.load_weights_folder, "{}.pth".format(n))

            model_dict = getattr(self, n).state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            getattr(self, n).load_state_dict(model_dict)

        # loading optimizer state
        optimizer_depth_load_path = os.path.join(self.load_weights_folder, "optimizer_depth.pth")
        optimizer_pose_load_path = os.path.join(self.load_weights_folder, "optimizer_pose.pth")
        if os.path.isfile(optimizer_depth_load_path) and os.path.isfile(optimizer_pose_load_path):
            print("Loading optimizer weights")
            self.optimizer_depth.load_state_dict(torch.load(optimizer_depth_load_path))
            self.optimizer_pose.load_state_dict(torch.load(optimizer_pose_load_path))
        else:
            print("Cannot find optimizer weights so Adam is randomly initialized")

    def load_shvit_weights(self, path):
        """为 SHViT-S1 加载预训练权重"""
        if not path or not os.path.isfile(path):
            print(f"=> [Error] Pretrain file not found at: {path}")
            return

        print(f"=> Loading SHViT-S1 weights from: {path}")
        checkpoint = torch.load(path, map_location="cpu")

        # 提取 state_dict
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # 清理键名：
        # 1. 移除 'backbone.' 或 'module.' 前缀
        # 2. 忽略分类头 'head.' 或 'fc.'
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k
            if name.startswith('module.'): name = name[7:]
            if name.startswith('backbone.'): name = name[9:]

            # 过滤掉不属于编码器的层（分类层）
            if any(x in name for x in ['head', 'fc', 'classifier']):
                continue

            new_state_dict[name] = v

        # 加载到 net_depth_encoder.encoder 中
        # 因为我们的封装结构是：SHViTEncoder -> self.encoder (Partial_ViT_Exp)
        msg = self.net_depth_encoder.encoder.load_state_dict(new_state_dict, strict=False)

        print(f"=> Successfully loaded weights.")
        print(f"=> Missing keys (should only be head-related): {msg.missing_keys[:5]}")

    def load_pretrain(self):

        # 检查是否是 SHViT 模型
        if self.model_str == 'shvit':
            self.load_shvit_weights(self.mypretrain)
            return  # 加载完直接返回

        # 以下是原有的加载逻辑 (针对 lite-mono 等)
        if self.mypretrain is None:
            return

        # only designed for lite-mono
        # self.mypretrain = os.path.expanduser(self.mypretrain)
        path = self.mypretrain
        ckpt = torch.load(path, map_location="cpu")
        if 'state_dict' in ckpt:
            _state_dict = ckpt['state_dict']
        elif 'model' in ckpt:
            _state_dict = ckpt['model']
        else:
            _state_dict = ckpt
        state_dict = _state_dict
        self.net_depth_encoder.load_state_dict(state_dict, strict=False)
        print('mypretrain loaded.')

    def load_model_state(self, cp):
        for k in cp:
            if k and k in self.network_names and k not in self.not_load_nets:
                print("Loading ", k)
                model_dict = getattr(self, k).state_dict()
                getattr(self, k).load_state_dict({k: v for k, v in cp[k].items() if k in model_dict})

    def load_optimizer_state(self, cp):
        for k in cp:
            if k and k in self.optimizer_names and k not in self.not_load_optimizer:
                print("Loading ", k)
                getattr(self, k).load_state_dict(cp[k])

    def get_model_state(self):
        states = {}
        for net_name in self.network_names:
            states[net_name] = getattr(self, net_name).state_dict()
        return states

    def get_optimizer_state(self):
        states = {}
        for optim_name in self.optimizer_names:
            states[optim_name] = getattr(self, optim_name).state_dict()
        return states

    def to_device(self, device):
        self.device = device
        for net_name in self.network_names:
            setattr(self, net_name, getattr(self, net_name).to(device))

        if self.other_param_names:
            for param_name in self.other_param_names:
                setattr(self, param_name, getattr(self, param_name).to(device))

    def set_train(self):
        for net_name in self.network_names:
            getattr(self, net_name).train()

    def set_eval(self):
        for net_name in self.network_names:
            getattr(self, net_name).eval()

    def backward(self, losses):
        for optim_name in self.optimizer_names:
            getattr(self, optim_name).zero_grad()
        losses["loss"].backward()
        for optim_name in self.optimizer_names:
            getattr(self, optim_name).step()

    def forward(self, inputs):
        """Feedforward once."""
        for key, ipt in inputs.items():
            if "correspondences" in key:
                inputs[key] = ipt
            else:
                inputs[key] = ipt.to(self.device)

        # we only feed the image with frame_id 0 through the depth encoder
        features = self.net_depth_encoder(inputs["color_aug", 0, 0])

        outputs = self.net_depth_decoder(features)

        x = self.net_depth_encoder(inputs[("color_local_aug", 0, 0)])
        o = self.net_depth_decoder(x)
        for i in range(1):
            outputs[("disp_local", i)] = o[("disp", i)]

        x = self.net_depth_encoder(inputs[("color_reshuffle_aug", 0, 0)])
        o = self.net_depth_decoder(x)
        for i in range(1):
            outputs[("disp_reshuffle", i)] = o[("disp", i)]

            all_disp = []
            for b in range(self.batch_size):
                ### Split-Permute as depicted in paper (vertical + horizontal)
                split_x = inputs[("split_xy")][b][0].item()
                split_y = inputs[("split_xy")][b][1].item()
                split_x = round(split_x / (2 ** i))
                split_y = round(split_y / (2 ** i))
                disp_reshuffle = outputs[("disp_reshuffle", i)][b]  # 1*H*W
                patch1 = disp_reshuffle[:, 0:split_y, :]
                patch2 = disp_reshuffle[:, split_y:, :]
                disp_restore = torch.cat([patch2, patch1], dim=1)
                patch1 = disp_restore[:, :, 0:split_x]
                patch2 = disp_restore[:, :, split_x:]
                disp_restore = torch.cat([patch2, patch1], dim=2)
                all_disp.append(disp_restore)

                ### Split-Permute (vertical or horizontal, randomly choose one)
                # split_x = inputs[("split_xy", i)][b][0].item()
                # split_y = inputs[("split_xy", i)][b][1].item()
                # split_x = round(split_x / (2 ** i))
                # split_y = round(split_y / (2 ** i))
                # disp_reshuffle = outputs[("disp_reshuffle", i)][b]   #1*H*W
                # if split_x == 0:
                #     patch1 = disp_reshuffle[:, 0:split_y, :]
                #     patch2 = disp_reshuffle[:, split_y:, :]
                #     disp_restore = torch.cat([patch2, patch1], dim=1)
                # else:
                #     patch1 = disp_reshuffle[:, :, 0:split_x]
                #     patch2 = disp_reshuffle[:, :, split_x:]
                #     disp_restore = torch.cat([patch2, patch1], dim=2)
                # all_disp.append(disp_restore)
            disp_restore = torch.stack(all_disp, dim=0)
            outputs[("disp_reshuffle", i)] = disp_restore

        outputs.update(self.predict_poses(inputs))

        # depth in frame [-1, 1] and scales [0, 1, 2, 3]
        if self.geometry_loss:
            for i, frame_id in enumerate(self.frame_ids[1:]):
                outputs_disp_scales = self.net_depth_decoder(self.net_depth_encoder(inputs["color_aug", frame_id, 0]))
                for scale in self.scales:
                    disp = outputs_disp_scales[("disp", scale)]
                    disp = F.interpolate(
                        disp, [self.height, self.width], mode="bilinear", align_corners=False)
                    _, depth = disp_to_depth(disp, self.min_depth, self.max_depth)
                    outputs[("depth", frame_id, scale)] = depth
                    if self.geometry_loss_disp_mode:
                        outputs[("disp", frame_id, scale)] = disp

        # shuffle
        outputs["do_shuffle"] = random.random() > 0.5
        if self.ratio_consistency and outputs["do_shuffle"]:
            direction = random.random() > 0.5
            inputs["shuffle_color_aug", 0, 0] = self.layer_shuffle(inputs["color_aug", 0, 0], direction)
            shuffle_features = self.net_depth_encoder(inputs["shuffle_color_aug", 0, 0])
            shuffle_outputs = self.net_depth_decoder(shuffle_features)
            for scale in self.scales:
                outputs[("shuffle_disp", scale)] = self.layer_shuffle(shuffle_outputs[("disp", scale)], direction)
        elif self.ratio_consistency_crop and outputs["do_shuffle"]:
            b, _, h, w = inputs["color_aug", 0, 0].shape
            crop_info = self.get_crop_info(h // 2 ** (self.num_scales - 1), w // 2 ** (self.num_scales - 1),
                                           align_crop_position=self.align_crop_position) * 2 ** (self.num_scales - 1)
            inputs['crop_info'] = crop_info

            inputs["shuffle_color_aug", 0, 0] = self.layer_crop_shuffle(inputs["color_aug", 0, 0], crop_info)
            shuffle_features = self.net_depth_encoder(inputs["shuffle_color_aug", 0, 0])
            shuffle_outputs = self.net_depth_decoder(shuffle_features)

            for scale in self.scales:
                outputs[("shuffle_disp", scale)] = self.layer_crop_shuffle(shuffle_outputs[("disp", scale)],
                                                                           crop_info // 2 ** scale)

        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    def layer_shuffle(self, input_raw, direction=True):
        # input raw: (b, 3, h, w)
        if direction:
            chunk_0, chunk_1 = torch.chunk(input_raw, 2)
            chunk_0_up, chunk_0_botton = torch.chunk(chunk_0, 2, 2)
            chunk_1_up, chunk_1_botton = torch.chunk(chunk_1, 2, 2)
            chunk_0_up_1_botton = torch.cat([chunk_0_up, chunk_1_botton], 2)
            chunk_1_up_0_botton = torch.cat([chunk_1_up, chunk_0_botton], 2)
            shuffle_input = torch.cat([chunk_0_up_1_botton, chunk_1_up_0_botton], 0)
        else:
            chunk_0, chunk_1 = torch.chunk(input_raw, 2)
            chunk_0_left, chunk_0_right = torch.chunk(chunk_0, 2, 3)
            chunk_1_left, chunk_1_right = torch.chunk(chunk_1, 2, 3)
            chunk_0_left_1_right = torch.cat([chunk_0_left, chunk_1_right], 3)
            chunk_1_left_0_right = torch.cat([chunk_1_left, chunk_0_right], 3)
            shuffle_input = torch.cat([chunk_0_left_1_right, chunk_1_left_0_right], 0)
        return shuffle_input

    def layer_crop_shuffle(self, input_raw, crop_info):
        chunk_0, chunk_1 = torch.chunk(input_raw, 2)
        x1, y1, x1p, y1p, patch_width, patch_height, w, h = crop_info  # [232 136  64   8  56  80 320 256]

        split_xp = [x1p, patch_width, w - (x1p + patch_width)]
        split_yp = [y1p, patch_height, h - (y1p + patch_height)]

        chunk_0_left_middle_right = torch.split(chunk_0, split_xp, dim=3)
        chunk_0_middle_up_middle_center_middle_right = torch.split(chunk_0_left_middle_right[1], split_yp, dim=2)

        split_x = [x1, patch_width, w - (x1 + patch_width)]
        split_y = [y1, patch_height, h - (y1 + patch_height)]

        chunk_1_left_middle_right = torch.split(chunk_1, split_x, dim=3)
        chunk_1_middle_up_middle_center_middle_right = torch.split(chunk_1_left_middle_right[1], split_y, dim=2)

        shuffle_0_middle = torch.cat(
            [chunk_0_middle_up_middle_center_middle_right[0], chunk_1_middle_up_middle_center_middle_right[1],
             chunk_0_middle_up_middle_center_middle_right[2]], 2)
        shuffle_0 = torch.cat([chunk_0_left_middle_right[0], shuffle_0_middle, chunk_0_left_middle_right[2]], 3)

        shuffle_1_middle = torch.cat(
            [chunk_1_middle_up_middle_center_middle_right[0], chunk_0_middle_up_middle_center_middle_right[1],
             chunk_1_middle_up_middle_center_middle_right[2]], 2)
        shuffle_1 = torch.cat([chunk_1_left_middle_right[0], shuffle_1_middle, chunk_1_left_middle_right[2]], 3)

        shuffle_input = torch.cat([shuffle_0, shuffle_1], 0)
        return shuffle_input

    def get_crop_info(self, h, w, min_patch_ratio=0.6, max_path_ratio=0.8, align_crop_position=False):
        min_width = round(min_patch_ratio * w)
        max_width = round(max_path_ratio * w)

        min_height = round(min_patch_ratio * h)
        max_height = round(max_path_ratio * h)

        patch_width = np.random.randint(min_width, max_width + 1)
        patch_height = np.random.randint(min_height, max_height + 1)

        x1 = np.random.randint(0, w - patch_width)
        y1 = np.random.randint(0, h - patch_height)
        # x2 = x1 + patch_width
        # y2 = y1 + patch_height

        if align_crop_position:
            x1p = x1
            y1p = y1
        else:
            x1p = np.random.randint(0, w - patch_width)
            y1p = np.random.randint(0, h - patch_height)
        # x2p = x1p + patch_width
        # y2p = y1p + patch_height

        return np.array([x1, y1, x1p, y1p, patch_width, patch_height, w, h], dtype=int)

    def predict_poses(self, inputs):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.frame_ids}

            for f_i in self.frame_ids[1:]:
                # To maintain ordering we always pass frames in temporal order
                if f_i < 0:
                    pose_inputs = [pose_feats[f_i], pose_feats[0]]
                else:
                    pose_inputs = [pose_feats[0], pose_feats[f_i]]

                pose_inputs = [self.net_pose_encoder(torch.cat(pose_inputs, 1))]

                axisangle, translation, intermediate_feature = self.net_pose_decoder(pose_inputs)

                outputs[("axisangle", 0, f_i)] = axisangle
                outputs[("translation", 0, f_i)] = translation

                # Invert the matrix if the frame id is negative
                outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                    axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

                cam_K = self.net_depth_intrinsics(intermediate_feature, self.width, self.height)
                inv_K = torch.inverse(cam_K)
                outputs[('K', 0)] = cam_K
                outputs[('inv_K', 0)] = inv_K
        else:
            raise NotImplementedError

        return outputs

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.scales:
            disp = outputs[("disp", scale)]
            disp = F.interpolate(
                disp, [self.height, self.width], mode="bilinear", align_corners=False)
            source_scale = 0
            _, depth = disp_to_depth(disp, self.min_depth, self.max_depth)
            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.frame_ids[1:]):

                T = outputs[("cam_T_cam", 0, frame_id)]
                _backproject_depth = getattr(self, "backproject_depth_{}".format(source_scale))
                cam_points = _backproject_depth(
                    depth, outputs[('inv_K', 0)])
                _project_3d = getattr(self, "project_3d_{}".format(source_scale))

                if self.geometry_loss:
                    pix_coords, computed_depth = _project_3d(
                        cam_points, inputs[("K", source_scale)], T, compute_depth=True)
                    outputs[('computed_depth', frame_id, scale)] = computed_depth
                    if self.geometry_loss_disp_mode:
                        computed_disp = depth_to_disp(computed_depth, self.min_depth, self.max_depth)
                        outputs[('computed_disp', frame_id, scale)] = computed_disp
                else:
                    pix_coords = _project_3d(
                        cam_points, outputs[('K', 0)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border", align_corners=True)

                if self.geometry_loss:
                    outputs[("sampled_depth", frame_id, scale)] = F.grid_sample(
                        outputs[("depth", frame_id, source_scale)],
                        outputs[("sample", frame_id, scale)],
                        padding_mode="border", align_corners=True)
                    if self.geometry_loss_disp_mode:
                        outputs[("sampled_disp", frame_id, scale)] = F.grid_sample(
                            outputs[("disp", frame_id, source_scale)],
                            outputs[("sample", frame_id, scale)],
                            padding_mode="border", align_corners=True)

                outputs[("color_identity", frame_id, scale)] = inputs[("color", frame_id, source_scale)]

        for scale in [0]:
            source_scale = 0
            disp = outputs[("disp_reshuffle", scale)]
            disp = F.interpolate(disp, [self.height, self.width], mode="bilinear", align_corners=False)
            _, depth = disp_to_depth(disp, self.min_depth, self.max_depth)
            for _, frame_id in enumerate(self.frame_ids[1:]):
                T = outputs[("cam_T_cam", 0, frame_id)]
                _backproject_depth = getattr(self, "backproject_depth_{}".format(source_scale))

                cam_points = _backproject_depth(
                    depth, outputs[('inv_K', 0)])
                _project_3d = getattr(self, "project_3d_{}".format(source_scale))
                pix_coords = _project_3d(
                    cam_points, outputs[('K', 0)], T)
                outputs[("color_reshuffle", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)].clone(),
                    pix_coords,
                    padding_mode="border", align_corners=True)

            disp = outputs[("disp_local", scale)]
            disp = F.interpolate(disp, [self.height, self.width], mode="bilinear", align_corners=False)
            _, depth = disp_to_depth(disp, self.min_depth, self.max_depth)
            for _, frame_id in enumerate(self.frame_ids[1:]):
                T = outputs[("cam_T_cam", 0, frame_id)]
                Rt_Rc = torch.zeros_like(T).to(self.device)
                gx0 = (inputs[("grid_local")][:, 0, 0, -1] + inputs[("grid_local")][:, 0, 0, 0]) / 2.
                gy0 = (inputs[("grid_local")][:, 1, -1, 0] + inputs[("grid_local")][:, 1, 0, 0]) / 2.
                f = (inputs[("grid_local")][:, 0, 0, -1] - inputs[("grid_local")][:, 0, 0, 0]) / 2.
                fx = inputs[("K", 0)][0, 0, 0] / self.width
                fy = inputs[("K", 0)][0, 1, 1] / self.height
                Rc_v = torch.stack([-gx0 / (2 * fx), -gy0 / (2 * fy), f], dim=1)
                Rc = torch.eye(3).to(self.device)
                Rc = Rc[None, :, :].repeat(Rc_v.shape[0], 1, 1)
                Rc[:, :, 2] = Rc_v
                # outputs[("Rc", f_i)] = Rc
                Rt_Rc[:, :3, :3] = torch.matmul(Rc, torch.matmul(T[:, :3, :3], torch.inverse(Rc)))
                Rt_Rc[:, :3, 3:4] = torch.matmul(Rc, T[:, :3, 3:4])
                T = Rt_Rc

                _backproject_depth = getattr(self, "backproject_depth_{}".format(source_scale))

                cam_points = _backproject_depth(
                    depth, outputs[('inv_K', 0)])
                _project_3d = getattr(self, "project_3d_{}".format(source_scale))

                pix_coords = _project_3d(
                    cam_points, outputs[('K', 0)], T)
                outputs[("color_local", frame_id, scale)] = F.grid_sample(
                    inputs[("color_local", frame_id, source_scale)],
                    pix_coords, padding_mode="border", align_corners=True)

    def compute_losses_local(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """

        losses = {}
        total_loss = 0

        for scale in [0]:
            loss = 0
            reprojection_losses = []

            source_scale = 0

            disp = outputs[("disp_local", scale)]
            color = inputs[("color_local", 0, scale)]
            target = inputs[("color_local", 0, source_scale)]

            for frame_id in self.frame_ids[1:]:
                pred = outputs[("color_local", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            identity_reprojection_losses = []
            for frame_id in self.frame_ids[1:]:
                pred = inputs[("color_local", frame_id, source_scale)]
                identity_reprojection_losses.append(
                    # if camera does not move, pred and target are the same, so that loss=0
                    self.compute_reprojection_loss(pred, target))

            identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

            # save both images, and do min all at once below
            identity_reprojection_loss = identity_reprojection_losses

            reprojection_loss = reprojection_losses

            # add random numbers to break ties
            identity_reprojection_loss += torch.randn(
                identity_reprojection_loss.shape, device=self.device) * 0.00001

            # [not move, corretlymatch]
            combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)

            to_optimise, idxs = torch.min(combined, dim=1)

            # true means corretly match, false means not move
            outputs["identity_selection/{}".format(scale)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss

        return total_loss

    def compute_losses_reshuffle(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """

        losses = {}
        total_loss = 0

        for scale in [0]:
            loss = 0
            reprojection_losses = []

            source_scale = 0

            disp = outputs[("disp_reshuffle", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.frame_ids[1:]:
                pred = outputs[("color_reshuffle", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            identity_reprojection_losses = []
            for frame_id in self.frame_ids[1:]:
                pred = inputs[("color", frame_id, source_scale)]
                identity_reprojection_losses.append(
                    # if camera does not move, pred and target are the same, so that loss=0
                    self.compute_reprojection_loss(pred, target))

            identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

            # save both images, and do min all at once below
            identity_reprojection_loss = identity_reprojection_losses

            reprojection_loss = reprojection_losses

            # add random numbers to break ties
            identity_reprojection_loss += torch.randn(
                identity_reprojection_loss.shape, device=self.device) * 0.00001

            # [not move, corretlymatch]
            combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)

            to_optimise, idxs = torch.min(combined, dim=1)

            # true means corretly match, false means not move
            outputs["identity_selection/{}".format(scale)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss

        return total_loss

    def compute_losses_ori(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """

        losses = {}
        total_loss = 0

        for scale in self.scales:
            loss = 0
            reprojection_losses = []

            source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            identity_reprojection_losses = []
            for frame_id in self.frame_ids[1:]:
                pred = inputs[("color", frame_id, source_scale)]
                identity_reprojection_losses.append(
                    # if camera does not move, pred and target are the same, so that loss=0
                    self.compute_reprojection_loss(pred, target))

            identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

            # save both images, and do min all at once below
            identity_reprojection_loss = identity_reprojection_losses

            reprojection_loss = reprojection_losses

            # add random numbers to break ties
            identity_reprojection_loss += torch.randn(
                identity_reprojection_loss.shape, device=self.device) * 0.00001

            # [not move, corretlymatch]
            combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)

            to_optimise, idxs = torch.min(combined, dim=1)

            # true means corretly match, false means not move
            outputs["identity_selection/{}".format(scale)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
        total_loss /= 4

        return total_loss

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """

        losses = {}
        total_loss = 0
        total_loss1 = 0
        """
        for scale in self.scales:
            loss = 0
            reprojection_losses = []

            source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            identity_reprojection_losses = []
            for frame_id in self.frame_ids[1:]:
                pred = inputs[("color", frame_id, source_scale)]
                identity_reprojection_losses.append(
                    # if camera does not move, pred and target are the same, so that loss=0
                    self.compute_reprojection_loss(pred, target))

            identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

            # save both images, and do min all at once below
            identity_reprojection_loss = identity_reprojection_losses

            reprojection_loss = reprojection_losses

            # add random numbers to break ties
            identity_reprojection_loss += torch.randn(
                identity_reprojection_loss.shape, device=self.device) * 0.00001

            # [not move, corretlymatch]
            combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)

            to_optimise, idxs = torch.min(combined, dim=1)

            # true means corretly match, false means not move
            outputs["identity_selection/{}".format(scale)] = (
                idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/photometric_{}".format(scale)] = loss
        """
        total_loss1 += self.compute_losses_ori(inputs, outputs)
        losses["loss/photometric_ori{}"] = self.compute_losses_ori(inputs, outputs)

        total_loss1 += self.compute_losses_local(inputs, outputs) * 0.1
        losses["loss/photometric_local{}"] = self.compute_losses_local(inputs, outputs)
        """

        total_loss1 += self.compute_losses_reshuffle(inputs, outputs)
        losses["loss/photometric_reshuffle{}"] = self.compute_losses_reshuffle(inputs, outputs)
        """

        # total_loss1 /= 2
        """




        loss_dc = torch.tensor(0.0).to(self.device)
        loss_dc_local = 0
        for i in range(1):
            disp = outputs[("disp", i)]
            disp = F.interpolate(disp, [self.height, self.width], mode="bilinear", align_corners=False)
            loss_dc_i = 0
            for b in range(self.batch_size):
                disp_local = outputs[("disp_local", i)][b].clone()
                x0 = round(self.width * (inputs[("grid_local")][b, 0, 0, 0].item() - (-1)) / 2.)
                y0 = round(self.height * (inputs[("grid_local")][b, 1, 0, 0].item() - (-1)) / 2.)
                w = round(self.width / inputs[("ratio_local")][b, 0].item())
                h = round(self.height / inputs[("ratio_local")][b, 0].item())
                disp_local = F.interpolate(disp_local.unsqueeze(0), [h, w], mode="bilinear", align_corners=False)
                _, depth_local = disp_to_depth(disp_local, self.min_depth, self.max_depth)
                depth_local *= inputs[("ratio_local")][b, 0]
                _, depth_from_ori = disp_to_depth(disp[b, :, y0:y0 + h, x0:x0 + w].clone().unsqueeze(0), self.min_depth,
                                                  self.max_depth)

                loss_dc_i += self.compute_SI_log_depth_loss(depth_local, depth_from_ori)
            loss_dc_i /= self.batch_size
            loss_dc_local += loss_dc_i
        loss_dc_local /= 1
        losses["loss_dc_local"] = loss_dc_local
        loss_dc += loss_dc_local



        loss_dc_reshuffle = 0
        for i in range(1):
            _, depth = disp_to_depth(outputs[("disp", i)].clone(), self.min_depth, self.max_depth)
            disp_restore = outputs[("disp_reshuffle", i)]
            _, depth_restore = disp_to_depth(disp_restore, self.min_depth, self.max_depth)
            loss_dc_reshuffle += self.compute_SI_log_depth_loss(depth_restore, depth)
        loss_dc_reshuffle /= 1
        losses["loss_dc_reshuffle"] = loss_dc_reshuffle
        loss_dc += loss_dc_reshuffle



        losses["loss_dc"] = loss_dc


        total_loss1 = total_loss1 + 0.01 * loss_dc
        """

        if not self.disable_matcher:
            matcher_loss = 0
            for frame_id in self.frame_ids[1:]:
                # correspondences = self.matcher(inputs[('color', 0, source_scale)], inputs[('color', frame_id, source_scale)])
                correspondences = inputs[('correspondences', 0, frame_id)]
                for scale in self.scales:
                    matcher_loss += compute_matcher_errors_from_correspondences(correspondences,
                                                                                outputs[("sample", frame_id, scale)],
                                                                                self.width, self.height,
                                                                                self.batch_size, self.device,
                                                                                confidence=self.confidence,
                                                                                delta=self.matcher_loss_delta)
            matcher_loss *= self.matcher_loss_alpha
            total_loss += matcher_loss
            losses["loss/matcher"] = matcher_loss

        if self.ratio_consistency and outputs["do_shuffle"]:
            ratio_consistency_loss = 0
            for scale in self.scales:
                # ratio_consistency_loss += torch.abs(outputs[("shuffle_disp", scale)] - outputs[("disp", scale)]).mean()
                ratio_consistency_loss += self.compute_batch_image_shuffle_loss(outputs[("shuffle_disp", scale)],
                                                                                outputs[("disp", scale)],
                                                                                norm=self.ratio_consistency_normalization)
            total_loss += ratio_consistency_loss
            losses["loss/ratio_consistency"] = ratio_consistency_loss
        elif self.ratio_consistency_crop and outputs["do_shuffle"]:
            ratio_consistency_crop_loss = 0
            for scale in self.scales:
                ratio_consistency_crop_loss_tmp = torch.abs(outputs[("shuffle_disp", scale)] - outputs[("disp", scale)])
                if self.ratio_consistency_normalization:
                    ratio_consistency_crop_loss_tmp /= (outputs[("shuffle_disp", scale)] + outputs[("disp", scale)])
                if self.ratio_consistency_scales_normalization:
                    ratio_consistency_crop_loss_tmp /= (2 ** scale)
                ratio_consistency_crop_loss += self.compute_random_batch_image_shuffle_loss(
                    ratio_consistency_crop_loss_tmp, inputs["crop_info"] // 2 ** scale).mean()
            ratio_consistency_crop_loss *= self.weight_ratio_consistency_crop
            total_loss += ratio_consistency_crop_loss
            losses["loss/ratio_consistency_crop"] = ratio_consistency_crop_loss

        if self.geometry_loss:
            geometry_loss = 0
            for scale in self.scales:
                for frame_id in self.frame_ids[1:]:
                    if self.geometry_loss_disp_mode:
                        geometry_loss += torch.abs(outputs[("computed_disp", frame_id, scale)] - outputs[
                            ("sampled_disp", frame_id, scale)]).mean()
                    else:
                        geometry_loss += (torch.abs(outputs[("computed_depth", frame_id, scale)] - outputs[
                            ("sampled_depth", frame_id, scale)]) / (
                                                  outputs[("computed_depth", frame_id, scale)] + outputs[
                                              ("sampled_depth", frame_id, scale)])).mean() / (2 ** scale)
            geometry_loss *= 0.1
            total_loss += geometry_loss
            losses["loss/geometry_loss"] = geometry_loss

        total_loss /= len(self.scales)
        losses["loss"] = total_loss + total_loss1
        return losses

    def compute_random_batch_image_shuffle_loss(self, l1, crop_info):
        b, _, h, w = l1.shape
        x1, y1, x1p, y1p, patch_width, patch_height, _, _ = crop_info
        mask = torch.ones_like(l1)
        min_x1p = max(x1p - 1, 0)
        max_x1p = min(x1p + patch_width + 1, w)
        min_y1p = max(y1p - 1, 0)
        max_y1p = min(y1p + patch_height + 1, h)
        mask[:b // 2, :, min_y1p:min_y1p + 2, min_x1p:max_x1p] = 0  # top
        mask[:b // 2, :, max_y1p - 2:max_y1p, min_x1p:max_x1p] = 0  # botton
        mask[:b // 2, :, min_y1p:max_y1p, min_x1p:min_x1p + 2] = 0  # left
        mask[:b // 2, :, min_y1p:max_y1p, max_x1p - 2:max_x1p] = 0  # right

        min_x1 = max(x1 - 1, 0)
        max_x1 = min(x1 + patch_width + 1, w)
        min_y1 = max(y1 - 1, 0)
        max_y1 = min(y1 + patch_height + 1, h)
        mask[b // 2:, :, min_y1:min_y1 + 2, min_x1:max_x1] = 0  # top
        mask[b // 2:, :, max_y1 - 2:max_y1, min_x1:max_x1] = 0  # botton
        mask[b // 2:, :, min_y1:max_y1, min_x1:min_x1 + 2] = 0  # left
        mask[b // 2:, :, min_y1:max_y1, max_x1 - 2:max_x1] = 0  # right

        return l1 * mask

    def compute_batch_image_shuffle_loss(self, pred, target, norm=False):
        mask = torch.ones_like(pred)
        b, _, h, w = pred.shape
        mask[:, :, :, w // 2 - 1:w // 2 + 1] = 0
        mask[:, :, h // 2 - 1:h // 2 + 1, :] = 0
        if norm:
            return ((torch.abs(pred - target) / (pred + target)) * mask).mean()
        else:
            return torch.abs((pred - target) * mask).mean()

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_SI_log_depth_loss(self, pred, target, mask=None, lamda=0.5):
        # B*1*H*W  ->  B*H*W
        if mask is None:
            mask = torch.ones_like(pred).to(self.device)

        mask = mask[:, 0]
        log_pred = torch.log(pred[:, 0] + 1e-8) * mask
        log_tgt = torch.log(target[:, 0] + 1e-8) * mask

        log_diff = log_pred - log_tgt
        valid_num = mask.sum(1).sum(1) + 1e-8
        log_diff_squre_sum = (log_diff ** 2).sum(1).sum(1)
        log_diff_sum_squre = (log_diff.sum(1).sum(1)) ** 2
        loss = log_diff_squre_sum / valid_num - lamda * log_diff_sum_squre / (valid_num ** 2)

        return loss.mean()

    @torch.no_grad()
    def compute_depth_losses(self, inputs, outputs):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_losses = {}
        depth_gt = inputs["depth_gt"]
        b, c, gt_h, gt_w = depth_gt.shape
        mask = depth_gt > 0

        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = F.interpolate(
            depth_pred, [gt_h, gt_w], mode="bilinear", align_corners=False)
        depth_pred = depth_pred.detach()

        depth_gt_flatten = depth_gt.view(self.batch_size, -1)
        depth_pred_flatten = depth_pred.view(self.batch_size, -1)
        mask_flatten = mask.view(self.batch_size, -1)

        med_gt, _ = torch.masked_fill(depth_gt_flatten, ~mask_flatten, float("nan")).nanmedian(dim=1, keepdim=True)
        med_pred, _ = torch.masked_fill(depth_pred_flatten, ~mask_flatten, float("nan")).nanmedian(dim=1, keepdim=True)

        ratios = med_gt / med_pred
        avg = torch.mean(ratios)
        med = torch.median(ratios)
        std = torch.std(ratios / med)

        depth_losses['ratio/mean'] = np.array(avg.cpu())
        depth_losses['ratio/med'] = np.array(med.cpu())
        depth_losses['ratio/std'] = np.array(std.cpu())

        depth_pred *= ratios[..., None, None]

        depth_pred = depth_pred[mask]
        depth_gt = depth_gt[mask]

        depth_pred = torch.clamp(depth_pred, min=self.min_gt_depth, max=self.max_gt_depth)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            depth_losses[metric] = np.array(depth_errors[i].cpu())

        return depth_losses