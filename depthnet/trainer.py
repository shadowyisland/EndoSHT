import os
from pathlib import Path
from datetime import datetime
import torch
import torch.nn.functional as F
import time
from tensorboardX import SummaryWriter

from .utils import *
from .datasets.gastro_dataset import get_data_loaders
from . import meters


class Trainer():
    def __init__(self, cfgs, model):
        self.cfgs = cfgs
        self.log_path = Path(cfgs.get('log_dir', 'results')) / cfgs.get('model_name',
                                                                        datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
        self.device = cfgs.get('device', 'cpu')
        self.num_epochs = cfgs.get('num_epochs', 30)
        self.batch_size = cfgs.get('batch_size', 64)
        self.save_checkpoint_freq = cfgs.get('save_checkpoint_freq', 1)
        self.keep_num_checkpoint = cfgs.get('keep_num_checkpoint', 2)  # -1 for keeping all checkpoints
        self.start_epoch = cfgs.get('start_epoch', 0)
        self.load_weights_folder = cfgs.get('load_weights_folder', None)
        self.mypretrain = cfgs.get('mypretrain', None)
        self.use_logger = cfgs.get('use_logger', True)
        self.log_freq = cfgs.get('log_freq', 1000)
        self.run_val = cfgs.get('run_val', True)
        self.run_test = cfgs.get('run_test', False)
        # geometry loss
        self.geometry_loss = cfgs.get('geometry_loss', False)
        self.supervised = cfgs.get('supervised', False)

        # --- 新增早停相关配置 ---
        self.early_stop_patience = cfgs.get('early_stop_patience', 5)  # 容忍度
        self.early_stop_metric = cfgs.get('early_stop_metric', 'de/abs_rel')  # 监控指标
        self.best_val_metric = float('inf')
        self.patience_counter = 0
        self.best_epoch = -1

        self.model = model(cfgs)
        self.model.trainer = self

        data_loaders = get_data_loaders(cfgs)
        self.train_loader = data_loaders["train_loader"]
        if self.run_val:
            self.val_loader = data_loaders["val_loader"]
            self.val_iter = iter(self.val_loader)
        if self.run_test:
            self.test_loader = data_loaders["test_loader"]
        info_dict = data_loaders["info_dict"]

        cfgs.update(info_dict)
        self.num_total_steps = info_dict['num_total_steps']
        if self.start_epoch > 0:
            self.num_total_steps -= self.num_total_steps / self.num_epochs * self.start_epoch
        self.metrics_trace = meters.MetricsTrace()
        self.metric_str_exclude = cfgs.get('metric_str_exclude', [])
        self.make_metrics = lambda m=None: meters.StandardMetrics(m, self.metric_str_exclude)

    def load_checkpoint(self, custom_path=None):
        """支持加载指定路径或默认配置路径的权重"""
        if custom_path is not None:
            # 临时修改模型中的加载路径并加载
            original_folder = self.model.load_weights_folder
            self.model.load_weights_folder = custom_path
            self.model.load_model()
            self.model.load_weights_folder = original_folder
            return

        if self.mypretrain is not None:
            self.model.load_pretrain()

        if self.load_weights_folder is not None:
            self.model.load_model()

    def save_checkpoint(self, is_best=False):
        """保存模型权重"""
        folder_name = "weights_best" if is_best else "weights_{}".format(self.epoch)
        save_folder = os.path.join(self.log_path, "models", folder_name)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        if not is_best and self.epoch + 1 == self.num_epochs:
            last_folder = os.path.join(self.log_path, "models", "weights_last")
            if os.path.exists(last_folder): os.remove(last_folder)
            os.symlink(folder_name, last_folder)

        for net_name in self.model.network_names:
            save_path = os.path.join(save_folder, "{}.pth".format(net_name))
            to_save = getattr(self.model, net_name).state_dict()
            if net_name == "net_depth_encoder":
                to_save['height'] = self.model.height
                to_save['width'] = self.model.width
            torch.save(to_save, save_path)

        for optim_name in self.model.optimizer_names:
            save_path = os.path.join(save_folder, "{}.pth".format(optim_name))
            torch.save(getattr(self.model, optim_name).state_dict(), save_path)

    def train(self):
        """主训练流程"""
        self.metrics_trace.reset()
        self.start_time = time.time()
        self.train_iter_per_epoch = len(self.train_loader)
        if self.run_val:
            self.val_iter_per_epoch = len(self.val_loader)
        if self.run_test:
            self.test_iter_per_epoch = len(self.test_loader)

        self.model.to_device(self.device)
        self.model.init_optimizers()

        self.load_checkpoint()

        self.writers = {}
        for mode in ["train", "val", "test"]:
            self.writers[mode] = SummaryWriter(Path(self.log_path) / "logs" / mode)

        print("Training model named:\n  ", self.model.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.log_path)
        print("Training is using:\n  ", self.device)

        print(f"{self.model.model_name}: optimizing to {self.num_epochs} epochs")

        if self.start_epoch > 0:
            for _ in range(self.start_epoch):
                for scheduler_name in self.model.scheduler_names:
                    getattr(self.model, scheduler_name).step()

        for self.epoch in range(self.start_epoch, self.num_epochs):
            epoch_start_time = time.time()

            # --- 1. 训练阶段 ---
            metrics_train = self.run_epoch(self.train_loader, self.epoch, is_train=True)
            self.metrics_trace.append("train", metrics_train)

            # --- 2. 验证阶段 ---
            if self.run_val:
                with torch.no_grad():
                    metrics_val = self.run_epoch(self.val_loader, self.epoch, is_train=False, is_val=True)
                    self.metrics_trace.append("val", metrics_val)

                # 早停逻辑检查
                val_stats = metrics_val.get_data_dict()
                current_val_metric = val_stats.get(self.early_stop_metric, val_stats.get('loss', 0))

                if current_val_metric < self.best_val_metric:
                    self.best_val_metric = current_val_metric
                    self.best_epoch = self.epoch
                    self.patience_counter = 0
                    self.save_checkpoint(is_best=True)
                    print(f"✨ 发现最佳模型! ({self.early_stop_metric}: {self.best_val_metric:.6f})")
                else:
                    self.patience_counter += 1
                    print(f"⚠️  验证指标未改善 (耐心计数: {self.patience_counter}/{self.early_stop_patience})")

            # --- 3. 测试阶段 ---
            if self.run_test:
                with torch.no_grad():
                    metrics_test = self.run_epoch(self.test_loader, self.epoch, is_train=False, is_test=True)
                    self.metrics_trace.append("test", metrics_test)

            # 定期保存
            if (self.epoch + 1) % self.save_checkpoint_freq == 0 or (self.epoch + 1) == self.num_epochs:
                self.save_checkpoint()

            self.metrics_trace.save(os.path.join(self.log_path, 'metrics.json'))

            epoch_duration = time.time() - epoch_start_time
            print(f"✅ Epoch {self.epoch} duration: {epoch_duration:.2f} seconds.\n")

            # --- 4. 触发早停 ---
            if self.patience_counter >= self.early_stop_patience:
                print(f"🚨 早停触发！在epoch {self.epoch} 停止训练。")
                print(f"最佳验证指标: {self.best_val_metric:.6f} (epoch {self.best_epoch})")

                # 恢复最佳权重
                best_weights_path = os.path.join(self.log_path, "models", "weights_best")
                print(f"恢复最佳模型权重 (epoch {self.best_epoch})...")
                self.load_checkpoint(custom_path=best_weights_path)

                for net_name in self.model.network_names:
                    print(f"✅ 恢复 {net_name} 权重")
                print("🎯 最佳模型权重已恢复")
                print(f"🎯 训练因早停而提前结束于epoch {self.epoch + 1}")
                break

        print(f"Training completed after {self.epoch + 1} epochs.")

    def run_epoch(self, loader, epoch=0, is_train=True, is_val=False, is_test=False):
        """运行单个 Epoch"""
        metrics = self.make_metrics()

        if is_train:
            self.model.set_train()
            mode = "TRAIN"
            iter_per_epoch = self.train_iter_per_epoch
            for scheduler_name in self.model.scheduler_names:
                getattr(self.model, scheduler_name).step()
        elif is_val:
            self.model.set_eval()
            mode = "VAL"
            iter_per_epoch = self.val_iter_per_epoch
        elif is_test:
            self.model.set_eval()
            mode = "TEST"
            iter_per_epoch = self.test_iter_per_epoch
        else:
            raise NotImplementedError

        for batch_idx, inputs in enumerate(loader):
            before_op_time = time.time()

            outputs, losses = self.model.forward(inputs)

            if is_train:
                self.model.backward(losses)

            duration = time.time() - before_op_time
            metrics.update(losses, self.batch_size)

            # 处理指标更新 (如果有GT深度)
            if "depth_gt" in inputs:
                depth_losses = self.model.compute_depth_losses(inputs, outputs)
                metrics.update(depth_losses, self.batch_size)
                losses.update(depth_losses)

            # 日志记录 (Tensorboard)
            total_iter = batch_idx + epoch * iter_per_epoch
            if self.use_logger and (total_iter % self.log_freq == 0):
                if is_train:
                    iter_sofar = batch_idx + (epoch - self.start_epoch) * self.train_iter_per_epoch
                    self.log_time(batch_idx, duration, losses["loss"].cpu().data, iter_sofar)
                self.log(mode.lower(), inputs, outputs, losses, total_iter)

            # --- 实时后台打印 ---
            if batch_idx % 10 == 0 or batch_idx == iter_per_epoch - 1:
                fps = self.batch_size / duration if duration > 0 else 0
                loss_str = " ".join(
                    [f"{k}: {v:.5f}" for k, v in losses.items() if "loss" in k or "de/" in k or "da/" in k])
                print(f"📊 [{mode:^6}] 第{epoch}个epoch | 进度: {batch_idx:04d}/{iter_per_epoch:04d} | "
                      f"耗时: {duration:.2f}s | {fps:7.1f}Hz      {loss_str}")

        # --- Epoch 结束时的总结打印 ---
        final_metrics = metrics.get_data_dict()
        if self.use_logger:
            for k, v in final_metrics.items():
                self.writers[mode.lower()].add_scalar(f'Metrics/{k}', v, epoch)

        print("=" * 60)
        print(f"📈 [{mode} 指标] 第{epoch}个epoch：")
        print("─" * 60)
        for k, v in final_metrics.items():
            print(f"  {k:<25} :   {v:.6f}")
        print("=" * 60)

        return metrics

    def log(self, mode, inputs, outputs, losses, step):
        """写入 Tensorboard 事件"""
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, step)

        for j in range(min(4, self.batch_size)):
            for s in self.model.scales:
                for frame_id in self.model.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, step)

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), step)

                if not self.model.disable_automasking:
                    if "identity_selection/{}".format(s) in outputs:
                        writer.add_image(
                            "automask_{}/{}".format(s, j),
                            outputs["identity_selection/{}".format(s)][j][None, ...], step)

                if self.geometry_loss:
                    for frame_id in self.model.frame_ids[1:]:
                        if ("computed_depth", frame_id, s) in outputs:
                            writer.add_image(
                                "computed_depth_{}_{}/{}".format(frame_id, s, j),
                                normalize_image(outputs[("computed_depth", frame_id, s)][j]), step)
                            writer.add_image(
                                "sampled_depth_{}_{}/{}".format(frame_id, s, j),
                                normalize_image(outputs[("sampled_depth", frame_id, s)][j]), step)

    def log_time(self, batch_idx, duration, loss, step):
        """打印标准时间日志（可选保留）"""
        samples_per_sec = self.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (self.num_total_steps / step - 1.0) * time_sofar if step > 0 else 0

        lr_d = self.model.optimizer_depth.param_groups[0]['lr']
        if self.supervised:
            print_string = "epoch {:>3} | lr {:.6f} | batch {:>6} | examples/s: {:5.1f} | loss: {:.5f}"
            print(print_string.format(self.epoch, lr_d, batch_idx, samples_per_sec, loss))
        else:
            lr_p = self.model.optimizer_pose.param_groups[0]['lr']
            print_string = "epoch {:>3} | lr_d {:.6f} | lr_p {:.6f} | batch {:>6} | loss: {:.5f}"
            print(print_string.format(self.epoch, lr_d, lr_p, batch_idx, loss))