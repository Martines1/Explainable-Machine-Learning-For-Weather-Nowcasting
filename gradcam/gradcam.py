from pathlib import Path

from matplotlib import pyplot as plt
from pytorch_grad_cam import GradCAMPlusPlus, HiResCAM
import torch.nn as nn
import torch
import numpy as np
from gradcam.regression_target import RegressionTarget


class GradCam:

    def __init__(self, model, input_array, module=None, cam_class=GradCAMPlusPlus):
        self.model = model
        self.module_name = module
        self.cam_class = cam_class
        if module is None:
            self.module = self._get_last_layer()
        else:
            self.module = self._find_last(module)
        self.input = input_array
        self.cam_algo = self._get_method()

    def _get_method(self):
        algo = self.cam_class(model=self.model, target_layers=[self.module])
        try:
            algo.relu = False
        except AttributeError:
            pass
        return algo

    def _get_last_layer(self):
        for module in reversed(list(self.model.modules())):
            if isinstance(module, nn.Conv2d):
                return module
        return None

    def _find_last(self, selected_module):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) and name == selected_module:
                return module
        return None

    def run(self, target: RegressionTarget):
        with self.cam_algo:
            grayscale_cam = self.cam_algo(input_tensor=self.input,
                                          targets=[target],
                                          eigen_smooth=False,
                                          aug_smooth=False)
            return grayscale_cam[0]

    def _disable_inplace_relu(self):
        for m in self.model.modules():
            if isinstance(m, (nn.ReLU, nn.LeakyReLU)) and getattr(m, "inplace", False):
                m.inplace = False

    def _build_isolated_input(self, x_nchw: torch.Tensor, c) -> torch.Tensor:
        device = x_nchw.device
        baseline = torch.log(torch.tensor(0.01, dtype=x_nchw.dtype, device=device))
        x_iso = torch.empty_like(x_nchw, device=device, dtype=x_nchw.dtype)
        x_iso[:] = baseline
        x_iso[:, c, :, :] = x_nchw[:, c, :, :]
        return x_iso

    def _save_cam(self, cam, name: str, title: str):
        cam = np.asarray(cam, dtype=np.float32)
        h, w = cam.shape

        fig, ax = plt.subplots(figsize=(7, 7))
        im = ax.imshow(
            cam,
            cmap="inferno",
            interpolation="nearest",
            origin="upper",
            extent=(0, w, 0, h)
        )

        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Importance", fontsize=15, fontweight="bold")
        cb.ax.tick_params(labelsize=13)

        ax.set_title(title, fontsize=18, fontweight="bold")
        ax.set_xlabel("X (pixels)", fontsize=15, fontweight="bold")
        ax.set_ylabel("Y (pixels)", fontsize=15, fontweight="bold")

        ax.set_xlim(0, w)
        ax.set_ylim(0, h)
        ax.set_xticks(np.arange(0, w + 1, 200))
        ax.set_yticks(np.arange(0, h + 1, 200))
        ax.tick_params(axis='both', labelsize=13)

        plt.tight_layout()
        Path(f"output/gradcam/{self.module_name}").mkdir(parents=True, exist_ok=True)
        fig.savefig(f"output/gradcam/{self.module_name}/{name}.png", dpi=150)
        plt.close(fig)

    def _save_cams_grid(self, cams: list):
        fig, axes = plt.subplots(2, 2, figsize=(13, 13))
        fig.suptitle(
            f'Isolated Channels Grad-CAM from {self.module_name} layer',
            fontsize=20,
            fontweight="bold"
        )

        axes_flat = axes.flatten()

        for idx, cam in enumerate(cams):
            cam = np.asarray(cam, dtype=np.float32)
            h, w = cam.shape

            ax = axes_flat[idx]
            im = ax.imshow(
                cam,
                cmap="inferno",
                interpolation="nearest",
                origin="upper",
                extent=(0, w, 0, h)
            )

            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Importance", fontsize=13, fontweight="bold")
            cbar.ax.tick_params(labelsize=11)

            ax.set_title(f'Channel {idx + 1}', fontsize=17, fontweight="bold")
            ax.set_xlabel("X (pixels)", fontsize=14, fontweight="bold")
            ax.set_ylabel("Y (pixels)", fontsize=14, fontweight="bold")

            ax.set_xlim(0, w)
            ax.set_ylim(0, h)
            ax.set_xticks(np.arange(0, w + 1, 200))
            ax.set_yticks(np.arange(0, h + 1, 200))
            ax.tick_params(axis='both', labelsize=12)

        plt.tight_layout(rect=(0, 0, 1, 0.96))
        Path(f"output/gradcam/{self.module_name}").mkdir(parents=True, exist_ok=True)
        fig.savefig(f"output/gradcam/{self.module_name}/cam_grid_2x2.png", dpi=150)
        plt.close(fig)

    def _pred_to_rad(self, pred, from_shape=928, to_shape=900):
        if hasattr(pred, "detach"):
            pred = pred.detach().cpu().numpy()
        padding = int((from_shape - to_shape) / 2)
        return pred[padding:padding + to_shape, padding:padding + to_shape].copy()

    def run_isolated_channels(self, target: RegressionTarget, aug_smooth: bool = False) -> list:
        self._disable_inplace_relu()
        cams = []

        for c in range(self.input.size(1)):
            print(f"Running for channel {c + 1}!")
            x_iso = self._build_isolated_input(self.input, c)

            with self._get_method() as cam_algo:
                gray = cam_algo(
                    input_tensor=x_iso,
                    targets=[target],
                    aug_smooth=aug_smooth,
                    eigen_smooth=False
                )

            cams.append(np.asarray(gray[0], dtype=np.float32))

        self._save_cams_grid(cams)
        return cams

    def run_all_channels(self, target: RegressionTarget, aug_smooth=False):
        self._disable_inplace_relu()
        with self._get_method() as cam_algo:
            gray = cam_algo(
                input_tensor=self.input,
                targets=[target],
                aug_smooth=aug_smooth,
                eigen_smooth=False
            )
        cam = np.asarray(gray[0], dtype=np.float32)
        self._save_cam(cam, "cam_", f'Merged Grad-CAM from all channels\n from {self.module_name} layer')
        return cam

    def test_target_functions(self, c, targets, aug_smooth=False):
        self._disable_inplace_relu()

        results = []
        x_iso = self._build_isolated_input(self.input, c)

        for target in targets:
            with self._get_method() as cam_algo:
                gray = cam_algo(
                    input_tensor=x_iso,
                    targets=[target],
                    aug_smooth=aug_smooth,
                    eigen_smooth=False
                )

            cam = np.asarray(gray[0], dtype=np.float32)
            results.append((target, cam))

        fig, axes = plt.subplots(2, 2, figsize=(13, 13))
        fig.suptitle(
            f"Target functions testing on channel {c + 1}\n from layer {self.module_name}",
            fontsize=20,
            fontweight="bold"
        )

        axes_flat = axes.flatten()
        positions = [0, 1, 2, 3]

        for (target, cam), pos in zip(results, positions):
            ax = axes_flat[pos]
            h, w = cam.shape

            im = ax.imshow(
                cam,
                cmap="inferno",
                interpolation="nearest",
                origin="upper",
                extent=(0, w, 0, h)
            )

            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label("Importance", fontsize=13, fontweight="bold")
            cb.ax.tick_params(labelsize=11)

            target_name = getattr(target, "mode", str(target)).upper()
            subtitle_name = target_name

            if getattr(target, "mode", None) == "topk":
                subtitle_name = f"TOPK ({target.k_value})"

            ax.set_title(subtitle_name, fontsize=17, fontweight="bold")
            ax.set_xlabel("X (pixels)", fontsize=14, fontweight="bold")
            ax.set_ylabel("Y (pixels)", fontsize=14, fontweight="bold")

            ax.set_xlim(0, w)
            ax.set_ylim(0, h)
            ax.set_xticks(np.arange(0, w + 1, 200))
            ax.set_yticks(np.arange(0, h + 1, 200))
            ax.tick_params(axis='both', labelsize=12)


        plt.tight_layout(rect=(0, 0, 1, 0.97))
        Path(f"output/gradcam/{self.module_name}").mkdir(parents=True, exist_ok=True)
        fig.savefig(
            f"output/gradcam/{self.module_name}/target_functions_channel_{c + 1}.png",
            dpi=150
        )
        plt.close(fig)

        return [cam for _, cam in results]

