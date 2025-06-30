import os
import subprocess


def train_od_model(
    config,
    output_dir_exp,
    new_coco_path,
    base_img_dir,
    cuda_device,
    eval_every=10,
    epochs=72,
    batch_size=2,
    batch_size_val=2,
    learning_rate=0.0001,
    resume_path=None,
):
    # Create a copy of the current environment and add the CUDA_VISIBLE_DEVICES variable.
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)

    base_img_dir = os.path.abspath(base_img_dir)
    new_coco_path = os.path.abspath(new_coco_path)
    if resume_path is not None:
        resume_path = os.path.abspath(resume_path)

    # Build the command as a list.
    command = [
        "python",
        "tools/train.py",
        "-c",
        config,
        "-u",
        f"train_dataloader.dataset.ann_file={new_coco_path}",
        f"epoches={epochs}",
        f"train_dataloader.total_batch_size={batch_size}",
        f"val_dataloader.total_batch_size={batch_size_val}",
        f"train_dataloader.dataset.img_folder={base_img_dir}",
        f"val_dataloader.dataset.ann_file={new_coco_path}",
        f"val_dataloader.dataset.img_folder={base_img_dir}",
        f"optimizer.lr={learning_rate}",
        "lr_warmup_scheduler=False",
        f"eval_every={eval_every}",
        "--addon",
        "al",
    ]
    if resume_path is not None:
        command.extend(["--tuning", resume_path])
    command.extend(["--use-amp", "--output-dir", output_dir_exp])

    # Run the command in the target directory.
    result = subprocess.run(
        command, env=env, cwd="object_detectors/RT-DETR/rtdetrv2_pytorch"
    )
    return result.returncode  # You can also return the entire result object if needed.


def evaluate_od_model(
    config, output_dir_exp, cuda_device, new_coco_path, base_img_dir, weights_file
):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)

    base_img_dir = os.path.abspath(base_img_dir)
    new_coco_path = os.path.abspath(new_coco_path)

    weights_file = os.path.abspath(weights_file)

    command = [
        "python",
        "tools/train.py",
        "-c",
        config,
        "--addon",
        "al",
        "--output-dir",
        output_dir_exp,
        "--test-only",
        "--use-amp",
        "--resume",
        weights_file,
        "-u",
        f"val_dataloader.dataset.ann_file={new_coco_path}",
        f"val_dataloader.dataset.img_folder={base_img_dir}",
    ]

    result = subprocess.run(
        command, env=env, cwd="object_detectors/RT-DETR/rtdetrv2_pytorch"
    )
    return result.returncode


def extract_features_od_model(
    config,
    output_dir_exp,
    weights_file,
    base_coco_json,
    base_img_dir,
    cuda_device,
    extract_features=True,
):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)

    # Use the appropriate save-results option based on the extract_features flag.
    save_results_option = "full" if extract_features else "simple"

    base_img_dir = os.path.abspath(base_img_dir)
    base_coco_json = os.path.abspath(base_coco_json)

    weights_file = os.path.abspath(weights_file)

    command = [
        "python",
        "tools/train.py",
        "-c",
        config,
        "--addon",
        "al",
        "--output-dir",
        output_dir_exp,
        "--save-results",
        save_results_option,
        "--test-only",
        "--use-amp",
        "--resume",
        weights_file,
        "-u",
        f"val_dataloader.dataset.ann_file={base_coco_json}",
        f"val_dataloader.dataset.img_folder={base_img_dir}",
        "val_dataloader.total_batch_size=16",
    ]

    result = subprocess.run(
        command, env=env, cwd="object_detectors/RT-DETR/rtdetrv2_pytorch"
    )
    return result.returncode
