import argparse
import os
import time
import json
import hashlib
from fastapi.responses import JSONResponse
import sys
from fastapi import BackgroundTasks
import shutil

sys.path.append("object_detectors/")


def parse_args():
    parser = argparse.ArgumentParser(description="Backend server for ORNL")
    parser.add_argument("--port", type=int, default=8005, help="Port number")
    parser.add_argument(
        "--eval_every", type=int, default=5, help="Evaluate every N epochs"
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=0.0005, help="Learning rate"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/generic_rtv2/rtdetrv2_r50fmow_frozen_generic.yml",
        help="Path to config file",
    )
    parser.add_argument(
        "--workdir", type=str, default="server_workdir", help="Working directory"
    )
    parser.add_argument("--cuda", type=int, default=0, help="CUDA device number")
    return parser.parse_args()


args = parse_args()
# os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

from fastapi import FastAPI, UploadFile, File, Form, Body
import numpy as np
import torch
import uvicorn
from server_ml_functions.train_or_eval_od_model import (
    train_od_model,
    evaluate_od_model,
    extract_features_od_model,
)
from server_ml_functions.utils import load_annotations_for_ids
from object_detectors.al_methods.FAST.sample_fast_main import sample_FAST_images
from server_ml_functions.get_masks_for_new_ids import (
    extract_gt_masks,
    extract_dense_masks,
)
import random
from pycocotools.coco import COCO


class BackendServer:
    def __init__(self, args):
        self.app = FastAPI()
        self.setup_routes()
        self.output_dir = None
        self.dataset_name = None
        self.eval_every = args.eval_every
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.config = args.config
        self.base_workdir = args.workdir
        self.workdir = None
        self.base_weights = None
        self.cuda_device = args.cuda
        self.dataset_path = "datasets/"

        # New properties for logging and state tracking:
        self.username = None
        self.port = args.port  # used in logging
        self.active_ids = None
        self.labeled_ids = None
        self.available_ids = None

        self.coco_json = None
        self.coco_pret_json = None
        self.coco = None
        self.use_sam_masks = True

        self.num_images = 20

        self.extraction_progress = {"total": 0, "current": 0, "message": ""}

    def log_event(self, event_type, **kwargs):
        """Log an event with a timestamp, event type, and additional details."""
        event = {"timestamp": time.time(), "event_type": event_type, **kwargs}
        os.makedirs("logs", exist_ok=True)
        mode = getattr(self, "mode", "Random")
        log_filename = (
            f"logs/user_activity_log_{self.username}_localhost:{self.port}_{mode}.jsonl"
        )
        with open(log_filename, "a") as f:
            f.write(json.dumps(event) + "\n")

    def setup_routes(self):
        self.app.add_api_route(
            "/extract_prototype_masks", self.extract_prototype_masks, methods=["GET"]
        )
        self.app.add_api_route(
            "/extract_prototype_features",
            self.extract_prototype_features,
            methods=["GET"],
        )
        self.app.add_api_route("/train_model", self.train_model, methods=["POST"])
        self.app.add_api_route("/evaluate_model", self.evaluate_model, methods=["GET"])
        self.app.add_api_route(
            "/get_new_training_ids_smart",
            self.get_new_training_ids_smart,
            methods=["GET"],
        )
        self.app.add_api_route(
            "/get_next_active_image_ids",
            self.get_initial_active_image_ids,
            methods=["POST"],
        )
        self.app.add_api_route(
            "/get_available_objects", self.get_available_objects, methods=["GET"]
        )
        self.app.add_api_route(
            "/check_dataset_status", self.check_dataset_status, methods=["POST"]
        )
        self.app.add_api_route(
            "/prepare_dataset_sync", self.prepare_dataset_sync, methods=["POST"]
        )
        self.app.add_api_route(
            "/upload_dataset_file", self.upload_dataset_file, methods=["POST"]
        )
        self.app.add_api_route(
            "/finalize_dataset_sync", self.finalize_dataset_sync, methods=["POST"]
        )
        self.app.add_api_route(
            "/extraction_progress", self.get_extraction_progress, methods=["GET"]
        )
        self.app.add_api_route(
            "/extract_dense_masks", self.extract_dense_masks_api, methods=["POST"]
        )

    async def get_extraction_progress(self):
        return self.extraction_progress

    def run_extraction(
        self,
        sam_model_type,
        coco_json,
        img_dir,
        output_dir,
        dataset_name,
        pixels_between_points,
        img_size,
    ):
        extract_dense_masks(
            sam_model_type,
            coco_json,
            img_dir,
            output_dir,
            dataset_name,
            pixels_between_points,
            img_size,
            self,
        )
        self.log_event("method_end", method="extract_dense_masks", runtime=time.time())

    async def extract_dense_masks_api(
        self,
        background_tasks: BackgroundTasks,
        pixels_between_points: int = Body(16, embed=True),
    ):
        self.log_event(
            "method_start",
            method="extract_dense_masks",
            details="Starting extract_dense_masks in background",
        )
        dataset_path = os.path.join(self.dataset_path, self.dataset_name)
        coco_json = os.path.join(dataset_path, "annotations.json")
        img_dir = os.path.join(dataset_path, "images")
        output_dir = dataset_path
        self.extraction_progress = {"total": 0, "current": 0, "message": ""}
        background_tasks.add_task(
            self.run_extraction,
            "vit_h",
            coco_json,
            img_dir,
            output_dir,
            self.dataset_name,
            pixels_between_points,
            1024,
        )
        return {"message": "Dense mask extraction started in background."}

    async def check_dataset_status(self, payload: dict = Body(...)):
        dataset_name = payload.get("dataset_name")
        if not dataset_name:
            return JSONResponse(
                content={"error": "Dataset name is required"}, status_code=400
            )
        dataset_path = os.path.join(self.dataset_path, dataset_name)
        if not os.path.exists(dataset_path):
            return {"exists": False}
        try:
            missing_files = []
            different_files = []
            client_hashes = payload.get("file_hashes", {})
            print(client_hashes)
            server_hashes = {}
            for root, _, files in os.walk(dataset_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, dataset_path)
                    with open(file_path, "rb") as f:
                        md5_hash = hashlib.md5(f.read()).hexdigest()
                    server_hashes[rel_path] = md5_hash
            if client_hashes:
                for file_path in client_hashes:
                    if file_path not in server_hashes:
                        missing_files.append(file_path)
                    elif client_hashes[file_path] != server_hashes[file_path]:
                        different_files.append(file_path)
            return {
                "exists": True,
                "missing_files": missing_files,
                "different_files": different_files,
                "server_hashes": server_hashes,
            }
        except Exception as e:
            return JSONResponse(
                content={"error": f"Error checking dataset status: {str(e)}"},
                status_code=500,
            )

    async def prepare_dataset_sync(self, payload: dict = Body(...)):
        dataset_name = payload.get("dataset_name")
        if not dataset_name:
            return JSONResponse(
                content={"error": "Dataset name is required"}, status_code=400
            )
        dataset_path = os.path.join(self.dataset_path, dataset_name)
        images_path = os.path.join(dataset_path, "images")
        try:
            os.makedirs(dataset_path, exist_ok=True)
            os.makedirs(images_path, exist_ok=True)
            return {"status": "ready", "dataset_path": dataset_path}
        except Exception as e:
            return JSONResponse(
                content={"error": f"Error preparing dataset directory: {str(e)}"},
                status_code=500,
            )

    async def upload_dataset_file(
        self,
        file: UploadFile = File(...),
        dataset_name: str = Form(...),
        file_path: str = Form(...),
        md5_hash: str = Form(...),
    ):
        if not dataset_name or not file_path:
            return JSONResponse(
                content={"error": "Dataset name and file path are required"},
                status_code=400,
            )
        dataset_path = os.path.join(self.dataset_path, dataset_name)
        if not os.path.exists(dataset_path):
            return JSONResponse(
                content={
                    "error": "Dataset directory does not exist. Call prepare_dataset_sync first."
                },
                status_code=400,
            )
        full_path = os.path.join(dataset_path, file_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        try:
            contents = await file.read()
            calculated_hash = hashlib.md5(contents).hexdigest()
            if calculated_hash != md5_hash:
                return JSONResponse(
                    content={"error": "File hash mismatch. Upload may be corrupted."},
                    status_code=400,
                )
            with open(full_path, "wb") as f:
                f.write(contents)
            return {"status": "success", "file_path": file_path}
        except Exception as e:
            return JSONResponse(
                content={"error": f"Error saving file: {str(e)}"}, status_code=500
            )

    async def finalize_dataset_sync(self, payload: dict = Body(...)):
        dataset_name = payload.get("dataset_name")
        self.dataset_name = dataset_name
        if not dataset_name:
            return JSONResponse(
                content={"error": "Dataset name is required"}, status_code=400
            )
        dataset_path = os.path.join(self.dataset_path, dataset_name)
        images_path = os.path.join(dataset_path, "images")
        annotations_path = os.path.join(dataset_path, "annotations.json")
        if not os.path.exists(images_path):
            return JSONResponse(
                content={"error": "Dataset is missing 'images' directory"},
                status_code=400,
            )
        if not os.path.exists(annotations_path):
            return JSONResponse(
                content={"error": "Dataset is missing 'annotations.json' file"},
                status_code=400,
            )
        try:
            with open(annotations_path, "r") as f:
                annotations = json.load(f)
            required_keys = ["images", "annotations", "categories"]
            for key in required_keys:
                if key not in annotations:
                    return JSONResponse(
                        content={
                            "error": f"annotations.json is missing required key: {key}"
                        },
                        status_code=400,
                    )
            self.log_event("dataset_sync_completed", dataset_name=dataset_name)
            return {
                "status": "success",
                "message": "Dataset synchronization completed successfully",
            }
        except json.JSONDecodeError:
            return JSONResponse(
                content={"error": "annotations.json is not valid JSON"}, status_code=400
            )
        except Exception as e:
            return JSONResponse(
                content={"error": f"Error finalizing dataset sync: {str(e)}"},
                status_code=500,
            )

    async def get_available_objects(self):
        try:
            objects = []
            if os.path.exists(self.base_workdir):
                for item in os.listdir(self.base_workdir):
                    obj_path = os.path.join(self.base_workdir, item)
                    if os.path.isdir(obj_path):
                        if any(file.endswith(".pth") for file in os.listdir(obj_path)):
                            objects.append(item)
                        elif os.path.exists(
                            os.path.join(obj_path, "train_current.json")
                        ):
                            objects.append(item)
            return {"objects": objects}
        except Exception as e:
            print(f"Error getting available objects: {e}")
            return {"objects": [], "error": str(e)}

    async def get_initial_active_image_ids(self, payload: dict = Body(...)):
        start_time = time.time()
        self.log_event(
            "method_start",
            method="get_initial_active_image_ids",
            details="Starting get_initial_active_image_ids",
        )

        folder = payload.get("folder")
        self.username = payload.get("username")

        object_name = payload.get("object")
        is_new_object = payload.get("is_new_object", False)

        # Create and set object-specific workdir
        if object_name:
            object_folder = object_name.lower().replace(" ", "_")
            object_workdir = os.path.join(self.base_workdir, object_folder)
            os.makedirs(object_workdir, exist_ok=True)
            self.workdir = object_workdir
            self.log_event(
                "object_workdir_set", object=object_name, workdir=object_workdir
            )
            if not is_new_object:

                weights_path = os.path.join(object_workdir, "object_weights.pth")

                if os.path.exists(weights_path):
                    self.base_weights = weights_path
                else:
                    self.base_weights = None
            else:
                self.base_weights = None
        else:
            self.workdir = self.base_workdir
            self.base_weights = None

        # Modified: use the object-specific workdir for output instead of a formatted string.
        self.output_dir = self.workdir

        folder_name = os.path.basename(folder)
        annotation_path = os.path.join(folder, "annotations.json")
        assert os.path.exists(
            annotation_path
        ), f"Annotations file not found: {annotation_path}"
        self.dataset_name = folder_name
        self.num_images = payload.get("num_images", 5)

        if is_new_object:
            pret_annotations = os.path.join(self.workdir, "train_initial.json")
            with open(annotation_path, "r") as src_f:
                coco_data = json.load(src_f)
                empty_train = {
                    "images": [],
                    "annotations": [],
                    "categories": coco_data.get("categories", []),
                }
            os.makedirs(os.path.dirname(pret_annotations), exist_ok=True)
            with open(pret_annotations, "w") as f:
                json.dump(empty_train, f, indent=4)
        else:
            pret_annotations = os.path.join(self.workdir, "train_current.json")
            if not os.path.exists(pret_annotations):
                with open(annotation_path, "r") as src_f:
                    coco_data = json.load(src_f)
                    empty_train = {
                        "images": [],
                        "annotations": [],
                        "categories": coco_data.get("categories", []),
                    }
                os.makedirs(os.path.dirname(pret_annotations), exist_ok=True)
                with open(pret_annotations, "w") as f:
                    json.dump(empty_train, f, indent=4)

        self.coco_json = annotation_path
        self.coco_pret_json = pret_annotations

        img_dir = os.path.join(folder, "images")
        output_dir = os.path.abspath(self.output_dir)
        coco_json = os.path.abspath(self.coco_json)

        weights_file = os.path.abspath(self.base_weights) if self.base_weights else None

        os.makedirs(output_dir, exist_ok=True)

        if weights_file:
            extract_features_od_model(
                self.config,
                output_dir,
                weights_file,
                coco_json,
                img_dir,
                self.cuda_device,
            )
            predictions = os.path.join(self.output_dir, "predictions.json")
        else:
            predictions = os.path.join(self.output_dir, "predictions.json")
            os.makedirs(os.path.dirname(predictions), exist_ok=True)
            with open(predictions, "w") as f:
                json.dump({}, f, indent=4)

        if not os.path.isfile(annotation_path):
            return {"error": "annotations.json not found in folder."}

        coco = COCO(annotation_path)
        self.coco = coco

        img_ids = self.coco.getImgIds()
        coco_pret = COCO(pret_annotations)
        img_ids_pret = coco_pret.getImgIds()
        self.labeled_ids = img_ids_pret
        available_ids = set(img_ids) - set(img_ids_pret)

        if not self.base_weights:
            self.log_event(
                "Random Sampling", object=object_name, is_new_object=is_new_object
            )
            new_img_ids = random.sample(
                available_ids, min(self.num_images, len(available_ids))
            )
            result = {"new_training_ids": new_img_ids, "certain_boxes": {}}
        else:
            self.log_event(
                "Smart Sampling", object=object_name, is_new_object=is_new_object
            )
            mask_dir = os.path.join(self.dataset_path, self.dataset_name)

            sam_boxes_file = os.path.join(
                mask_dir, f"boxes_{folder_name.lower()}_16.npy"
            )
            sam_idx_file = os.path.join(
                mask_dir, f"img_idx_{folder_name.lower()}_16.npy"
            )
            masks_file = os.path.join(mask_dir, f"objects_{folder_name.lower()}_16.npy")
            masks_gt_file = os.path.join(self.workdir, f"objects_al_gt.npy")
            if not all(
                os.path.exists(f)
                for f in [sam_boxes_file, sam_idx_file, masks_file, masks_gt_file]
            ):
                print(
                    "Couldn't find all required files for smart sampling. Using random sampling instead."
                )
                new_img_ids = random.sample(
                    available_ids, min(self.num_images, len(available_ids))
                )
                result = {"new_training_ids": new_img_ids, "certain_boxes": {}}
            else:
                sam_boxes = np.load(sam_boxes_file)
                sam_idx = np.load(sam_idx_file)
                masks = np.load(masks_file)
                masks = torch.from_numpy(masks).long()
                masks_gt = np.load(masks_gt_file)
                masks_gt = torch.from_numpy(masks_gt).long()
                new_img_ids = sample_FAST_images(
                    selected_ids=self.labeled_ids,
                    n_images=self.num_images,
                    last_exp_path=self.output_dir,
                    oracle_coco=self.coco,
                    masks=masks,
                    masks_gt=masks_gt,
                    sam_boxes=sam_boxes,
                    sam_idx=sam_idx,
                    agg_method="max",
                    save_memory=True,
                    device="cuda",
                    distance="euclidean",
                    uncertainty_method="exponential",
                    classifier=None,
                    uncertainty_only=True,
                    alpha=1,
                    beta=0.5,
                    tintra=0.7,
                    tinter=0.3,
                    expand_ratio=4,
                    diversity_method="original",
                    img_size=1024,
                    base_coco_json=self.coco_json,
                )
                result = load_annotations_for_ids(
                    predictions,
                    new_img_ids,
                    sam_boxes,
                    sam_idx,
                    0.5,
                    self.use_sam_masks,
                )

        self.available_ids = available_ids - set(new_img_ids)
        self.active_ids = new_img_ids
        self.labeled_ids = self.labeled_ids + new_img_ids

        end_time = time.time()
        self.log_event(
            "method_end",
            method="get_initial_active_image_ids",
            runtime=end_time - start_time,
            active_ids=self.active_ids,
            labeled_ids=self.labeled_ids,
            available_ids=list(self.available_ids),
            object=object_name,
            is_new_object=is_new_object,
        )
        return result

    async def train_model(self, payload: dict = Body(...)):
        start_time = time.time()
        self.log_event(
            "method_start", method="train_model", details="Starting train_model"
        )
        print("Called /train_model")

        image_annotations = payload.get("image_annotations", {})
        epochs = payload.get("epochs", self.epochs)
        use_sam_masks = payload.get("use_sam_masks", False)
        retrain = payload.get("retrain", False)

        self.use_sam_masks = use_sam_masks

        with open(self.coco_pret_json, "r") as f:
            prev_data = json.load(f)
        with open(self.coco_json, "r") as f:
            gt_data = json.load(f)
        prev_images = prev_data.get("images", [])
        prev_annotations = prev_data.get("annotations", [])
        prev_image_ids = {img["id"] for img in prev_images}
        new_annotations = []
        next_ann_id = max([ann["id"] for ann in prev_annotations], default=0) + 1
        new_image_ids = set()

        for img_id_str, anns in image_annotations.items():
            try:
                img_id = int(img_id_str)
                if img_id in prev_image_ids:
                    continue
                new_image_ids.add(img_id)
                for ann in anns:
                    if ann.get("type") == "positive":
                        new_ann = {
                            "id": next_ann_id,
                            "image_id": img_id,
                            "bbox": [ann["x"], ann["y"], ann["width"], ann["height"]],
                            "category_id": 1,
                            "iscrowd": 0,
                            "area": ann["width"] * ann["height"],
                            "segmentation": [],
                        }
                        new_annotations.append(new_ann)
                        next_ann_id += 1
            except Exception as e:
                print(f"Error processing annotations for {img_id_str}: {e}")

        gt_images = {img["id"]: img for img in gt_data.get("images", [])}
        additional_images = []
        for img_id in new_image_ids:
            if img_id not in prev_image_ids and img_id in gt_images:
                additional_images.append(gt_images[img_id])

        combined_images = prev_images + additional_images
        combined_annotations = prev_annotations + new_annotations
        current_training = {
            "images": combined_images,
            "annotations": combined_annotations,
            "categories": gt_data.get("categories", []),
        }

        print("Number of images:", len(current_training["images"]))

        current_train_coco = os.path.join(self.workdir, "train_current.json")
        os.makedirs(os.path.dirname(current_train_coco), exist_ok=True)
        with open(current_train_coco, "w") as f:
            json.dump(current_training, f, indent=4)

        self.coco_pret_json = current_train_coco

        img_dir = os.path.join(self.dataset_path, self.dataset_name, "images")
        current_train_coco_abs = os.path.abspath(current_train_coco)

        # Modified: Use a common "exp" folder rather than separate folders per iteration.
        exp_dir = os.path.abspath(os.path.join(self.output_dir, "exp"))
        os.makedirs(exp_dir, exist_ok=True)

        if self.base_weights and not retrain:
            current_weights = os.path.abspath(self.base_weights)
        else:
            current_weights = None  # Replace with default model path if available

        print("#" * 20)
        print("Current weights:", current_weights)
        print("#" * 20)

        train_od_model(
            self.config,
            exp_dir,
            current_train_coco_abs,
            img_dir,
            self.cuda_device,
            self.eval_every,
            epochs,
            self.batch_size,
            self.batch_size,
            self.learning_rate,
            current_weights,
        )

        best_weights_path = os.path.join(exp_dir, "best.pth")

        # After training, copy best weights to the object folder and remove the exp folder.
        if best_weights_path and os.path.exists(best_weights_path):
            object_weights_path = os.path.join(self.workdir, f"object_weights.pth")
            try:
                shutil.copy(best_weights_path, object_weights_path)
                self.base_weights = object_weights_path
                print(f"Saved weights to {object_weights_path}")
                # Delete the exp folder to save space
                shutil.rmtree(exp_dir)
            except Exception as e:
                print(f"Error saving weights: {e}")
                self.base_weights = best_weights_path

        end_time = time.time()
        self.log_event(
            "method_end",
            method="train_model",
            runtime=end_time - start_time,
            weights_path=self.base_weights,
        )
        return {"message": "Train model completed", "weights_path": self.base_weights}

    async def evaluate_model(self):
        start_time = time.time()
        self.log_event(
            "method_start", method="evaluate_model", details="Starting evaluate_model"
        )
        print("Called /evaluate_model")
        current_output_dir = os.path.join(self.output_dir, "exp")

        current_train_coco = os.path.join(self.workdir, "train_current.json")
        img_dir = os.path.join(self.dataset_path, self.dataset_name, "images")

        evaluate_od_model(
            self.config,
            current_output_dir,
            self.cuda_device,
            current_train_coco,
            img_dir,
            self.base_weights,
        )

        results_file = os.path.join(self.output_dir, "eval.pth")
        print("Results file:", results_file, os.path.exists(results_file))
        if not os.path.exists(results_file):
            return {"error": "Evaluation results not found."}
        results = torch.load(results_file, weights_only=False)
        precision = results["precision"]
        ap50_per_class = precision[0, :, :, 0, 2].mean(axis=0)
        ap75_per_class = precision[5, :, :, 0, 2].mean(axis=0)
        ap50_all = ap50_per_class.mean()
        ap75_all = ap75_per_class.mean()
        coco_results = {
            "AP50_per_class": ap50_per_class.tolist(),
            "AP75_per_class": ap75_per_class.tolist(),
            "AP50_all": ap50_all.item(),
            "AP75_all": ap75_all.item(),
        }

        if os.path.exists(current_output_dir):
            shutil.rmtree(current_output_dir)

        end_time = time.time()
        self.log_event(
            "method_end",
            method="evaluate_model",
            runtime=end_time - start_time,
            coco_results=coco_results,
        )
        return coco_results

    async def extract_prototype_masks(self):
        start_time = time.time()
        self.log_event(
            "method_start",
            method="extract_prototype_masks",
            details="Starting extract_prototype_masks",
        )
        print("Called /extract_prototype_masks")
        img_dir = os.path.join(self.dataset_path, self.dataset_name, "images")
        coco_json = self.coco_pret_json

        extract_gt_masks(
            "vit_h", coco_json, img_dir, self.output_dir, self.dataset_name
        )
        masks = np.load(os.path.join(self.output_dir, "objects_al_gt.npy"))
        print("Masks shape:", masks.shape)
        end_time = time.time()
        self.log_event(
            "method_end",
            method="extract_prototype_masks",
            runtime=end_time - start_time,
        )
        return {"message": "Extract prototype masks"}

    async def extract_prototype_features(self):
        start_time = time.time()
        self.log_event(
            "method_start",
            method="extract_prototype_features",
            details="Starting extract_prototype_features",
        )
        print("Called /extract_prototype_features")
        img_dir = os.path.join(self.dataset_path, self.dataset_name, "images")
        exp_dir = os.path.abspath(os.path.join(self.output_dir, "exp"))

        extract_features_od_model(
            self.config,
            exp_dir,
            self.base_weights,
            self.coco_json,
            img_dir,
            self.cuda_device,
        )

        current_predictions = os.path.join(self.output_dir, "predictions.json")
        current_features = os.path.join(self.output_dir, "features_256_2.npy")

        shutil.copy(os.path.join(exp_dir, "predictions.json"), current_predictions)
        shutil.copy(os.path.join(exp_dir, "features_256_2.npy"), current_features)

        shutil.rmtree(exp_dir)
        end_time = time.time()
        self.log_event(
            "method_end",
            method="extract_prototype_features",
            runtime=end_time - start_time,
        )
        return {"message": "Extract prototype features"}

    async def get_new_training_ids_smart(self):
        start_time = time.time()
        self.log_event(
            "method_start",
            method="get_new_training_ids_smart",
            details="Starting get_new_training_ids_smart",
        )
        print("Called /get_new_training_ids")
        predictions = os.path.join(self.output_dir, "predictions.json")
        mask_dir = os.path.join(self.dataset_path, self.dataset_name)
        sam_boxes_file = os.path.join(
            mask_dir, f"boxes_{self.dataset_name.lower()}_16.npy"
        )
        sam_idx_file = os.path.join(
            mask_dir, f"img_idx_{self.dataset_name.lower()}_16.npy"
        )
        masks_file = os.path.join(
            mask_dir, f"objects_{self.dataset_name.lower()}_16.npy"
        )
        current_masks_gt_file = os.path.join(self.output_dir, "objects_al_gt.npy")
        sam_boxes = np.load(sam_boxes_file)
        sam_idx = np.load(sam_idx_file)
        masks = np.load(masks_file)
        masks = torch.from_numpy(masks).long()
        masks_gt = np.load(current_masks_gt_file)
        masks_gt = torch.from_numpy(masks_gt).long()
        new_img_ids = sample_FAST_images(
            selected_ids=self.labeled_ids,
            n_images=self.num_images,
            last_exp_path=self.output_dir,
            oracle_coco=self.coco,
            masks=masks,
            masks_gt=masks_gt,
            sam_boxes=sam_boxes,
            sam_idx=sam_idx,
            agg_method="max",
            save_memory=True,
            device="cuda",
            distance="euclidean",
            uncertainty_method="exponential",
            classifier=None,
            uncertainty_only=True,
            alpha=1,
            beta=0.5,
            tintra=0.7,
            tinter=0.3,
            expand_ratio=4,
            diversity_method="original",
            img_size=1024,
            base_coco_json=self.coco_json,
        )
        output = load_annotations_for_ids(
            predictions, new_img_ids, sam_boxes, sam_idx, 0.5, self.use_sam_masks
        )

        self.active_ids = new_img_ids
        self.labeled_ids = self.labeled_ids + new_img_ids
        self.available_ids = self.available_ids - set(new_img_ids)
        end_time = time.time()
        self.log_event(
            "method_end",
            method="get_new_training_ids_smart",
            runtime=end_time - start_time,
            active_ids=self.active_ids,
            labeled_ids=self.labeled_ids,
            available_ids=list(self.available_ids),
        )
        return output

    def run(self, port: int):
        uvicorn.run(self.app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    server = BackendServer(args)
    server.run(args.port)
