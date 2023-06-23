import argparse, os

from icevision.all import *
from PIL import Image
import wandb
import pandas as pd
from fastai.callback.wandb import WandbCallback
from fastai.torch_core import set_seed
from fastai.callback.tracker import SaveModelCallback

import params

train_config = SimpleNamespace(
    framework="fastai",
    img_size=384,
    batch_size=16,
    augment=True,  # use data augmentation
    epochs=5,
    lr=1.45e-3,
    arch=0,
    pretrained=True,  # whether to use pretrained encoder
    seed=42,
)


def parse_args():
    "Overriding default argments"
    argparser = argparse.ArgumentParser(description="Process hyper-parameters")
    argparser.add_argument(
        "--img_size", type=int, default=train_config.img_size, help="image size"
    )
    argparser.add_argument(
        "--batch_size", type=int, default=train_config.batch_size, help="batch size"
    )
    argparser.add_argument(
        "--epochs",
        type=int,
        default=train_config.epochs,
        help="number of training epochs",
    )
    argparser.add_argument(
        "--lr", type=float, default=train_config.lr, help="learning rate"
    )
    argparser.add_argument(
        "--arch",
        type=int,
        default=train_config.arch,
        help="timm backbone architecture",
    )
    argparser.add_argument(
        "--augment",
        type=bool,
        default=train_config.augment,
        help="Use image augmentation",
    )
    argparser.add_argument(
        "--seed", type=int, default=train_config.seed, help="random seed"
    )
    # argparser.add_argument(
    #     "--log_preds",
    #     type=bool,
    #     default=train_config.log_preds,
    #     help="log model predictions",
    # )
    argparser.add_argument(
        "--pretrained",
        type=bool,
        default=train_config.pretrained,
        help="Use pretrained model",
    )

    args = argparser.parse_args()
    vars(train_config).update(vars(args))
    return


def download_data():
    processed_data_at = wandb.use_artifact(f"{params.PROCESSED_DATA_AT}:latest")
    processed_dataset_dir = Path(processed_data_at.download())
    return processed_dataset_dir


template_record = ObjectDetectionRecord()


class CustomCocoParser(Parser):
    def __init__(self, template_record, data_dir):
        super().__init__(template_record=template_record)
        self.data_dir = data_dir
        annot_dict = json.load(open(data_dir / "train_sample.json"))
        df = pd.DataFrame(annot_dict["annotations"])
        images_corr = pd.DataFrame(annot_dict["images"])
        df = df.merge(images_corr, how="left", left_on="image_id", right_on="id")
        df.drop(columns="id", inplace=True)
        self.df = df
        self.add_size()
        classes = annot_dict["categories"]
        class_map = {c["id"]: c["name"] for c in classes}
        self.df["category"] = self.df["category_id"].replace(class_map)
        idx_map = {c["id"]: i + 1 for i, c in enumerate(classes)}
        self.df["category_id"] = self.df["category_id"].replace(idx_map)
        classes = [c["name"] for c in classes]
        self.class_map = ClassMap(classes)

    def __iter__(self) -> Any:
        for o in self.df.itertuples():
            yield o

    def __len__(self) -> int:
        return len(self.df)

    def add_size(self):
        image_height = []
        image_width = []
        for i in self.df.file_name:
            image = Image.open(self.data_dir / "images" / i)
            width, height = image.size
            image_height.append(height)
            image_width.append(width)
        self.df["height"] = image_height
        self.df["width"] = image_width

    def record_id(self, o) -> Hashable:
        return o.file_name

    def parse_fields(self, o, record, is_new):
        if is_new:
            record.set_filepath(self.data_dir / "images" / o.file_name)
            record.set_img_size(ImgSize(width=o.width, height=o.height))
            record.detection.set_class_map(self.class_map)

        record.detection.add_bboxes(
            [BBox.from_xywh(o.bbox[0], o.bbox[1], o.bbox[2], o.bbox[3])]
        )
        record.detection.add_labels([o.category])


def get_data(data_dir, bs=4, image_size=384, augment=True):
    parser = CustomCocoParser(template_record=template_record, data_dir=data_dir)
    # Build Fixed Splitter
    splits = pd.read_csv(data_dir / "data_split.csv")
    split_train = splits[splits["Stage"] == "train"]["File_Name"].tolist()
    split_val = splits[splits["Stage"] == "val"]["File_Name"].tolist()
    split_test = splits[splits["Stage"] == "test"]["File_Name"].tolist()

    splitter_list = []
    splitter_list.append(split_train)
    splitter_list.append(split_val)
    splitter_list.append(split_test)

    splitter = FixedSplitter(splitter_list)

    # Get records
    train_records, valid_records, test_records = parser.parse(data_splitter=splitter)

    # Transforms
    # size is set to 384 because EfficientDet requires its inputs to be divisible by 128
    train_tfms = tfms.A.Adapter(
        [*tfms.A.aug_tfms(size=image_size, presize=512), tfms.A.Normalize()]
    )
    valid_tfms = tfms.A.Adapter(
        [*tfms.A.resize_and_pad(image_size), tfms.A.Normalize()]
    )

    # Datasets
    train_ds = Dataset(train_records, train_tfms)
    valid_ds = Dataset(valid_records, valid_tfms)
    test_ds = Dataset(test_records, valid_tfms)
    return train_ds, valid_ds, parser, test_ds


def select_model(selection, image_size):
    extra_args = {}

    if selection == 0:
        model_type = models.mmdet.vfnet
        backbone = model_type.backbones.resnet50_fpn_mstrain_2x

    if selection == 1:
        model_type = models.mmdet.retinanet
        backbone = model_type.backbones.resnet50_fpn_1x
        # extra_args['cfg_options'] = {
        #   'model.bbox_head.loss_bbox.loss_weight': 2,
        #   'model.bbox_head.loss_cls.loss_weight': 0.8,
        #    }

    if selection == 2:
        model_type = models.mmdet.faster_rcnn
        backbone = model_type.backbones.resnet101_fpn_2x
        # extra_args['cfg_options'] = {
        #   'model.roi_head.bbox_head.loss_bbox.loss_weight': 2,
        #   'model.roi_head.bbox_head.loss_cls.loss_weight': 0.8,
        #    }

    if selection == 3:
        model_type = models.mmdet.ssd
        backbone = model_type.backbones.ssd300

    if selection == 4:
        model_type = models.mmdet.yolox
        backbone = model_type.backbones.yolox_s_8x8

    if selection == 5:
        model_type = models.mmdet.yolof
        backbone = model_type.backbones.yolof_r50_c5_8x8_1x_coco

    if selection == 6:
        model_type = models.mmdet.detr
        backbone = model_type.backbones.r50_8x2_150e_coco

    if selection == 7:
        model_type = models.mmdet.deformable_detr
        backbone = model_type.backbones.twostage_refine_r50_16x2_50e_coco

    if selection == 8:
        model_type = models.mmdet.fsaf
        backbone = model_type.backbones.x101_64x4d_fpn_1x_coco

    if selection == 9:
        model_type = models.mmdet.sabl
        backbone = model_type.backbones.r101_fpn_gn_2x_ms_640_800_coco

    if selection == 10:
        model_type = models.mmdet.centripetalnet
        backbone = model_type.backbones.hourglass104_mstest_16x6_210e_coco

    elif selection == 11:
        # The Retinanet model is also implemented in the torchvision library
        model_type = models.torchvision.retinanet
        backbone = model_type.backbones.resnet50_fpn

    elif selection == 12:
        model_type = models.ross.efficientdet
        backbone = model_type.backbones.tf_lite0
        # The efficientdet model requires an img_size parameter
        extra_args["img_size"] = image_size

    elif selection == 13:
        model_type = models.ultralytics.yolov5
        backbone = model_type.backbones.small
        # The yolov5 model requires an img_size parameter
        extra_args["img_size"] = image_size

    return model_type, backbone, extra_args


class COCOMetric_perclass(COCOMetric):
    def finalize(self) -> Dict[str, float]:
        with CaptureStdout():
            coco_eval = create_coco_eval(
                records=self._records,
                preds=self._preds,
                metric_type=self.metric_type.value,
                iou_thresholds=self.iou_thresholds,
                show_pbar=self.show_pbar,
            )
            coco_eval.params.catIds = self.class_ids  # <== Add this row!!!!!!!!!!
            coco_eval.evaluate()
            coco_eval.accumulate()

        with CaptureStdout(propagate_stdout=self.print_summary):
            coco_eval.summarize()
        stats = coco_eval.stats
        logs = {
            "AP (IoU=0.50:0.95) area=all": stats[0],
            "AP (IoU=0.50) area=all": stats[1],
            "AP (IoU=0.75) area=all": stats[2],
            "AP (IoU=0.50:0.95) area=small": stats[3],
            "AP (IoU=0.50:0.95) area=medium": stats[4],
            "AP (IoU=0.50:0.95) area=large": stats[5],
            "AR (IoU=0.50:0.95) area=all maxDets=1": stats[6],
            "AR (IoU=0.50:0.95) area=all maxDets=10": stats[7],
            "AR (IoU=0.50:0.95) area=all maxDets=100": stats[8],
            "AR (IoU=0.50:0.95) area=small maxDets=100": stats[9],
            "AR (IoU=0.50:0.95) area=medium maxDets=100": stats[10],
            "AR (IoU=0.50:0.95) area=large maxDets=100": stats[11],
        }
        self._reset()
        return logs


class COCOMetric_Chair(COCOMetric_perclass):
    class_ids = [1]


class COCOMetric_Couch(COCOMetric_perclass):
    class_ids = [2]


class COCOMetric_TV(COCOMetric_perclass):
    class_ids = [3]


class COCOMetric_Remote(COCOMetric_perclass):
    class_ids = [4]


class COCOMetric_Book(COCOMetric_perclass):
    class_ids = [5]


class COCOMetric_Vase(COCOMetric_perclass):
    class_ids = [6]


def train(config, processed_dataset_dir=None):
    set_seed(config.seed, reproducible=True)
    run = wandb.init(
        project=params.WANDB_PROJECT,
        entity=params.ENTITY,
        job_type="training",
        config=config,
    )

    config = wandb.config

    if processed_dataset_dir is None:
        processed_dataset_dir = download_data()

    train_ds, valid_ds, parser, _ = get_data(
        processed_dataset_dir,
        bs=config.batch_size,
        image_size=config.img_size,
        augment=config.augment,
    )
    # metrics = [COCOMetric(metric_type=COCOMetricType.bbox)]
    metrics = [
        COCOMetric_Chair(),
        COCOMetric_Couch(),
        COCOMetric_TV(),
        COCOMetric_Remote(),
        COCOMetric_Book(),
        COCOMetric_Vase(),
        COCOMetric(metric_type=COCOMetricType.bbox),
    ]
    model_type, backbone, extra_args = select_model(config.arch, config.img_size)

    # Instantiate the model
    model = model_type.model(
        backbone=backbone(pretrained=True),
        num_classes=len(parser.class_map),
        **extra_args,
    )
    # Data Loaders
    train_dl = model_type.train_dl(
        train_ds, batch_size=config.batch_size, num_workers=10, shuffle=True
    )
    valid_dl = model_type.valid_dl(
        valid_ds, batch_size=config.batch_size, num_workers=4, shuffle=False
    )
    learn = model_type.fastai.learner(
        dls=[train_dl, valid_dl],
        model=model,
        metrics=metrics,
        cbs=[
            WandbCallback(log_dataset=True, log_model=True),
            SaveModelCallback(fname=f"run-{wandb.run.id}-model", monitor="COCOMetric"),
        ],
    )
    learn.fine_tune(config.epochs, config.lr, freeze_epochs=1)

    # Infer
    infer_dl = model_type.infer_dl(
        valid_ds, batch_size=config.batch_size, shuffle=False
    )
    preds = model_type.predict_from_dl(model, infer_dl, keep_images=True)
    wandb_images = wandb_img_preds(preds, add_ground_truth=True)
    wandb.log({"Predicted images": wandb_images})
    wandb.finish()


if __name__ == "__main__":
    parse_args()
    train(train_config)
