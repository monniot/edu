import wandb
import torchvision.models as tvmodels
import pandas as pd
from icevision.all import *

import params

from train import (
    select_model,
    download_data,
    get_data,
    COCOMetric_Chair,
    COCOMetric_Couch,
    COCOMetric_TV,
    COCOMetric_Remote,
    COCOMetric_Book,
    COCOMetric_Vase,
)

run = wandb.init(
    project=params.WANDB_PROJECT,
    entity=params.ENTITY,
    job_type="evaluation",
    tags=["staging"],
)

artifact = run.use_artifact("pmon/model-registry/assignment2-model:v0", type="model")

artifact_dir = Path(artifact.download())

_model_pth = artifact_dir.ls()[0]
model_path = _model_pth.parent.absolute() / _model_pth.stem

producer_run = artifact.logged_by()
wandb.config.update(producer_run.config)
config = wandb.config


def eval(config, processed_dataset_dir=None):
    set_seed(config.seed, reproducible=True)
    if processed_dataset_dir is None:
        processed_dataset_dir = download_data()
    _, _, parser, test_ds = get_data(
        processed_dataset_dir,
        bs=config.batch_size,
        image_size=config.img_size,
        augment=config.augment,
    )
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
    model = model_type.model(
        backbone=backbone(pretrained=True),
        num_classes=len(parser.class_map),
        **extra_args,
    )
    # Data Loaders
    # TODO: Change to have test instead
    test_dl = model_type.train_dl(
        test_ds, batch_size=config.batch_size, num_workers=10, shuffle=True
    )

    learn = model_type.fastai.learner(
        dls=[test_ds],
        model=model,
        pretrained=config.pretrained,
        metrics=metrics,
    )
    learn.load(model_path)
