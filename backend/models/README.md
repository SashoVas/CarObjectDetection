# CarObjectDetection

## Model Metrics

| Dataset                                    | Model File         | box_loss(train) | cls_loss(train) | dfl_loss(train) | Precision (P) | Recall (R)| mAP50 | mAP50-95 | Train images|Time per epoch |
|---------------------------------------------|--------------------|----------|----------|----------|---------------|------------|-------|----------|------|----------|
| TrafficDetectionProjectDataset              | 70_epochs.pt       | 0.5923   | 0.3237   | 0.8353   |    0.94       |   0.907    |  0.95 |  0.781   |5 805 |2 min|
| TrafficRoadObjectDetectionPolish12kDataset  | 40_epochs.pt       | 0.7864   | 0.4288   | 0.8576   |    0.833      |   0.704    | 0.765 |  0.464   |10 500|4 min 30 sec|
| TrafficRoadObjectDetectionPolish12kDataset  | 50_epochs.pt       | 0.7074   | 0.3897   | 0.8429   |    0.838      |   0.7      | 0.767 |  0.477   |10 500|4 min 30 sec|
| TrafficRoadObjectDetectionPolish12kDataset  | 60_epochs.pt       | 0.6371   | 0.355    | 0.8305   |    0.847      |   0.707     | 0.775 |  0.48   |10 500|4 min 30 sec|
| TrafficRoadObjectDetectionPolish12kDataset  | 70_epochs.pt       | 0.5132   | 0.2903    | 0.8029   |   0.833      |   0.705     | 0.77  |  0.477  |10 500|4 min 30 sec|



Each subfolder in the `models` directory corresponds to a dataset, and each row in the table summarizes the metrics for a model checkpoint in that dataset.