def eval_metrics(metrics):
    # source
    # ultralytics.utils.metrics.DetMetrics
    """
    args
    ----
        metrics: dict
            metrics from model.val()

    return
    ------
        json: dict
    """
    # metrics
    map5095 = metrics.box.map.round(4)
    map50 = metrics.box.map50.round(4)
    precision = metrics.box.p[0].round(4)
    recall = metrics.box.r[0].round(4)
    f1 = metrics.box.f1[0].round(4)
    # confusion matrix
    conf_mat = metrics.confusion_matrix.matrix  # conf=0.25, iou_thres=0.45
    n_all = conf_mat[:, 0].sum()
    n_fn = conf_mat[1, 0].sum()
    n_fp = conf_mat[0, 1].sum()
    # write json
    json_out = dict(
        map5095=map5095,
        map50=map50,
        precision=precision,
        recall=recall,
        f1=f1,
        n_all=int(n_all),
        n_fn=int(n_fn),  # false negative
        n_fp=int(n_fp),
    )
    return json_out