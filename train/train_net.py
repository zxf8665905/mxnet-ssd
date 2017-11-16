import tools.find_mxnet
import mxnet as mx
import logging
import sys
import os
import importlib
import re
from dataset.iterator import DetRecordIter
from train.metric import MultiBoxMetric
from evaluate.eval_metric import MApMetric, VOC07MApMetric
from config.config import cfg
from symbol.symbol_factory import get_symbol_train
from mxnet import metric
from mxnet.initializer import Uniform
from mxnet.model import BatchEndParam
import time

def convert_pretrained(name, args):
    """
    Special operations need to be made due to name inconsistance, etc

    Parameters:
    ---------
    name : str
        pretrained model name
    args : dict
        loaded arguments

    Returns:
    ---------
    processed arguments as dict
    """
    return args

def get_lr_scheduler(learning_rate, lr_refactor_step, lr_refactor_ratio,
                     num_example, batch_size, begin_epoch):
    """
    Compute learning rate and refactor scheduler

    Parameters:
    ---------
    learning_rate : float
        original learning rate
    lr_refactor_step : comma separated str
        epochs to change learning rate
    lr_refactor_ratio : float
        lr *= ratio at certain steps
    num_example : int
        number of training images, used to estimate the iterations given epochs
    batch_size : int
        training batch size
    begin_epoch : int
        starting epoch

    Returns:
    ---------
    (learning_rate, mx.lr_scheduler) as tuple
    """
    assert lr_refactor_ratio > 0
    iter_refactor = [int(r) for r in lr_refactor_step.split(',') if r.strip()]
    if lr_refactor_ratio >= 1:
        return (learning_rate, None)
    else:
        lr = learning_rate
        epoch_size = num_example // batch_size
        for s in iter_refactor:
            if begin_epoch >= s:
                lr *= lr_refactor_ratio
        if lr != learning_rate:
            logging.getLogger().info("Adjusted learning rate to {} for epoch {}".format(lr, begin_epoch))
        steps = [epoch_size * (x - begin_epoch) for x in iter_refactor if x > begin_epoch]
        if not steps:
            return (lr, None)
        lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(step=steps, factor=lr_refactor_ratio)
        return (lr, lr_scheduler)

def _as_list(obj):
    """A utility function that converts the argument to a list if it is not already.

    Parameters
    ----------
    obj : object

    Returns
    -------
    If `obj` is a list or tuple, return it. Otherwise, return `[obj]` as a
    single-element list.

    """
    if isinstance(obj, (list, tuple)):
        return obj
    else:
        return [obj]

def fit(model, train_data, eval_data=None, eval_metric='acc',
        epoch_end_callback=None, batch_end_callback=None, kvstore='local',
        optimizer='sgd', optimizer_params=(('learning_rate', 0.01),),
        eval_end_callback=None,
        eval_batch_end_callback=None, initializer=Uniform(0.01),
        arg_params=None, aux_params=None, allow_missing=False,
        force_rebind=False, force_init=False, begin_epoch=0, num_epoch=None,
        validation_metric=None, monitor=None):

    assert num_epoch is not None, 'please specify number of epochs'

    model.bind(data_shapes=[('data',(1,3,300,300))],
               label_shapes=[('loc_cls_label',((1,58,6))), ('orien_label', (1,58,1))],
              for_training=True, force_rebind=force_rebind)

    model.init_params(initializer=initializer, arg_params=arg_params, aux_params=aux_params,
                     allow_missing=allow_missing, force_init=force_init)
    model.init_optimizer(kvstore=kvstore, optimizer=optimizer,
                        optimizer_params=optimizer_params)



    ################################################################################
    # training loop
    ################################################################################
    for epoch in range(begin_epoch, num_epoch):
        tic = time.time()
        nbatch = 0
        data_batch = None # todo
        model.forward_backward(data_batch)
        model.update()

        nbatch += 1

        toc = time.time()
        model.logger.info('Epoch[%d] Time cost=%.3f', epoch, (toc-tic))

        # sync aux params across devices
        arg_params, aux_params = model.get_params()
        model.set_params(arg_params, aux_params)


def train_net(net, train_path, num_classes, batch_size,
              data_shape, mean_pixels, resume, finetune, pretrained, epoch,
              prefix, ctx, begin_epoch, end_epoch, frequent, learning_rate,
              momentum, weight_decay, lr_refactor_step, lr_refactor_ratio,
              freeze_layer_pattern='',
              num_example=10000, label_pad_width=350,
              nms_thresh=0.45, force_nms=False, ovp_thresh=0.5,
              use_difficult=False, class_names=None,
              voc07_metric=False, nms_topk=400, force_suppress=False,
              train_list="", val_path="", val_list="", iter_monitor=0,
              monitor_pattern=".*", log_file=None):
    """
    Wrapper for training phase.

    Parameters:
    ----------
    net : str
        symbol name for the network structure
    train_path : str
        record file path for training
    num_classes : int
        number of object classes, not including background
    batch_size : int
        training batch-size
    data_shape : int or tuple
        width/height as integer or (3, height, width) tuple
    mean_pixels : tuple of floats
        mean pixel values for red, green and blue
    resume : int
        resume from previous checkpoint if > 0
    finetune : int
        fine-tune from previous checkpoint if > 0
    pretrained : str
        prefix of pretrained model, including path
    epoch : int
        load epoch of either resume/finetune/pretrained model
    prefix : str
        prefix for saving checkpoints
    ctx : [mx.cpu()] or [mx.gpu(x)]
        list of mxnet contexts
    begin_epoch : int
        starting epoch for training, should be 0 if not otherwise specified
    end_epoch : int
        end epoch of training
    frequent : int
        frequency to print out training status
    learning_rate : float
        training learning rate
    momentum : float
        trainig momentum
    weight_decay : float
        training weight decay param
    lr_refactor_ratio : float
        multiplier for reducing learning rate
    lr_refactor_step : comma separated integers
        at which epoch to rescale learning rate, e.g. '30, 60, 90'
    freeze_layer_pattern : str
        regex pattern for layers need to be fixed
    num_example : int
        number of training images
    label_pad_width : int
        force padding training and validation labels to sync their label widths
    nms_thresh : float
        non-maximum suppression threshold for validation
    force_nms : boolean
        suppress overlaped objects from different classes
    train_list : str
        list file path for training, this will replace the embeded labels in record
    val_path : str
        record file path for validation
    val_list : str
        list file path for validation, this will replace the embeded labels in record
    iter_monitor : int
        monitor internal stats in networks if > 0, specified by monitor_pattern
    monitor_pattern : str
        regex pattern for monitoring network stats
    log_file : str
        log to file if enabled
    """
    # set up logger
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if log_file:
        fh = logging.FileHandler(log_file)
        logger.addHandler(fh)

    # check args
    if isinstance(data_shape, int):
        data_shape = (3, data_shape, data_shape)
    assert len(data_shape) == 3 and data_shape[0] == 3
    if prefix.endswith('_'):
        prefix += '_' + str(data_shape[1])

    if isinstance(mean_pixels, (int, float)):
        mean_pixels = [mean_pixels, mean_pixels, mean_pixels]
    assert len(mean_pixels) == 3, "must provide all RGB mean values"
    #
    # train_iter = DetRecordIter(train_path, batch_size, data_shape, mean_pixels=mean_pixels,
    #     label_pad_width=label_pad_width, path_imglist=train_list, **cfg.train)
    #
    # if val_path:
    #     val_iter = DetRecordIter(val_path, batch_size, data_shape, mean_pixels=mean_pixels,
    #         label_pad_width=label_pad_width, path_imglist=val_list, **cfg.valid)
    # else:
    #     val_iter = None
    train_iter,val_iter = None, None

    # load symbol
    net = get_symbol_train(net, data_shape[1], num_classes=num_classes,
        nms_thresh=nms_thresh, force_suppress=force_suppress, nms_topk=nms_topk)

    # define layers with fixed weight/bias
    if freeze_layer_pattern.strip():
        re_prog = re.compile(freeze_layer_pattern)
        fixed_param_names = [name for name in net.list_arguments() if re_prog.match(name)]
    else:
        fixed_param_names = None

    # load pretrained or resume from previous state
    ctx_str = '('+ ','.join([str(c) for c in ctx]) + ')'
    if resume > 0:
        logger.info("Resume training with {} from epoch {}"
            .format(ctx_str, resume))
        _, args, auxs = mx.model.load_checkpoint(prefix, resume)
        begin_epoch = resume
    elif finetune > 0:
        logger.info("Start finetuning with {} from epoch {}"
            .format(ctx_str, finetune))
        _, args, auxs = mx.model.load_checkpoint(prefix, finetune)
        begin_epoch = finetune
        # check what layers mismatch with the loaded parameters
        exe = net.simple_bind(mx.cpu(), data=(1, 3, 300, 300), label=(1, 1, 5), grad_req='null')
        arg_dict = exe.arg_dict
	fixed_param_names = []
        for k, v in arg_dict.items():
            if k in args:
                if v.shape != args[k].shape:
                    del args[k]
                    logging.info("Removed %s" % k)
                else:
		    if not 'pred' in k:
		    	fixed_param_names.append(k)
    elif pretrained:
        logger.info("Start training with {} from pretrained model {}"
            .format(ctx_str, pretrained))
        _, args, auxs = mx.model.load_checkpoint(pretrained, epoch)
        args = convert_pretrained(pretrained, args)
    else:
        logger.info("Experimental: start training from scratch with {}"
            .format(ctx_str))
        args = None
        auxs = None
        fixed_param_names = None

    # helper information
    if fixed_param_names:
        logger.info("Freezed parameters: [" + ','.join(fixed_param_names) + ']')

    # init training module
    mod = mx.mod.Module(net, label_names=('loc_cls_label','orien_label'), logger=logger, context=ctx,
                        fixed_param_names=fixed_param_names)

    # fit parameters
    # batch_end_callback = mx.callback.Speedometer(train_iter.batch_size, frequent=frequent)
    # epoch_end_callback = mx.callback.do_checkpoint(prefix)
    batch_end_callback = None
    epoch_end_callback = None
    learning_rate, lr_scheduler = get_lr_scheduler(learning_rate, lr_refactor_step,
        lr_refactor_ratio, num_example, batch_size, begin_epoch)
    optimizer_params={'learning_rate':learning_rate,
                      'momentum':momentum,
                      'wd':weight_decay,
                      'lr_scheduler':lr_scheduler,
                      'clip_gradient':None,
                      'rescale_grad': 1.0 / len(ctx) if len(ctx) > 0 else 1.0 }
    monitor = mx.mon.Monitor(iter_monitor, pattern=monitor_pattern) if iter_monitor > 0 else None

    # run fit net, every n epochs we run evaluation network to get mAP
    if voc07_metric:
        valid_metric = VOC07MApMetric(ovp_thresh, use_difficult, class_names, pred_idx=3)
    else:
        valid_metric = MApMetric(ovp_thresh, use_difficult, class_names, pred_idx=3)
    #
    # mod.fit(train_iter,
    #     val_iter,
    #     eval_metric=MultiBoxMetric(),
    #     validation_metric=valid_metric,
    #     batch_end_callback=batch_end_callback,
    #     epoch_end_callback=epoch_end_callback,
    #     optimizer='sgd',
    #     optimizer_params=optimizer_params,
    #     begin_epoch=begin_epoch,
    #     num_epoch=end_epoch,
    #     initializer=mx.init.Xavier(),
    #     arg_params=args,
    #     aux_params=auxs,
    #     allow_missing=True,
    #     monitor=monitor)

    fit(mod,train_iter,
        val_iter,
        eval_metric=MultiBoxMetric(),
        validation_metric=valid_metric,
        batch_end_callback=batch_end_callback,
        epoch_end_callback=epoch_end_callback,
        optimizer='sgd',
        optimizer_params=optimizer_params,
        begin_epoch=begin_epoch,
        num_epoch=end_epoch,
        initializer=mx.init.Xavier(),
        arg_params=args,
        aux_params=auxs,
        allow_missing=True,
        monitor=monitor)
