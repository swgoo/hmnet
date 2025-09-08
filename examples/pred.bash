python examples/squad.py predict configs/hmnet_3stage_XL_MW.yaml --ckpt-path ckpts/HMNet-squad-hmnet_3stage_XL_MW-epoch=10-val_loss=0.28.ckpt &&\
python examples/squad.py predict configs/hmnet_3stage_XL_SW.yaml --ckpt-path ckpts/HMNet-squad-hmnet_3stage_XL_SW-epoch=10-val_loss=0.28.ckpt &&\
python examples/squad.py predict configs/hmnet_3stage_XL_XXLW_XLS.yaml --ckpt-path ckpts/HMNet-squad-hmnet_3stage_XL_XXLW_XLS-epoch=18-val_loss=0.25.ckpt&&\
python examples/squad.py predict configs/hnet_3stage_XL.yaml --ckpt-path ckpts/HNet-squad-hnet_3stage_XL-epoch=08-val_loss=0.20.ckpt --model-type HNet