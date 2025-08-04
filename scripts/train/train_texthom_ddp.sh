# python train/train_texthom.py dataset=h2o dataset.augm=False texthom.iteration=90000
# python train/train_texthom.py dataset=grab
# python train/train_texthom.py dataset=arctic texthom.obj_nfeats=10
CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun --nproc_per_node=1 train/train_texthom_ddp.py dataset=hot3d dataset.augm=False texthom.iteration=90000