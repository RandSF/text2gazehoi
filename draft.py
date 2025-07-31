# from constants.hot3d_constants import hot3d_obj_name
# from lib.datasets.hot3d import SequenceHOT3D, ContactHOT3D, MotionHOT3D

# if __name__ == '__main__':
#     # Dataset = SequenceHOT3D
#     # Dataset = ContactHOT3D
#     Dataset = MotionHOT3D
#     ds = Dataset(
#         data_path="data/hot3d/data.npz", 
#         data_obj_pc_path="data/hot3d/obj.pkl", 
#         text_json="data/hot3d/text.json", 
#         obj_name=hot3d_obj_name, 
#         max_nframes=300, 
#         data_ratio=1.0, 
#         augm=False, 
#     )
#     for i in range(2):
#         print("="*22)
#         for k, v in ds[i].items():
#             try:
#                 print(f"{k}: {v.shape}, {v.dtype}")
#             except:
#                 print(f"{k}: {v}")

#     import matplotlib.pyplot as plt

#     plt.imshow(ds[0]['cov_map'][...,0])
#     plt.show()