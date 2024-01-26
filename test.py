import sys
import torch
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
from ReLA2.Inference import GRES_Inference
from ReLA2.model_instance import get_model, raw_data2feature

def demo_infer(image, text_input):
    image_path = '/mnt/local/wwx/data/GRES/tmp_test/' + image
    image = Image.open(image_path).convert("RGB")
    image = np.array(image)
    
    focus_dict = {}
    focus_dict['image'] = image
    focus_dict['text_input'] = text_input

    focus_dict = gres_model.gres_data2feature(focus_dict)

    res = gres_model.infer(focus_dict)
    print(res)
    print("=====" * 20)
    
    resize_image, focus_dict = gres_model.infer(focus_dict)
    
    print(focus_dict['mask'][0].cpu())
    
    focus_image = resize_image * focus_dict['mask'][0].cpu()

    # Start the plot
    fig, axs = plt.subplots(1, 2, figsize=(10,5))  # 1 row, 2 columns

    # Display raw image on the left
    axs[0].imshow(resize_image.to("cpu").permute(1,2,0).detach().numpy())
    axs[0].set_title(text_input)
    axs[0].axis('off')  # to hide axis values

    # Display target mask on the right
    axs[1].imshow(focus_image.to("cpu").permute(1,2,0).detach().numpy())
    axs[1].set_title("target mask")
    axs[1].axis('off')  # to hide axis values

#     plt.tight_layout()
    plt.show()

    return


gres_model = GRES_Inference()

image = "test_00.png"
query = "What are the letters on a green apple?"
demo_infer(image, query)
