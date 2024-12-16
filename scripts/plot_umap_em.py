import seaborn as sns
import umap
import matplotlib.pyplot as plt
import scripts.embd_fgit as embd
import pandas as pd
import numpy as np
import time
# Plotting with seaborn (if you prefer more sophisticated plots)
cp = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

def umap_plt(train_emb, val_emb, test_emb, ny_train, ny_val, ny_test, name):
    
    #to get the umap embedding
    train_um, val_um, test_um = embd.umap_embedding(train_emb, val_emb, test_emb)
    embedding=test_um
    
    # Ensure all classes are in the legend by specifying hue_order
    hue_order = sorted(np.unique(ny_test.values))
    plt.figure(figsize=(12, 8))
    
    
    # Adjusting the text size for the plot
    sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=ny_test.values.flatten(), palette=cp,s=50,hue_order=hue_order)
                    #"Spectral", s=50,hue_order=hue_order)
    title_name='UMAP Projection of'+' '+name
    # Set the title with a larger font size
    plt.title(title_name, fontsize=18)
    
    # Adjust the legend text size
    # Adjust the legend text size and place it at the bottom
    plt.legend(title='Classes', title_fontsize='13', fontsize='12', loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=10)
    
    
    # Save the figure
    plt.savefig(title_name+".png", format="png", dpi=600, bbox_inches="tight")
    
    # Display the plot
    plt.show()
