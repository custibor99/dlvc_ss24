import matplotlib.pyplot as plt

def display_images_and_masks(dataset, indexes):
    """Display images and segmentation masks next to each other.
    """
    # Display a maximum of 2 sets of (image, mask) pairs per row.
    nrows = (len(indexes) + 1) // 2
    # 3 units height per row.
    fig = plt.figure(figsize=(10, 3 * nrows))
    for i in range(len(indexes)):
        image, mask = dataset[i][0], dataset[i][1]
        fig.add_subplot(nrows, 4, i*2+1)
        plt.imshow(image.permute(1,2,0))
        plt.axis("off")
        
        fig.add_subplot(nrows, 4, i*2+2)
        plt.imshow(mask.permute(1,2,0))
        plt.axis("off")
    # end for