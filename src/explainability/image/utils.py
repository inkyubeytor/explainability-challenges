def prepare_image_for_plotting(image):
    # If the image is a PyTorch tensor, convert it to a NumPy array
    if torch.is_tensor(image):
        image = image.detach().cpu().numpy()
        if image.dtype.kind == 'f':
            if image.max() > 1:
                image = (image * 255).astype(np.uint8)
            else:
                image = (image * 255).round().astype(np.uint8)
        elif image.dtype.kind == 'u':
            if image.max() > 255:
                raise ValueError('Image with dtype uint8 must have values in range 0-255')
    # If the image is a NumPy array, make sure the data type is either float or uint8
    elif isinstance(image, np.ndarray):
        if image.dtype.kind == 'f':
            if image.max() > 1:
                image = (image * 255).astype(np.uint8)
            else:
                image = (image * 255).round().astype(np.uint8)
        elif image.dtype.kind == 'u':
            if image.max() > 255:
                raise ValueError('Image with dtype uint8 must have values in range 0-255')
        else:
            raise ValueError('Image dtype must be float or uint8')
    else:
        raise ValueError('Image type must be a PyTorch tensor or a NumPy array')

    # If the image has channels first, move the channels to the last dimension
    if image.shape[0] < image.shape[-1]:
        image = np.moveaxis(image, 0, -1)

    return image


def plot_images(images, captions=None):
    # Determine the grid size based on the number of images
    grid_size = math.ceil(math.sqrt(len(images)))
    fig, axes = plt.subplots(nrows=grid_size, ncols=grid_size)

    # Make sure axes is always a 2D array, even when grid_size is 1
    if grid_size == 1:
        axes = np.array([[axes]])

    # Loop over the grid and plot the images
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i*grid_size + j
            if idx < len(images):
                axes[i, j].imshow(images[idx])
                axes[i, j].axis('off')  # Turn off the axis labels
                if captions is not None and idx < len(captions):
                    axes[i, j].set_title(captions[idx])  # Set the caption as title
            else:
                axes[i, j].axis('off')  # Turn off the axis if there's no image

    # Adjust the spacing between subplots
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    # Show the plot
    plt.show()

